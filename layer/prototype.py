import os
import copy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.cluster import KMeans

class Prototype_Layer(nn.Module):
    def __init__(
        self, prototype_dim, input_length, output_dim, 
        init_prototype_num, prototype_split_num, top_k_activate, split_ratio
    ):
        super(Prototype_Layer, self).__init__()

        self.prototype_dim = prototype_dim
        self.split_num = prototype_split_num
        self.input_length = input_length
        self.output_dim = output_dim
        self.prototype_num = [init_prototype_num]
        self.layer_num = 1
        self.top_k = top_k_activate
        self.split_ratio = split_ratio

        self.layer_bias = nn.ModuleList()
        self.layer_bias.append(
            nn.Sequential(
                nn.Linear(self.prototype_dim, self.prototype_dim * 2, bias=False),
                nn.ReLU(),
                nn.Linear(self.prototype_dim * 2, init_prototype_num, bias=False)
            )
        )

        self.prototype_embedding = nn.ParameterList()
        self.prototype_embedding.append(
            nn.Parameter(
                torch.zeros(init_prototype_num, self.prototype_dim)
            )
        )

        self.prototype_value = nn.ParameterList()
        self.prototype_value.append(
            nn.Parameter(
                torch.randn(init_prototype_num, self.input_length, self.output_dim)
            )
        )

        # ------------------------------------------------------------------
        # C2 architecture: matching key = matching_head(encoder.mean(dim=1)).
        # The matching_head is a single Linear(D, D) with NO bias and
        # initialized to identity, so at training start match_key == time_mean
        # (preserving the C1 property that μ₀ lives in the encoder's feature
        # space) but the layer can drift away from identity during joint
        # training to absorb any encoder/prototype coordinate-system tension.
        # This restores the role the original ``patch_embedding`` played as an
        # adapter, without sacrificing the LLM-init coordinate alignment.
        # ------------------------------------------------------------------
        self.matching_head = nn.Linear(self.prototype_dim,
                                        self.prototype_dim, bias=False)
        # NOTE: identity init must be applied AFTER _init_linear_layers()
        # below, otherwise the Kaiming-uniform pass in that helper will
        # overwrite the identity. See bottom of __init__.

        # Original ``patch_embedding`` (kept for reference; do not enable):
        # self.patch_embedding = nn.Sequential(
        #     nn.Linear(self.input_length, self.input_length * 2, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(self.input_length * 2, self.prototype_dim, bias=False)
        # )

        self.parent_indices = []
        self.cumulative_loss = []
        self.activation_count = []

        self._init_linear_layers()

        # matching_head identity init AFTER _init_linear_layers, so the
        # generic Kaiming-uniform pass over all Linear modules doesn't
        # clobber it. At training start, matching_head(x) == x.
        nn.init.eye_(self.matching_head.weight)

        self._init_stats(init_prototype_num)

    def _init_linear_layers(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        for param in self.prototype_embedding:
            if param.dim() > 1:
                nn.init.kaiming_uniform_(param, nonlinearity='relu')
            else:
                nn.init.zeros_(param)
    
    def _init_stats(self, num):
        self.cumulative_loss.append(torch.zeros(num))
        self.activation_count.append(torch.zeros(num))

    # ------------------------------------------------------------------
    # Hybrid-training hooks (LLM-guided anchoring for layer-0 prototypes)
    # ------------------------------------------------------------------
    def set_initial_prototypes(self, embedding, value, anchor_mask):
        """Overwrite layer-0 μ/p with the LLM-initialized tensors, and
        register anchor buffers so we can regularize later prototypes back
        toward them during training.

        Args:
            embedding:   Tensor of shape (P, D)
            value:       Tensor of shape (P, L, O)
            anchor_mask: Tensor of shape (P,) with 1.0 for LLM-initialized
                         prototypes (to be anchored) and 0.0 for "blank"
                         prototypes (random init, free to evolve).
        """
        device = self.prototype_embedding[0].device
        embedding = embedding.to(device).to(self.prototype_embedding[0].dtype)
        value = value.to(device).to(self.prototype_value[0].dtype)
        anchor_mask = anchor_mask.to(device).float()

        with torch.no_grad():
            self.prototype_embedding[0].data.copy_(embedding)
            self.prototype_value[0].data.copy_(value)

        # register_buffer() gives the anchors free ride through .to(device)
        # and state_dict but keeps them out of the optimizer.
        self.register_buffer("anchor_embedding", embedding.detach().clone(),
                             persistent=False)
        self.register_buffer("anchor_value", value.detach().clone(),
                             persistent=False)
        self.register_buffer("anchor_mask", anchor_mask.detach().clone(),
                             persistent=False)

    def anchor_loss(self, lambda_emb=1.0, lambda_val=1.0):
        """Hybrid-training regularizer.

        Pulls layer-0 prototypes back toward their LLM-initialized
        (μ₀, p₀) anchors, *only* for prototypes whose anchor_mask==1.
        Blank prototypes (mask==0) contribute nothing, so they can adapt
        to local details unconstrained.

        The returned scalar is mean-over-anchored-prototypes so it doesn't
        grow with N.
        """
        if not hasattr(self, "anchor_mask"):
            # No anchors have been registered → no-op so training still works.
            return self.prototype_embedding[0].new_zeros(())

        mask = self.anchor_mask
        if mask.sum() == 0:
            return self.prototype_embedding[0].new_zeros(())

        mu  = self.prototype_embedding[0]
        mu0 = self.anchor_embedding
        p   = self.prototype_value[0]
        p0  = self.anchor_value

        mse_mu = ((mu - mu0) ** 2).mean(dim=-1)          # (P,)
        mse_p  = ((p  - p0)  ** 2).mean(dim=(1, 2))      # (P,)

        per_proto = lambda_emb * mse_mu + lambda_val * mse_p
        return (mask * per_proto).sum() / mask.sum().clamp(min=1.0)

    def clear_error_statistics(self):
        curr_layer = self.layer_num - 1
        self.cumulative_loss[curr_layer].fill_(0)
        self.activation_count[curr_layer].fill_(0)

    def update_error_statistics(self, all_score, all_ans, target):
        batch = target.shape[0]

        curr_layer = self.layer_num - 1
        scores = all_score[curr_layer].squeeze(1) 
        preds = all_ans[curr_layer]
        
        loss = torch.abs(preds - target).mean(dim=[1, 2])
        
        _, topk_indices = torch.topk(scores, min(self.top_k, scores.shape[1]), dim=-1)

        for b in range(batch):
            for idx in topk_indices[b]:
                self.cumulative_loss[curr_layer][idx] += loss[b].item()
                self.activation_count[curr_layer][idx] += 1


    def forward(self, x):
        # x: (B, length=output_length, embedding_dim=prototype_dim)
        # The encoder produces a (B, 96, 128) tensor where dim 1 is time and
        # dim 2 is the encoder feature axis. We collapse over time first
        # (time_mean), then pass through the matching_head adapter to obtain
        # match_key — the prototype-side adapter that lets matching geometry
        # drift away from the encoder's natural feature shape during training,
        # without breaking the LLM-init coordinate system at epoch 0
        # (matching_head is identity-initialized).
        batch, length, embedding_dim = x.shape
        time_mean = x.mean(dim=1, keepdim=True)              # (B, 1, D)
        match_key = self.matching_head(time_mean)             # (B, 1, D), identity at init

        # --- original patch_embedding path (kept for reference) -----------
        # x_flat = x.mean(dim=-1)                              # (B, L)
        # patch_emb = self.patch_embedding(x_flat).unsqueeze(1) # (B, 1, prototype_dim)
        # then ``patch_emb`` was used everywhere ``match_key`` is used below.
        # ------------------------------------------------------------------

        all_score = []
        all_ans = []

        for i in range(self.layer_num):
            raw_score = torch.einsum('...ij,...jk->...ik', match_key,
                                     self.prototype_embedding[i].unsqueeze(0).expand(batch, -1, -1).transpose(1, 2))

            if i == 0:
                score = raw_score.softmax(dim=-1)
            else:
                local_score = self._compute_local_softmax(raw_score, i)

                parent_idx = self.parent_indices[i-1]
                parent_score = all_score[i-1][:, :, parent_idx]
                score = parent_score * local_score

            all_score.append(score)

            value_bias = torch.abs(self.layer_bias[i](match_key))
            value_bias = value_bias.reshape(batch, self.prototype_num[i], 1, 1)
            weighted_prototype = value_bias * self.prototype_value[i].unsqueeze(0)

            ans = (score.reshape(batch, -1, 1, 1) * weighted_prototype).sum(dim=1)
            all_ans.append(ans)

        return all_ans, all_score

    def _compute_local_softmax(self, raw_score, layer_idx):

        parent_map = self.parent_indices[layer_idx-1]
        unique_parents = sorted(list(set(parent_map)))
        
        local_softmax_score = torch.zeros_like(raw_score)
        for p in unique_parents:
            child_indices = [idx for idx, p_val in enumerate(parent_map) if p_val == p]
            subset = raw_score[:, :, child_indices]
            local_softmax_score[:, :, child_indices] = subset.softmax(dim=-1)
            
        return local_softmax_score

    def add_prototype(self, device):

        curr_layer = self.layer_num - 1
        
        counts = self.activation_count[curr_layer]
        losses = self.cumulative_loss[curr_layer]
        norm_loss = torch.where(counts > 0, losses / counts, torch.zeros_like(losses))

        num_to_split = max(1, int(len(norm_loss) * self.split_ratio))
        _, split_indices = torch.topk(norm_loss, num_to_split)
        
        last_emb = self.prototype_embedding[-1].detach()
        last_val = self.prototype_value[-1].detach()
        
        split_set = set(split_indices.tolist())
        
        new_total = len(split_set) * self.split_num + (self.prototype_num[-1] - len(split_set))
        print(f"new_total: {new_total}")
        
        new_emb_list, new_val_list = [], []
        new_parent_map = [] 

        old_bias_net = self.layer_bias[-1]
        old_weight = old_bias_net[-1].weight.data 
        print(f"old_weight.shape: {old_weight.shape}")
        new_bias_net = nn.Sequential(
            nn.Linear(self.prototype_dim, self.prototype_dim * 2, bias=False),
            nn.ReLU(),
            nn.Linear(self.prototype_dim * 2, new_total, bias=False)
        ).to(device)
        new_bias_net[0].load_state_dict(old_bias_net[0].state_dict())
        new_weight = torch.zeros(new_total, self.prototype_dim * 2).to(device)
        print(f"new_weight.shape: {new_weight.shape}")

        curr_idx = 0
        for i in range(self.prototype_num[-1]):
            noise = torch.randn_like(last_emb[i:i+1]) * 1e-3
            if i in split_set:
                new_emb_list.append(last_emb[i:i+1].repeat(self.split_num, 1) + noise)
                new_val_list.append(last_val[i:i+1].repeat(self.split_num, 1, 1))
                new_parent_map.extend([i] * self.split_num)
                new_weight[curr_idx:curr_idx+self.split_num] = old_weight[i] 
                curr_idx += self.split_num
            else:
                new_emb_list.append(last_emb[i:i+1] + noise)
                new_val_list.append(last_val[i:i+1])
                new_parent_map.append(i)
                new_weight[curr_idx] = old_weight[i]
                curr_idx += 1

        self.parent_indices.append(new_parent_map)
        
        self.prototype_embedding.append(nn.Parameter(torch.cat(new_emb_list, dim=0)).to(device))
        self.prototype_value.append(nn.Parameter(torch.cat(new_val_list, dim=0)).to(device))
        self.prototype_num.append(new_total)
        self._init_stats(new_total)
        
        new_bias_net[-1].weight.data = new_weight
        self.layer_bias.append(new_bias_net)
        
        self.layer_num += 1
        
        print(f"Layer {self.layer_num}: Selected {split_set} to split. Total prototypes: {self.prototype_num}")
