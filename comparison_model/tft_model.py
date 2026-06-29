"""
Temporal Fusion Transformer (Lim et al. 2019, https://arxiv.org/abs/1912.09363)
adapted as a FAIR baseline for ProtoLLM's Concept Bottleneck Model on the
Beijing PRSA Aotizhongxin PM2.5 task.

Why this re-implementation
--------------------------
The upstream PyTorch re-implementation in ../Temporal_Fusion_Transform/ is
written batch-second, uses BatchNorm over time (which is unstable for
variable-length sequences and irrelevant for a single fixed window), restricts
attention to "decoder queries onto encoder keys only" rather than the causal
self-attention over the full sequence used in the original paper, and assumes
static covariates exist. PRSA has no static covariates and ProtoLLM's CBM
uses causal interpretable attention over the whole window. To compare
architectures fairly we therefore re-implement TFT cleanly:

  * batch_first throughout, LayerNorm in every gated residual block
  * VariableSelectionNetwork supports zero static context (no-op gracefully)
  * LSTM seq2seq -- decoder hidden state initialised with encoder final state
  * Static enrichment path becomes identity when there are no statics
  * Causal Interpretable Multi-Head Attention over (L+H) tokens with the
    SHARED value projection from the paper -- the heads disagree on attention
    weights but compute a single (B, T, d) value, so the average head weight
    is a faithful per-timestep importance score
  * Single point output, MSE-trained, so test MAE is directly comparable to
    TiDE and to ProtoLLM's CBM (all live in the same StandardScaler z-space)

forward signature
-----------------
    forward(y_past_tgt, x_real_past, x_real_future, x_cat_past, x_cat_future)
                                     -> y_hat,  diagnostics_dict
        y_past_tgt    (B, L, 1)
        x_real_past   (B, L, R)
        x_real_future (B, H, R)
        x_cat_past    (B, L, K)  int64
        x_cat_future  (B, H, K)  int64
        y_hat         (B, H, 1)  scaled forecast in the same z-space as y_past_tgt
        diagnostics   dict with VSN sparse weights and attention weights for
                                future interpretability inspection
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class GatedLinearUnit(nn.Module):
    """y = sigmoid(W1 x) * (W2 x). Optional input dropout for regularization."""

    def __init__(self, input_size: int, output_size: Optional[int] = None,
                 dropout: float = 0.0):
        super().__init__()
        output_size = output_size or input_size
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(input_size, output_size * 2)
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        a, b = self.fc(x).chunk(2, dim=-1)
        return torch.sigmoid(a) * b


class AddNorm(nn.Module):
    """Residual add + LayerNorm (the post-block "Add & Norm" sub-layer)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.norm(x + residual)


class GatedResidualNetwork(nn.Module):
    """TFT's GRN block: ELU MLP -> dropout -> GLU -> Add&Norm (with skip).

    If ``context_size`` is given, an optional context vector is added to the
    pre-activation of the first linear (this is how static metadata flows into
    the VSN and the static-enrichment layer in the original paper).
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dropout: float = 0.1, context_size: Optional[int] = None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.skip = (nn.Identity() if input_size == output_size
                     else nn.Linear(input_size, output_size))
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.context_proj = (nn.Linear(context_size, hidden_size, bias=False)
                             if context_size is not None else None)
        self.glu = GatedLinearUnit(output_size, output_size, dropout=dropout)
        self.addnorm = AddNorm(output_size)

    def forward(self, x: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = self.skip(x)
        h = self.fc1(x)
        if self.context_proj is not None and context is not None:
            # broadcast context across the time dimension if needed
            if context.dim() == 2 and h.dim() == 3:
                context = context.unsqueeze(1).expand(-1, h.size(1), -1)
            h = h + self.context_proj(context)
        h = F.elu(h)
        h = self.fc2(h)
        h = self.glu(h)
        return self.addnorm(h, residual)


class VariableSelectionNetwork(nn.Module):
    """Per-timestep VSN: softmax weights over input variables, weighted sum of
    per-variable GRN-projected embeddings.

    Inputs are expected as a list of per-variable (B, T, hidden_size) tensors
    (already projected to the model's hidden_size). Returns the selected
    representation (B, T, hidden_size) and the sparse weights (B, T, V) where V
    = len(variable_tensors). The (optional) static ``context`` is fed to the
    weighting GRN, exactly as in the original TFT.
    """

    def __init__(self, num_inputs: int, hidden_size: int,
                 dropout: float = 0.1, context_size: Optional[int] = None):
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        # GRN that produces sparse weights from the *flattened* input vector.
        # Its output dimension is num_inputs (one weight per variable).
        self.weight_grn = GatedResidualNetwork(
            input_size=num_inputs * hidden_size,
            hidden_size=hidden_size,
            output_size=num_inputs,
            dropout=dropout,
            context_size=context_size,
        )
        # Per-variable GRN that processes each variable's embedding independently.
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
            for _ in range(num_inputs)
        ])

    def forward(self, var_tensors: List[torch.Tensor],
                context: Optional[torch.Tensor] = None):
        # var_tensors: list of V tensors each shape (B, T, hidden_size)
        assert len(var_tensors) == self.num_inputs
        flat = torch.cat(var_tensors, dim=-1)                  # (B, T, V*H)
        w_logits = self.weight_grn(flat, context=context)      # (B, T, V)
        weights = F.softmax(w_logits, dim=-1).unsqueeze(-1)    # (B, T, V, 1)

        per_var = torch.stack(
            [grn(v) for grn, v in zip(self.var_grns, var_tensors)], dim=-2,
        )                                                       # (B, T, V, H)
        out = (per_var * weights).sum(dim=-2)                  # (B, T, H)
        return out, weights.squeeze(-1)                        # (B, T, V)


class InterpretableMultiHeadAttention(nn.Module):
    """TFT's interpretable MHA: per-head Q/K projections, SHARED V projection
    across heads. Output = mean over heads of softmax(QK^T / sqrt(d_k)) V,
    then linear projection. Causal mask is supplied externally.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qs = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(n_heads)])
        self.ks = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(n_heads)])
        self.v_shared = nn.Linear(d_model, self.d_k)            # SHARED across heads
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(self.d_k, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        # q,k,v: (B, T, d_model). mask: (T_q, T_k) bool; True = block.
        v_proj = self.v_shared(v)                                # (B, T_k, d_k)
        head_outputs = []
        head_attns = []
        scale = 1.0 / math.sqrt(self.d_k)
        for h in range(self.n_heads):
            qh = self.qs[h](q)                                    # (B, T_q, d_k)
            kh = self.ks[h](k)                                    # (B, T_k, d_k)
            scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale  # (B, T_q, T_k)
            if mask is not None:
                scores = scores.masked_fill(mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)
            head_outputs.append(torch.matmul(attn, v_proj))      # (B, T_q, d_k)
            head_attns.append(attn)
        # Average over heads (interpretable: a single attention per (q,k))
        avg_out = torch.stack(head_outputs, dim=0).mean(dim=0)   # (B, T_q, d_k)
        avg_attn = torch.stack(head_attns, dim=0).mean(dim=0)    # (B, T_q, T_k)
        out = self.out_proj(avg_out)                              # (B, T_q, d_model)
        return out, avg_attn


# ---------------------------------------------------------------------------
# TFT
# ---------------------------------------------------------------------------
class TFTModel(nn.Module):
    """TFT for univariate-target forecasting with known-future covariates.

    Args
    ----
    n_real_known         : R, number of real covariates known over the WHOLE
                           window (e.g. meteorology + other pollutants).
    cat_cardinalities    : list of vocabulary sizes for the K categorical
                           covariates known over the whole window.
    lookback_len         : L.
    pred_len             : H.
    hidden_size          : d_model (LSTM hidden + attention dim).
    n_heads              : number of attention heads.
    num_lstm_layers      : LSTM depth for both encoder and decoder.
    dropout              : applied in GRNs / GLUs / attention / LSTM stacks.
    embedding_dim        : per-categorical embedding dim before VSN projection.

    forward(...) -> (y_hat, diagnostics)
    """

    def __init__(self, n_real_known: int, cat_cardinalities: List[int],
                 lookback_len: int, pred_len: int,
                 hidden_size: int = 64, n_heads: int = 4,
                 num_lstm_layers: int = 1, dropout: float = 0.1,
                 embedding_dim: int = 8):
        super().__init__()
        self.L = int(lookback_len)
        self.H = int(pred_len)
        self.R = int(n_real_known)
        self.K = len(cat_cardinalities)
        self.hidden_size = int(hidden_size)
        self.n_heads = int(n_heads)
        self.num_lstm_layers = int(num_lstm_layers)

        # --- Per-variable projections to a common ``hidden_size`` -------------
        # PAST stream variables: [PM2.5_lookback] + [R reals] + [K categoricals]
        # FUTURE stream variables: [R reals] + [K categoricals] (no target)
        self.target_proj = nn.Linear(1, hidden_size)
        self.real_projs = nn.ModuleList([
            nn.Linear(1, hidden_size) for _ in range(self.R)
        ])
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, embedding_dim) for card in cat_cardinalities
        ])
        self.cat_to_hidden = nn.ModuleList([
            nn.Linear(embedding_dim, hidden_size) for _ in range(self.K)
        ])

        # --- Variable selection networks ------------------------------------
        n_past_vars = 1 + self.R + self.K
        n_future_vars = self.R + self.K
        self.past_vsn = VariableSelectionNetwork(
            num_inputs=n_past_vars, hidden_size=hidden_size, dropout=dropout,
        )
        self.future_vsn = VariableSelectionNetwork(
            num_inputs=n_future_vars, hidden_size=hidden_size, dropout=dropout,
        )

        # --- Sequence-to-sequence LSTM ---------------------------------------
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.lstm_decoder = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        # Gated residual over the LSTM stack output (skip the *input* not the
        # output, as in the paper figure: LSTM output --GLU+Add&Norm--> ...).
        self.post_lstm_glu = GatedLinearUnit(hidden_size, hidden_size, dropout=dropout)
        self.post_lstm_addnorm = AddNorm(hidden_size)

        # --- Static enrichment (identity when no statics) --------------------
        self.static_enrichment = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout,
        )

        # --- Interpretable causal self-attention -----------------------------
        self.attention = InterpretableMultiHeadAttention(
            d_model=hidden_size, n_heads=n_heads, dropout=dropout,
        )
        self.post_attn_glu = GatedLinearUnit(hidden_size, hidden_size, dropout=dropout)
        self.post_attn_addnorm = AddNorm(hidden_size)

        # --- Position-wise feed-forward -------------------------------------
        self.pos_wise_ff = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout,
        )
        self.pre_output_glu = GatedLinearUnit(hidden_size, hidden_size, dropout=dropout)
        self.pre_output_addnorm = AddNorm(hidden_size)

        # --- Output head: single point prediction per future step -----------
        self.output_head = nn.Linear(hidden_size, 1)

        # Precompute the causal mask (registered as a buffer so .to(device)
        # follows the model and we never re-allocate on every forward pass).
        T = self.L + self.H
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    # -----------------------------------------------------------------
    # Projection helpers
    # -----------------------------------------------------------------
    def _project_reals(self, x_real: torch.Tensor) -> List[torch.Tensor]:
        # x_real (B, T, R) -> list of R tensors each (B, T, hidden_size)
        out = []
        for i, proj in enumerate(self.real_projs):
            out.append(proj(x_real[..., i:i + 1]))
        return out

    def _project_cats(self, x_cat: torch.Tensor) -> List[torch.Tensor]:
        # x_cat (B, T, K) int64 -> list of K tensors each (B, T, hidden_size)
        out = []
        for k in range(self.K):
            emb = self.cat_embeddings[k](x_cat[..., k])         # (B, T, E)
            out.append(self.cat_to_hidden[k](emb))              # (B, T, H)
        return out

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    def forward(self, y_past_tgt: torch.Tensor,
                x_real_past: torch.Tensor, x_real_future: torch.Tensor,
                x_cat_past: torch.Tensor, x_cat_future: torch.Tensor):
        # Shapes:
        #   y_past_tgt   (B, L, 1)
        #   x_real_past  (B, L, R)  x_real_future (B, H, R)
        #   x_cat_past   (B, L, K)  x_cat_future  (B, H, K)
        B = y_past_tgt.size(0)
        L, H = self.L, self.H

        # 1) Per-variable projections -------------------------------------
        past_vars = (
            [self.target_proj(y_past_tgt)]                       # target on lookback
            + self._project_reals(x_real_past)
            + self._project_cats(x_cat_past)
        )
        future_vars = (
            self._project_reals(x_real_future)
            + self._project_cats(x_cat_future)
        )

        # 2) Variable selection on each stream ----------------------------
        past_sel,  past_w  = self.past_vsn(past_vars)            # (B, L, H), (B, L, V_past)
        future_sel, future_w = self.future_vsn(future_vars)      # (B, H, H), (B, H, V_future)

        # 3) LSTM seq2seq --------------------------------------------------
        enc_out, (h_n, c_n) = self.lstm_encoder(past_sel)        # (B, L, H)
        dec_out, _ = self.lstm_decoder(future_sel, (h_n, c_n))   # (B, H, H)
        lstm_seq = torch.cat([enc_out, dec_out], dim=1)          # (B, L+H, H)
        vsn_seq = torch.cat([past_sel, future_sel], dim=1)       # (B, L+H, H)
        # Gated residual over the (VSN -> LSTM) sub-block
        post_lstm = self.post_lstm_addnorm(self.post_lstm_glu(lstm_seq), vsn_seq)

        # 4) Static enrichment (identity-shaped; no static context here) ---
        enriched = self.static_enrichment(post_lstm)             # (B, L+H, H)

        # 5) Causal interpretable multi-head self-attention ---------------
        attn_in = enriched                                        # query/key/value
        attn_out, attn_w = self.attention(
            attn_in, attn_in, attn_in, mask=self.causal_mask,
        )                                                         # (B, L+H, H)
        attn_post = self.post_attn_addnorm(self.post_attn_glu(attn_out), enriched)

        # 6) Position-wise feed-forward -----------------------------------
        ff = self.pos_wise_ff(attn_post)                          # (B, L+H, H)
        # Skip directly from the post-LSTM stream (paper Fig. 2)
        decoded = self.pre_output_addnorm(self.pre_output_glu(ff), post_lstm)

        # 7) Output head: take only the forecast horizon ------------------
        y_hat = self.output_head(decoded[:, L:L + H, :])          # (B, H, 1)

        diagnostics = {
            "past_var_weights":   past_w,                         # (B, L, 1+R+K)
            "future_var_weights": future_w,                       # (B, H, R+K)
            "attention_weights":  attn_w,                         # (B, L+H, L+H)
        }
        return y_hat, diagnostics
