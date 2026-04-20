import os
import argparse
import random
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.embedding import Embedding_Learning
from model.prototype import Prototype
from dataloader.dataloader import LOF_Dataset



def get_args():
    parser = argparse.ArgumentParser(description="Prototype-enhanced Training (formatted like the reference)")

    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset CSV / root')
    parser.add_argument('--epoch', type=int, required=True, help='Number of training epochs')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_length', type=int, default=96, help='Output sequence length')
    parser.add_argument('--input_length', type=int, default=96 * 4, help='Input sequence length')
    parser.add_argument('--input_dim', type=int, default=23, help='Input feature dimension')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')

    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--de', type=float, default=0.5, help='Learning rate decay factor')
    parser.add_argument('--de_step', type=int, default=2, help='Learning rate decay step (epochs)')

    parser.add_argument('--num_layer', type=int, default=2, help='Number of blocks/layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--bottle_dim_p', type=int, default=0.1, help='percentage of Embedding dimension')

    parser.add_argument('--prototype_num', type=int, default=12, help='Number of root prototypes')
    parser.add_argument('--prototype_layer_num', type=int, default=2, help='Number of prototype layers')
    parser.add_argument('--prototype_split_num', type=int, default=2, help='Number of each prototype can split into')
    parser.add_argument('--split_ratio', type=float, default=0.2, help='Selection ratio (alpha) for splitting leaf prototypes')
    parser.add_argument('--top_k_activate', type=int, default=3, help='Top-k activation count for similarity calculation')
    
    parser.add_argument('--pretrain_checkpoint', type=str, default='None', help='pretrained embedding checkpoints')
    parser.add_argument('--init_weight', type=str, default='None', help='Path to init_weight tensor (legacy, prototype_value only)')

    # LLM-guided initialization + hybrid training
    parser.add_argument('--init_embedding', type=str, default='None',
                        help='Path to init_embedding.pt (shape (P,D)) from prototype_initializer.py')
    parser.add_argument('--init_value', type=str, default='None',
                        help='Path to init_value.pt (shape (P,L,O)) from prototype_initializer.py')
    parser.add_argument('--anchor_mask', type=str, default='None',
                        help='Path to anchor_mask.pt (shape (P,)) from prototype_initializer.py')
    parser.add_argument('--anchor_lambda_emb', type=float, default=1.0,
                        help='Initial weight on the μ anchor term of the hybrid loss')
    parser.add_argument('--anchor_lambda_val', type=float, default=1.0,
                        help='Initial weight on the p anchor term of the hybrid loss')
    parser.add_argument('--anchor_decay', type=float, default=0.9,
                        help='Per-epoch multiplicative decay of the anchor weights '
                             '(1.0 = constant, <1 = gradually release prototypes)')

    return parser.parse_args()


def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def train_loss(pred, ans, score, anchor_term=None):
    """Forecasting loss + entropy reg + (optional) hybrid anchor term.

    The anchor term is the weighted MSE between current layer-0
    (μ, p) and their LLM-initialized anchors (μ₀, p₀). It acts as
    an elastic tether: big when a prototype strays too far, zero
    for blank (unanchored) prototypes. This is the "hybrid" half
    of the hybrid-training strategy — predictions push prototypes
    toward the data, anchors push them back toward their semantic
    meaning.
    """
    ls = 0
    for i in range(len(pred) - 1):
        ls += 0.1 * nn.L1Loss()(pred[i], ans)

    ls += nn.L1Loss()(pred[-1], ans)

    entropy = -torch.sum(score * torch.log(score + 1e-8), dim=-1)
    ls += (1e-3) * torch.mean(entropy)

    if anchor_term is not None:
        ls = ls + anchor_term

    return ls


def test_loss(pred, ans):
    mse = ((pred - ans) ** 2).sum()
    mae = torch.abs(pred - ans).sum()
    return mse, mae


@torch.no_grad()
def evaluate_with_prototype(model, prototype_model, dataset, batch_size, device):
    model.eval()
    prototype_model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_mse, total_mae = 0.0, 0.0
    denom = None

    for data, ans in dataloader:
        data = data.to(device)
        ans = ans.to(device)

        pred_raw, encoder = model(data)
        pred_list, _ = prototype_model(encoder, data)
        pred = model.embeding.denorm(pred_list[-1])

        mse, mae = test_loss(pred, ans)
        total_mse += mse.item()
        total_mae += mae.item()

        if denom is None:
            denom = len(dataset) * pred.size(1) * pred.size(2)

    model.train()
    prototype_model.train()
    return total_mse / denom, total_mae / denom


def train_loop(
    model, prototype_model, train_dataset, val_dataset,
    test_dataset, batch_size, num_epochs, prototype_layer_num,
    optimizer, lr, lr_decay_factor, lr_decay_step,
    checkpoint_path_model, checkpoint_path_prototype,
    init_weight_path, device,
    anchor_lambda_emb=0.0, anchor_lambda_val=0.0, anchor_decay=1.0,
):
    if os.path.exists(init_weight_path):
        init_value = torch.load(init_weight_path, map_location=device).float().to(device)
        with torch.no_grad():
            prototype_model.prototype.prototype_value[0].data = init_value.reshape(
                prototype_model.prototype.prototype_value[0].shape
            )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    best_val_mae = float('inf')
    patience_counter = 0
    time_step = 0

    current_layer_num = 1
    current_lr = optimizer.param_groups[0]["lr"]

    
    for epoch in range(num_epochs):
        # Decayed anchor coefficients: start strong so the prototypes stay
        # close to their LLM-given meanings, then release over time so they
        # can adapt to local details ("hybrid" training).
        cur_lambda_emb = anchor_lambda_emb * (anchor_decay ** epoch)
        cur_lambda_val = anchor_lambda_val * (anchor_decay ** epoch)

        print(f"[Epoch {epoch + 1}/{num_epochs}] Current LR = {optimizer.param_groups[0]['lr']:.6f}  "
              f"λ_μ={cur_lambda_emb:.4f}  λ_p={cur_lambda_val:.4f}")
        prototype_model.prototype.clear_error_statistics()

        for data, ans in train_loader:
            data = data.to(device)
            ans = ans.to(device)

            _, embd = model(data)
            all_ans, all_score = prototype_model(embd, data)
            for i in range(current_layer_num):
                all_ans[i] = model.embeding.denorm(all_ans[i])

            prototype_model.prototype.update_error_statistics(
                all_score, all_ans, ans.to(device)
            )

            anchor_term = None
            if cur_lambda_emb > 0 or cur_lambda_val > 0:
                anchor_term = prototype_model.anchor_loss(
                    lambda_emb=cur_lambda_emb,
                    lambda_val=cur_lambda_val,
                )

            loss = train_loss(
                all_ans,
                ans.to(device),
                all_score[0],
                anchor_term=anchor_term,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        _, val_mae = evaluate_with_prototype(model, prototype_model, val_dataset, batch_size, device)
        print(f"Validation MAE: {val_mae:.6f}")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path_model)
            torch.save(prototype_model.state_dict(), checkpoint_path_prototype)
            print(f"Base model saved at {checkpoint_path_model}")
            print(f"Prototype state saved at {checkpoint_path_prototype}")
        else:
            patience_counter += 1
            if patience_counter >= 5 and current_layer_num < prototype_layer_num:
                model.load_state_dict(torch.load(checkpoint_path_model, map_location=device))
                prototype_model.load_state_dict(torch.load(checkpoint_path_prototype, map_location=device))
                step_mse, step_mae = evaluate_with_prototype(model, prototype_model, test_dataset, batch_size, device)
                print(f"{current_layer_num} Layers Test Results -> MSE: {step_mse:.6f}, MAE: {step_mae:.6f}")
                
                
                current_lr = 2e-4
                optimizer = torch.optim.Adam(
                    [{"params": prototype_model.parameters()}, {"params": model.parameters()}],
                    # [{"params": prototype_model.parameters()}],
                    lr=current_lr
                )

                current_layer_num += 1
                prototype_model.prototype.add_prototype(device)

                best_val_mae = float('inf')
                patience_counter = 0
                time_step = 0

        time_step += 1

        if time_step % lr_decay_step == 0:
            current_lr *= lr_decay_factor
            if current_lr < 1e-6:
                break

        optimizer.param_groups[0]["lr"] = current_lr

        # test_mse, test_mae = evaluate_with_prototype(model, prototype_model, test_dataset, batch_size, device)
        # print(f"Test -> MSE: {test_mse:.6f}, MAE: {test_mae:.6f}")

    model.load_state_dict(torch.load(checkpoint_path_model, map_location=device))
    prototype_model.load_state_dict(torch.load(checkpoint_path_prototype, map_location=device))
    final_mse, final_mae = evaluate_with_prototype(model, prototype_model, test_dataset, batch_size, device)
    print(f"Final Test Results -> MSE: {final_mse:.6f}, MAE: {final_mae:.6f}")

def main():
    args = get_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ['device'] = device
    print(f"Using device: {device}")

    set_seed(0)

    os.makedirs("./checkpoint/prototype_learning", exist_ok=True)
    model_filename = (
        f"{os.path.splitext(os.path.basename(args.dataset))[0]}_"
        f"OL{args.output_length}_"
        f"IL{args.input_length}_"
        f"L{args.num_layer}_"
        f"D{args.dropout}_"
        f"EMB{args.embed_dim}.pth"
    )
    model_path = os.path.join("./checkpoint/prototype_learning", model_filename)
    
    prototype_filename = (
        f"{os.path.splitext(os.path.basename(args.dataset))[0]}_"
        f"OL{args.output_length}_"
        f"IL{args.input_length}_"
        f"PN{args.prototype_num}_"
        f"SR{args.split_ratio}.pth"
    )
    prototype_path = os.path.join("./checkpoint/prototype_learning", prototype_filename)
    
    model = Embedding_Learning(
        input_dim=args.input_dim,
        mid_dim=args.embed_dim,
        ext_dim=args.input_dim - args.output_dim,
        output_dim=args.output_dim,
        bottle_dim_p=args.bottle_dim_p,
        input_length=args.input_length,
        output_length=args.output_length,
        num_blocks=args.num_layer,
        dropout=args.dropout,
        norm_type='B',
        use_rev_in=1
    ).to(device)

    prototype_model = Prototype(
        prototype_num=args.prototype_num,
        prototype_dim=args.embed_dim,
        output_dim=args.output_dim,
        input_length=args.output_length,
        prototype_split_num=args.prototype_split_num,
        top_k_activate=args.top_k_activate,
        split_ratio=args.split_ratio,
    ).to(device)

    if os.path.exists(args.pretrain_checkpoint):
        model.load_state_dict(torch.load(args.pretrain_checkpoint, map_location=device))
        print("Successfully load pretrain checkpoint!")

    if os.path.exists(args.init_weight):
        init_value=torch.load(args.init_weight).float().to(os.environ.get('device'))
        with torch.no_grad():
            prototype_model.prototype.prototype_value[0].data=init_value.reshape(args.prototype_num,96,1)
        print("Successfully load init weight!")

    # --------------------------------------------------------------
    # LLM-guided prototype initialization + anchor registration.
    # Expects the three tensors produced by
    # LLM_prototype/prototype_initializer.py.
    # --------------------------------------------------------------
    if (os.path.exists(args.init_embedding)
            and os.path.exists(args.init_value)
            and os.path.exists(args.anchor_mask)):

        mu0 = torch.load(args.init_embedding, map_location=device).float()
        p0  = torch.load(args.init_value,     map_location=device).float()
        mk0 = torch.load(args.anchor_mask,    map_location=device).float()

        P_expected = prototype_model.prototype.prototype_embedding[0].shape[0]
        if mu0.shape[0] != P_expected:
            raise ValueError(
                f"init_embedding has {mu0.shape[0]} prototypes but "
                f"--prototype_num is {P_expected}. Set "
                f"--prototype_num {mu0.shape[0]} (N LLM + M blank)."
            )

        prototype_model.set_initial_prototypes(mu0, p0, mk0)
        n_anchored = int(mk0.sum().item())
        n_blank    = int((mk0 == 0).sum().item())
        print(f"LLM init loaded: {n_anchored} anchored + {n_blank} blank "
              f"(total {P_expected})")
    else:
        print("No LLM init tensors provided — running with the original "
              "ProtoTS random prototypes.")

    train_data = LOF_Dataset(args.input_length, args.output_length, 'train', args.input_dim, args.dataset)
    val_data   = LOF_Dataset(args.input_length, args.output_length, 'vali',  args.input_dim, args.dataset)
    test_data  = LOF_Dataset(args.input_length, args.output_length, 'test',  args.input_dim, args.dataset)

    optimizer = torch.optim.Adam(
        [{"params": prototype_model.parameters()}, {"params": model.parameters()}], 
        # [{"params": prototype_model.parameters()}], 
        lr=args.lr
    )

    train_loop(
        model=model,
        prototype_model=prototype_model,
        train_dataset=train_data,
        val_dataset=val_data,
        test_dataset=test_data,
        batch_size=args.batch_size,
        num_epochs=args.epoch,
        prototype_layer_num=args.prototype_layer_num,
        optimizer=optimizer,
        lr=args.lr,
        lr_decay_factor=args.de,
        lr_decay_step=args.de_step,
        checkpoint_path_model=model_path,
        checkpoint_path_prototype=prototype_path,
        init_weight_path=args.init_weight,
        device=device,
        anchor_lambda_emb=args.anchor_lambda_emb,
        anchor_lambda_val=args.anchor_lambda_val,
        anchor_decay=args.anchor_decay,
    )


if __name__ == "__main__":
    main()
