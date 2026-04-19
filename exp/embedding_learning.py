import os
import argparse
import random
import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from model.embedding import Embedding_Learning
from dataloader.dataloader import LOF_Dataset



def get_args():
    parser = argparse.ArgumentParser(description="ProtoTS Pre-Training")
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset CSV')
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

    return parser.parse_args()

    

def test_loss(pred, ans):
    mse = ((pred - ans) ** 2).sum()
    mae = torch.abs(pred - ans).sum()
    return mse, mae

@torch.no_grad()
def evaluate(model, dataset, batch_size, loss_fn, device):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_mse, total_mae = 0.0, 0.0
    pred_shape = None

    for data, ans in dataloader:
        prediction, _ = model(data.to(device))
        mse, mae = loss_fn(prediction, ans.to(device))
        total_mse += mse.item()
        total_mae += mae.item()

        if pred_shape is None:
            pred_shape = prediction.size()

    model.train()
    denom = len(dataset) * pred_shape[1] * pred_shape[2]
    return total_mse / denom, total_mae / denom

def train_loop(
    model,
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size,
    num_epochs,
    optimizer,
    lr_decay_factor,
    lr_decay_step,
    checkpoint_path,
    device
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"[Epoch {epoch+1}/{num_epochs}] Current LR = {optimizer.param_groups[0]['lr']:.6f}")

        # with tqdm(total=len(train_loader), desc="Training") as pbar:
        for data, ans in train_loader:
            prediction, _ = model(data.to(device))
            loss = nn.L1Loss()(prediction, ans.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

                # pbar.set_postfix(loss=f"{loss.item():.4f}")
                # pbar.update(1)

        _, val_mae = evaluate(model, val_dataset, batch_size, test_loss, device)
        print(f"Validation MAE: {val_mae:.6f}")

        if val_mae < best_val_loss:
            best_val_loss = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved at {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print("Early stopping triggered.")
                break

        if (epoch + 1) % lr_decay_step == 0:
            optimizer.param_groups[0]['lr'] *= lr_decay_factor

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    final_mse, final_mae = evaluate(model, test_dataset, batch_size, test_loss, device)
    print(f"Final Test Results -> MSE: {final_mse:.6f}, MAE: {final_mae:.6f}")

def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def main():
    args = get_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ['device'] = device
    print(f"Using device: {device}")

    set_seed(0)
    
    filename = (
        f"{os.path.splitext(os.path.basename(args.dataset))[0]}_"
        f"OL{args.output_length}_"
        f"IL{args.input_length}_"
        f"L{args.num_layer}_"
        f"D{args.dropout}_"
        f"EMB{args.embed_dim}_"
        f"BDP{args.bottle_dim_p}.pth"
    )
    os.makedirs("./checkpoint/embedding_learning", exist_ok=True)
    checkpoint_path = os.path.join("./checkpoint/embedding_learning", filename)
    print("model will save to " + checkpoint_path)

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

    train_data = LOF_Dataset(args.input_length, args.output_length, 'train', args.input_dim, args.dataset)
    val_data   = LOF_Dataset(args.input_length, args.output_length, 'vali',  args.input_dim, args.dataset)
    test_data  = LOF_Dataset(args.input_length, args.output_length, 'test',  args.input_dim, args.dataset)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loop(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        test_dataset=test_data,
        batch_size=args.batch_size,
        num_epochs=args.epoch,
        optimizer=optimizer,
        lr_decay_factor=args.de,
        lr_decay_step=args.de_step,
        checkpoint_path=checkpoint_path,
        device=device
    )


if __name__ == "__main__":
    main()