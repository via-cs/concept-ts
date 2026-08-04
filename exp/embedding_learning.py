import os
import argparse
import random
import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from model.embedding import Embedding_Learning, ENCODER_CKPT_TAG
from dataloader.dataloader import LOF_Dataset
from exp.cluster_supervision import (
    ClusterLoss,
    ClusteredDataset,
    cluster_purity_probe,
    compute_cluster_assignments,
    compute_cluster_assignments_from_llm,
)



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

    # ---- Cluster supervision (SwAV/DeepCluster-style) ----
    # When enabled, K-means is run once on per-window lookback means to
    # produce a cluster index per training segment. A learnable centroid
    # is maintained per cluster; the encoder is trained to map each segment
    # toward its cluster's centroid via cross-entropy on
    # ``softmax(encoder @ centroids.T)``, weighted by --cluster_alpha.
    # After training, an intra/inter cosine probe verifies that the loss
    # actually shaped encoder geometry.
    parser.add_argument('--use_cluster_loss', action='store_true',
                        help='Add SwAV/DeepCluster-style cluster supervision to encoder pretraining')
    parser.add_argument('--cluster_source', type=str, default='llm',
                        choices=['llm', 'kmeans'],
                        help='How to derive cluster assignments: "llm" uses LLM-proposed '
                             'filter functions from --final_prototypes (requires --segments_path); '
                             '"kmeans" runs K-means on per-window lookback channel means.')
    parser.add_argument('--num_clusters', type=int, default=8,
                        help='K — number of clusters. For --cluster_source=llm this is inferred '
                             'from the JSON; for --cluster_source=kmeans it must match prototype_num.')
    parser.add_argument('--cluster_alpha', type=float, default=0.3,
                        help='Weight on the cluster cross-entropy loss')
    parser.add_argument('--final_prototypes', type=str, default='',
                        help='Path to final_prototypes_<timestamp>.json (only when --cluster_source=llm)')
    parser.add_argument('--segments_path', type=str, default='',
                        help='Path to <dataset>_full_segments.json (only when --cluster_source=llm)')

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
    device,
    cluster_loss_fn=None,        # optional ClusterLoss module
    cluster_alpha: float = 0.0,  # weight on the cluster term
):
    use_cluster = cluster_loss_fn is not None and cluster_alpha > 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"[Epoch {epoch+1}/{num_epochs}] Current LR = {optimizer.param_groups[0]['lr']:.6f}")

        epoch_forecast, epoch_cluster, n_batches = 0.0, 0.0, 0
        for batch in train_loader:
            if use_cluster:
                data, ans, clusters = batch
                clusters = clusters.to(device)
            else:
                data, ans = batch

            prediction, encoder_out = model(data.to(device))
            forecast_loss = nn.L1Loss()(prediction, ans.to(device))

            if use_cluster:
                # Reduce per-timestep encoder output (B, T, D) to (B, D).
                # ConvEmbedding (Mode A) already returns 2D; Mixer / WithAdapter
                # return 3D and need a time-mean — same convention as
                # LLM_prototype/prototype_initializer.py.
                if encoder_out.dim() == 3:
                    enc_2d = encoder_out.mean(dim=1)
                else:
                    enc_2d = encoder_out
                cl_loss = cluster_loss_fn(enc_2d, clusters)
                loss = forecast_loss + cluster_alpha * cl_loss
                epoch_cluster += cl_loss.item()
            else:
                loss = forecast_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_forecast += forecast_loss.item()
            n_batches += 1

        if use_cluster:
            print(f"  train forecast={epoch_forecast/n_batches:.4f}  "
                  f"cluster={epoch_cluster/n_batches:.4f}  "
                  f"(alpha={cluster_alpha})")

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
        f"BDP{args.bottle_dim_p}{ENCODER_CKPT_TAG}.pth"
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

    # ------------------------------------------------------------------
    # Optional cluster supervision.
    # ------------------------------------------------------------------
    cluster_loss_fn = None
    cluster_assignments = None
    K_actual = args.num_clusters
    if args.use_cluster_loss:
        print(f"\n[cluster_supervision] enabled "
              f"(source={args.cluster_source}, alpha={args.cluster_alpha})")

        if args.cluster_source == "llm":
            if not args.final_prototypes or not args.segments_path:
                raise ValueError(
                    "--cluster_source=llm requires both --final_prototypes and "
                    "--segments_path to be set."
                )
            cluster_assignments = compute_cluster_assignments_from_llm(
                train_data,
                final_prototypes_path=args.final_prototypes,
                segments_path=args.segments_path,
            )
            # K is inferred from the JSON, not from --num_clusters.
            K_actual = int(cluster_assignments[cluster_assignments >= 0].max()) + 1
            print(f"[cluster_supervision] LLM mode: inferred K={K_actual} from "
                  f"{args.final_prototypes}")
        else:   # K-means fallback
            cluster_assignments = compute_cluster_assignments(
                train_data, K=args.num_clusters, seed=0,
            )
            K_actual = args.num_clusters

        # Wrap the train dataset so each sample also yields its cluster index.
        train_data = ClusteredDataset(train_data, cluster_assignments)
        # Learnable centroids that drift to match the encoder's per-cluster mean.
        cluster_loss_fn = ClusterLoss(K=K_actual,
                                       embed_dim=args.embed_dim).to(device)
        params = list(model.parameters()) + list(cluster_loss_fn.parameters())
    else:
        params = list(model.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)

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
        device=device,
        cluster_loss_fn=cluster_loss_fn,
        cluster_alpha=args.cluster_alpha,
    )

    # ------------------------------------------------------------------
    # Post-training diagnostic — only when cluster supervision was on.
    # If the gap (intra − inter cosine) is < ~0.05, the cluster loss did
    # not bite; retune --cluster_alpha and retrain before proceeding to
    # prototype training.
    # ------------------------------------------------------------------
    if args.use_cluster_loss:
        print("\n[cluster_supervision] running purity probe on best checkpoint …")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        # Re-encode every training window. Use the un-wrapped train dataset
        # (we need the (x, y) tuple, not the (x, y, cluster) triple).
        plain_train = LOF_Dataset(args.input_length, args.output_length,
                                  'train', args.input_dim, args.dataset)
        loader = DataLoader(plain_train, batch_size=args.batch_size,
                            shuffle=False, drop_last=False)
        enc_chunks = []
        with torch.no_grad():
            for data, _ in loader:
                _, enc_out = model(data.to(device))
                if enc_out.dim() == 3:
                    enc_out = enc_out.mean(dim=1)
                enc_chunks.append(enc_out.cpu())
        enc_all = torch.cat(enc_chunks, dim=0)
        cluster_purity_probe(enc_all, cluster_assignments, K=K_actual)


if __name__ == "__main__":
    main()