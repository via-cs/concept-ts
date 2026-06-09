"""
ConceptBottleneckModel — joint-trained CBM for multivariate time series
forecasting.

Pipeline:

    x  ∈ (B, T_in, C)
       │
       ↓  Embedding_Learning (the existing encoder from model/embedding.py;
       │   whatever's currently aliased: Conv / ConvWithAdapter / Mixer / TS2Vec)
       │
       z  ∈ (B, D)         (averaged over time if encoder returns 3D)
       │
       ↓  Linear(D, K)
       │
       c_logits ∈ (B, K)   ────── BCE-with-logits loss vs concept labels
       │
       ↓  sigmoid
       │
       c ∈ (B, K)           soft activations in [0, 1]
       │
       ↓  ── bottleneck ──   (only c flows past here)
       │
       Linear(K, D)
       │
       context ∈ (B, D)
       │
       ↓  Linear(D, H_dec) → GELU → Linear(H_dec, T_out · O_dim) → reshape
       │
       y_hat ∈ (B, T_out, O_dim)  ────── L1 loss vs ground-truth forecast

forward returns a dict with all four intermediate tensors so the trainer
can compute both losses without recomputing anything.
"""

import torch
import torch.nn as nn

from model.embedding import Embedding_Learning


class ConceptBottleneckModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mid_dim: int,
        ext_dim: int,
        output_dim: int,
        bottle_dim_p: float,
        input_length: int,
        output_length: int,
        num_blocks: int,
        dropout: float,
        n_concepts: int,
        decoder_hidden: int = 256,
        use_rev_in: int = 1,
        norm_type: str = "B",
    ):
        super().__init__()
        self.n_concepts    = n_concepts
        self.output_length = output_length
        self.output_dim    = output_dim

        # ---- Encoder backbone ----
        # Uses whatever is bound to Embedding_Learning in model/embedding.py.
        # We discard the encoder's own forecast head (the first return value)
        # and only use its feature output.
        self.encoder = Embedding_Learning(
            input_dim=input_dim,
            mid_dim=mid_dim,
            ext_dim=ext_dim,
            output_dim=output_dim,
            bottle_dim_p=bottle_dim_p,
            input_length=input_length,
            output_length=output_length,
            num_blocks=num_blocks,
            dropout=dropout,
            norm_type=norm_type,
            use_rev_in=use_rev_in,
        )

        # ---- Concept predictor ----
        # Maps the time-pooled encoder output to K concept logits.
        self.concept_predictor = nn.Linear(mid_dim, n_concepts)

        # ---- Concept embeddings ----
        # Espinosa Zarlenga 2022 — one D-dim vector per concept. The context
        # vector handed to the forecast decoder is the concept-weighted sum
        # of these embeddings: context = c @ E.
        # Implemented as a Linear(K, D, bias=False) so initialization +
        # optimizer state are standard.
        self.concept_embeddings = nn.Linear(n_concepts, mid_dim, bias=False)

        # ---- Forecast decoder ----
        # MLP from the context vector to the flat (T_out × O_dim) forecast.
        self.forecast_decoder = nn.Sequential(
            nn.Linear(mid_dim, decoder_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, output_length * output_dim),
        )

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------

    def _pool(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """Reduce a possibly-3D encoder output to (B, D) by mean over the
        time axis. Some encoders (Conv with internal time_pool) already
        return 2D and need no reduction."""
        if encoder_out.dim() == 3:
            return encoder_out.mean(dim=1)
        return encoder_out

    def forward(self, x: torch.Tensor) -> dict:
        B = x.shape[0]

        # 1) Encoder backbone — discard the encoder's own forecast head output.
        _, z_raw = self.encoder(x)
        z = self._pool(z_raw)                         # (B, D)

        # 2) Concept layer.
        c_logits = self.concept_predictor(z)          # (B, K)
        c        = torch.sigmoid(c_logits)             # (B, K)

        # 3) Bottleneck — context vector is the concept-weighted sum of
        # learned concept embeddings.
        context = self.concept_embeddings(c)          # (B, D)

        # 4) Forecast decoder.
        flat = self.forecast_decoder(context)         # (B, T_out * O_dim)
        y_hat = flat.reshape(B, self.output_length, self.output_dim)

        return {
            "y_hat":    y_hat,
            "c_logits": c_logits,
            "c":        c,
            "context":  context,
        }
