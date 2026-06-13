"""
ConceptBottleneckModel — 1 + N concept bottlenecks for interpretable
multivariate time-series forecasting.

Architecture
------------

    x  in (B, T_in + T_out, C)
       |
       v  Embedding_Learning (encoder backbone from model/embedding.py)
       |
       z  in (B, D)                              # time-pooled feature vector
       |
       +-- Lookback concept layer ----------------------------------------+
       |   lb_logits = Linear(D, K_lb)(z)                                 |
       |   lb_acts   = sigmoid(lb_logits)                                 |
       |   lb_ctx    = Linear(K_lb, D, bias=False)(lb_acts)               |
       |                                                                  |
       +-- Forecast concept layers (one per sub-window k = 0..N-1) -------+
       |   for k:                                                         |
       |     fc_logits[k] = Linear(D, K_fc)(z)        # K_sw distinct     |
       |     fc_acts[k]   = sigmoid(fc_logits[k])     # predictors        |
       |     fc_ctx[k]    = Linear(K_fc, D, bias=False)(fc_acts[k])       |
       |                                              # shared embedding  |
       |                                                                  |
       +-- Shared sub-decoder, parameterized by sub-window id k ----------+
           for k:
             sw_emb   = Embedding(N, sw_id_dim)(k)
             dec_in   = concat(lb_ctx, fc_ctx[k], sw_emb)
             sub_y[k] = MLP(dec_in)              # (B, T_sw, output_dim)
           y_hat = concat(sub_y[0], ..., sub_y[N-1])  # (B, T_out, output_dim)

Why this shape:

  - Lookback bottleneck answers "what regime are we coming from" -- ONE
    bottleneck whose K_lb concepts probe the history (e.g. is_history_cold,
    is_history_load_high).
  - Forecast bottlenecks answer "what is happening at sub-window k" -- N
    bottlenecks. Concept VOCABULARY (the K_fc rules) and concept EMBEDDINGS
    are shared across sub-windows, so the rules define WHAT regimes exist
    and the per-sub-window activations capture WHEN each regime fires.
  - Concept PREDICTORS are separate per sub-window because predicting
    "is morning in sub-window 0" given only the lookback is a different
    task from predicting "is morning in sub-window 3".
  - One shared sub-decoder takes (history_context, sub_window_context, k)
    -> T_sw forecast steps. This avoids the discontinuities you would get
    from independent per-sub-window decoders and keeps the parameter count
    bounded.

Forward returns a dict with both concept tensors so the trainer can apply
BCE losses against the corresponding labels:

    y_hat      : (B, T_out, output_dim)
    lb_logits  : (B, K_lb)
    lb_acts    : (B, K_lb)
    fc_logits  : (B, N, K_fc)
    fc_acts    : (B, N, K_fc)
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
        n_lookback_concepts: int,
        n_forecast_concepts: int,
        n_subwindows: int = 4,
        decoder_hidden: int = 256,
        sub_window_id_dim: int = 16,
        use_rev_in: int = 1,
        norm_type: str = "B",
    ):
        super().__init__()
        if output_length % n_subwindows != 0:
            raise ValueError(
                f"output_length ({output_length}) must be divisible by "
                f"n_subwindows ({n_subwindows})."
            )
        self.n_lookback_concepts = n_lookback_concepts
        self.n_forecast_concepts = n_forecast_concepts
        self.n_subwindows        = n_subwindows
        self.output_length       = output_length
        self.output_dim          = output_dim
        self.sub_window_length   = output_length // n_subwindows

        # ---- Encoder backbone ----
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

        # ---- Lookback concept bottleneck ----
        self.lookback_concept_predictor  = nn.Linear(mid_dim, n_lookback_concepts)
        self.lookback_concept_embeddings = nn.Linear(n_lookback_concepts, mid_dim, bias=False)

        # ---- Forecast concept bottlenecks ----
        # K_sw separate predictors that read the encoder output. They share
        # the embedding matrix so "is_temperature_rising" maps to the same
        # D-dim vector regardless of which sub-window fired it.
        self.forecast_concept_predictors = nn.ModuleList([
            nn.Linear(mid_dim, n_forecast_concepts)
            for _ in range(n_subwindows)
        ])
        self.forecast_concept_embeddings = nn.Linear(n_forecast_concepts, mid_dim, bias=False)

        # ---- Sub-window id embedding ----
        self.sub_window_id_embed = nn.Embedding(n_subwindows, sub_window_id_dim)

        # ---- Shared sub-decoder ----
        # Input: history context (D) + sub-window context (D) + id embed (sw_id_dim)
        # Output: T_sw forecast values
        dec_in_dim = 2 * mid_dim + sub_window_id_dim
        self.sub_decoder = nn.Sequential(
            nn.Linear(dec_in_dim, decoder_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, self.sub_window_length * output_dim),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pool(encoder_out: torch.Tensor) -> torch.Tensor:
        """Reduce a possibly-3D encoder output to (B, D) via mean-over-time."""
        if encoder_out.dim() == 3:
            return encoder_out.mean(dim=1)
        return encoder_out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict:
        B = x.shape[0]
        device = x.device

        # 1) Encode and time-pool.
        _, z_raw = self.encoder(x)
        z = self._pool(z_raw)                                  # (B, D)

        # 2) Lookback concept layer.
        lb_logits = self.lookback_concept_predictor(z)         # (B, K_lb)
        lb_acts   = torch.sigmoid(lb_logits)                   # (B, K_lb)
        lb_ctx    = self.lookback_concept_embeddings(lb_acts)  # (B, D)

        # 3) Forecast concept layers (one per sub-window).
        fc_logits_list = []
        fc_acts_list   = []
        fc_ctx_list    = []
        for k in range(self.n_subwindows):
            lg = self.forecast_concept_predictors[k](z)        # (B, K_fc)
            ac = torch.sigmoid(lg)                              # (B, K_fc)
            cx = self.forecast_concept_embeddings(ac)          # (B, D)
            fc_logits_list.append(lg)
            fc_acts_list.append(ac)
            fc_ctx_list.append(cx)
        fc_logits = torch.stack(fc_logits_list, dim=1)         # (B, N, K_fc)
        fc_acts   = torch.stack(fc_acts_list,   dim=1)         # (B, N, K_fc)

        # 4) Shared sub-decoder, parameterized by sub-window id.
        sub_forecasts = []
        for k in range(self.n_subwindows):
            sw_idx = torch.full((B,), k, dtype=torch.long, device=device)
            sw_emb = self.sub_window_id_embed(sw_idx)          # (B, sw_id_dim)
            dec_in = torch.cat([lb_ctx, fc_ctx_list[k], sw_emb], dim=-1)
            flat   = self.sub_decoder(dec_in)                  # (B, T_sw * O)
            sub_y  = flat.reshape(B, self.sub_window_length, self.output_dim)
            sub_forecasts.append(sub_y)
        y_hat = torch.cat(sub_forecasts, dim=1)                # (B, T_out, O)

        return {
            "y_hat":     y_hat,
            "lb_logits": lb_logits,
            "lb_acts":   lb_acts,
            "fc_logits": fc_logits,
            "fc_acts":   fc_acts,
        }
