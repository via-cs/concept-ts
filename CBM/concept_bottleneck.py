"""
ConceptBottleneckModel — 1 + N concept bottlenecks for interpretable
multivariate time-series forecasting.

Architecture
------------

    x  in (B, T_in + T_out, C)
       |
       v  Embedding_Learning (encoder backbone, use_time_pool=False)
       |
       enc in (B, T_out, D)                      # per-timestep encoder output
       |
       split into:
         z_global = enc.mean(dim=1)              # (B, D) -- whole forecast window
         z_sw_k   = enc[:, k*T_sw:(k+1)*T_sw, :].mean(dim=1)   # (B, D) per sub-window
       |
       +-- Lookback concept layer (full CEM) -----------------------------+
       |   lb_logits = Linear(D, K_lb)(z_global)                          |
       |   lb_acts   = sigmoid(lb_logits)                                 |
       |   lb_eplus  = MLP+(z_global) reshape (B, K_lb, d)                |
       |   lb_eminus = MLP-(z_global) reshape (B, K_lb, d)                |
       |   lb_ctx    = concat over i of                                   |
       |                  a_i * lb_eplus[i] + (1-a_i) * lb_eminus[i]      |
       |                  in R^(K_lb * d)                                 |
       |                                                                  |
       +-- Forecast concept layers (full CEM, per sub-window) ------------+
       |   for k:                                                         |
       |     fc_logits[k] = Linear(D, K_fc)(z_sw_k)  # K_sw distinct      |
       |     fc_acts[k]   = sigmoid(fc_logits[k])    # predictors         |
       |     fc_eplus[k]  = MLP+(z_sw_k)  shape (B, K_fc, d)              |
       |     fc_eminus[k] = MLP-(z_sw_k)  shape (B, K_fc, d)              |
       |     fc_ctx[k]    = concat over i of                              |
       |                       a_i^k * fc_eplus[k][i]                     |
       |                     + (1 - a_i^k) * fc_eminus[k][i]              |
       |                     in R^(K_fc * d)                              |
       |   (MLPs MLP+, MLP- are shared across sub-windows; only inputs    |
       |    z_sw_k differ.)                                               |
       |                                                                  |
       +-- Residual channel (optional, hybrid CBM) -----------------------+
       |   residual[k] = Linear(D, residual_dim)(z_sw_k)                   |
       |   (penalized in loss; concept-only forecast still extractable     |
       |    by calling forward(..., disable_residual=True))                |
       |                                                                   |
       +-- Shared sub-decoder, parameterized by sub-window id k ----------+
           for k:
             sw_emb   = Embedding(N, sw_id_dim)(k)
             dec_in   = concat(lb_ctx, fc_ctx[k], sw_emb, residual[k])
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

import inspect
import warnings

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
        concept_emb_dim: int = 16,
        decoder_hidden: int = 512,
        decoder_depth: int = 3,
        sub_window_id_dim: int = 16,
        residual_dim: int = 16,
        single_forecast_bottleneck: bool = False,
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
        self.residual_dim        = int(residual_dim)
        self.concept_emb_dim     = int(concept_emb_dim)
        self.single_forecast_bottleneck = bool(single_forecast_bottleneck)

        # ---- Encoder backbone ----
        # We deliberately disable the encoder's internal time_pool (if it has
        # one) so the encoder returns per-timestep features of shape
        # (B, output_length, D) instead of a global (B, D) summary. The
        # forecast concept predictors then slice these per sub-window, so
        # predictor k actually sees the encoder's features specific to
        # forecast sub-window k -- which is the whole point of having N
        # forecast bottlenecks. Without this, every forecast predictor would
        # receive the same mean-pooled vector and couldn't distinguish
        # "rising temperature in sub-window 0" from sub-window 3.
        encoder_kwargs = dict(
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
        # Only pass use_time_pool=False if the active encoder supports it
        # (ConvEmbedding_Learning does; Mixer/TS2Vec encoders never had it
        # and already return 3D features).
        enc_sig = inspect.signature(Embedding_Learning.__init__)
        if "use_time_pool" in enc_sig.parameters:
            encoder_kwargs["use_time_pool"] = False
        self.encoder = Embedding_Learning(**encoder_kwargs)

        # ---- Lookback concept bottleneck (full CEM) ----
        # Predictor stays the same: encoder feature -> K_lb concept logits.
        # Embeddings are now INPUT-CONDITIONAL via two MLPs per concept set:
        #   e_i^+(z) = "concept-present" embedding given encoder feature z
        #   e_i^-(z) = "concept-absent"  embedding given encoder feature z
        # Per-concept embedding handed to the decoder:
        #   ĉ_i = a_i * e_i^+(z) + (1 - a_i) * e_i^-(z)         in R^d
        # Concept-context for the decoder is the concatenation of all ĉ_i.
        # We implement each branch as ONE Linear(D, K*d) and reshape to (B, K, d)
        # -- exactly equivalent to K independent Linear(D, d) per concept, but
        # one matmul instead of K.
        self.lookback_concept_predictor = nn.Linear(mid_dim, n_lookback_concepts)
        self.lookback_emb_plus  = nn.Linear(mid_dim, n_lookback_concepts * self.concept_emb_dim)
        self.lookback_emb_minus = nn.Linear(mid_dim, n_lookback_concepts * self.concept_emb_dim)

        # ---- Forecast concept bottlenecks (full CEM) ----
        # K_sw separate PREDICTORS (one per sub-window). The two embedding
        # MLPs are SHARED across sub-windows -- "is_temperature_rising"
        # generates its embedding the same way regardless of which sub-window
        # fired it; only the input z_per_sw[k] (and thus the resulting
        # embedding vector) differs. This mirrors how the static embedding
        # matrix was shared previously.
        # Number of forecast predictors:
        #   single_forecast_bottleneck=False (default): N (one per sub-window)
        #   single_forecast_bottleneck=True            : 1 (shared across all sub-windows)
        n_fc_predictors = 1 if self.single_forecast_bottleneck else n_subwindows
        self.forecast_concept_predictors = nn.ModuleList([
            nn.Linear(mid_dim, n_forecast_concepts)
            for _ in range(n_fc_predictors)
        ])
        self.forecast_emb_plus  = nn.Linear(mid_dim, n_forecast_concepts * self.concept_emb_dim)
        self.forecast_emb_minus = nn.Linear(mid_dim, n_forecast_concepts * self.concept_emb_dim)

        # ---- Sub-window id embedding ----
        self.sub_window_id_embed = nn.Embedding(n_subwindows, sub_window_id_dim)

        # ---- Residual channel (hybrid CBM, optional) ----
        # A small linear projection from the per-sub-window encoder summary
        # straight to the decoder, BYPASSING the concept layer. The trainer
        # adds an L2 penalty (alpha_residual_reg * ||residual||^2) so the
        # decoder is incentivized to use it sparingly -- only for forecast
        # detail the concept activations can't capture. Set residual_dim=0
        # to recover the strict CBM (no residual path).
        # The interpretability story stays intact as long as concept_only_MAE
        # and full_MAE (reported at test time) are close: a small gap means
        # most of the forecast is concept-driven; a big gap means the
        # residual is doing the heavy lifting.
        if self.residual_dim > 0:
            self.residual_proj = nn.Linear(mid_dim, self.residual_dim)
        else:
            self.residual_proj = None

        # ---- Shared sub-decoder ----
        # Input dims (full-CEM layout):
        #   lookback context  = K_lb * d         (input-conditional embeddings)
        #   forecast context  = K_fc * d         (input-conditional embeddings)
        #   sub-window id     = sw_id_dim
        #   residual          = residual_dim
        # Output: T_sw forecast values. decoder_depth = number of Linear layers (>=1).
        lb_ctx_dim = n_lookback_concepts * self.concept_emb_dim
        fc_ctx_dim = n_forecast_concepts * self.concept_emb_dim
        dec_in_dim = lb_ctx_dim + fc_ctx_dim + sub_window_id_dim + self.residual_dim
        if decoder_depth < 1:
            raise ValueError(f"decoder_depth must be >= 1, got {decoder_depth}")
        layers: list[nn.Module] = []
        in_dim = dec_in_dim
        for _ in range(decoder_depth - 1):
            layers.extend([
                nn.Linear(in_dim, decoder_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = decoder_hidden
        layers.append(nn.Linear(in_dim, self.sub_window_length * output_dim))
        self.sub_decoder = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_encoder_features(self, encoder_out: torch.Tensor
                                ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Turn the encoder output into (z_global, [z_sw_0, ..., z_sw_{N-1}]).

        ``z_global`` is a (B, D) summary used by the LOOKBACK concept predictor.
        ``z_sw_k``    is a (B, D) summary specific to forecast sub-window k,
                      computed by mean-pooling encoder features within the
                      sub-window's slice of the time axis. Used by
                      ``forecast_concept_predictors[k]``.

        If the encoder returns a 2D (B, D) tensor (e.g. some downstream encoder
        without per-timestep output), the per-sub-window summaries fall back
        to the same vector and we emit a one-time warning -- in that case the
        sub-window predictors can only differ via their weights, not inputs.
        """
        if encoder_out.dim() == 2:
            if not getattr(self, "_warned_pooled_encoder", False):
                warnings.warn(
                    "ConceptBottleneckModel: encoder returned a 2D (B, D) tensor; "
                    "per-sub-window concept predictors will receive identical "
                    "inputs. To get per-sub-window features, use an encoder that "
                    "returns (B, T_out, D) (e.g. set use_time_pool=False).",
                    RuntimeWarning,
                )
                self._warned_pooled_encoder = True
            z_global = encoder_out
            return z_global, [z_global] * self.n_subwindows

        if encoder_out.dim() != 3:
            raise ValueError(
                f"Encoder output has unexpected ndim={encoder_out.dim()}; "
                f"expected 2 or 3."
            )

        # (B, T, D). If T != output_length (e.g. encoder emits per-patch features
        # across the full input+output window), take the last output_length
        # timesteps as the forecast window.
        T = encoder_out.shape[1]
        if T != self.output_length:
            encoder_out = encoder_out[:, -self.output_length:, :]

        z_global = encoder_out.mean(dim=1)               # (B, D)

        sw_len = self.sub_window_length
        z_per_sw = [
            encoder_out[:, k * sw_len : (k + 1) * sw_len, :].mean(dim=1)
            for k in range(self.n_subwindows)
        ]
        return z_global, z_per_sw

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, disable_residual: bool = False) -> dict:
        """Forward pass.

        Args
        ----
        x : (B, T_in + T_out, C) raw input from LOF_Dataset.
        disable_residual : if True, the residual channel is zeroed before
            entering the decoder. Used at evaluation time to compute the
            "concept-only" MAE for interpretability attribution.
        """
        B = x.shape[0]
        device = x.device

        # 1) Encode. encoder_out is (B, T_out, D) when use_time_pool was off.
        _, encoder_out = self.encoder(x)
        z_global, z_per_sw = self._split_encoder_features(encoder_out)

        # 2) Lookback concept layer. Activations from the standard predictor,
        # embeddings INPUT-CONDITIONAL via two MLP branches:
        #     ĉ_i = a_i * e_i^+(z_global) + (1 - a_i) * e_i^-(z_global)
        # The decoder receives concat(ĉ_0, ..., ĉ_{K_lb-1}) in R^{K_lb * d}.
        lb_logits = self.lookback_concept_predictor(z_global)  # (B, K_lb)
        lb_acts   = torch.sigmoid(lb_logits)                   # (B, K_lb)
        lb_eplus  = self.lookback_emb_plus(z_global).reshape(B,
                        self.n_lookback_concepts, self.concept_emb_dim)
        lb_eminus = self.lookback_emb_minus(z_global).reshape(B,
                        self.n_lookback_concepts, self.concept_emb_dim)
        a_lb = lb_acts.unsqueeze(-1)                            # (B, K_lb, 1)
        lb_ctx_3d = a_lb * lb_eplus + (1.0 - a_lb) * lb_eminus  # (B, K_lb, d)
        lb_ctx    = lb_ctx_3d.reshape(B, -1)                    # (B, K_lb * d)

        # 3) Forecast concept layer(s).
        if self.single_forecast_bottleneck:
            # Ablation mode: ONE forecast predictor on z_global. The same
            # concept context is reused for every sub-window's decoding.
            lg = self.forecast_concept_predictors[0](z_global)       # (B, K_fc)
            ac = torch.sigmoid(lg)
            ep = self.forecast_emb_plus(z_global).reshape(B,
                        self.n_forecast_concepts, self.concept_emb_dim)
            em = self.forecast_emb_minus(z_global).reshape(B,
                        self.n_forecast_concepts, self.concept_emb_dim)
            a_g = ac.unsqueeze(-1)                                    # (B, K_fc, 1)
            ctx_3d = a_g * ep + (1.0 - a_g) * em                      # (B, K_fc, d)
            ctx_global = ctx_3d.reshape(B, -1)                        # (B, K_fc * d)
            # Reuse this single context for all sub-window decodings.
            fc_ctx_list = [ctx_global] * self.n_subwindows
            # External output: (B, K_fc) -- one activation per concept, no sw axis.
            fc_logits = lg
            fc_acts   = ac
        else:
            # Standard mode: predictor k sees z_per_sw[k]. Embedding MLPs are
            # shared across sub-windows but produce DIFFERENT embeddings per
            # sub-window because their input (z_per_sw[k]) is different.
            fc_logits_list = []
            fc_acts_list   = []
            fc_ctx_list    = []
            for k in range(self.n_subwindows):
                lg = self.forecast_concept_predictors[k](z_per_sw[k])    # (B, K_fc)
                ac = torch.sigmoid(lg)                                    # (B, K_fc)
                ep = self.forecast_emb_plus(z_per_sw[k]).reshape(B,
                            self.n_forecast_concepts, self.concept_emb_dim)
                em = self.forecast_emb_minus(z_per_sw[k]).reshape(B,
                            self.n_forecast_concepts, self.concept_emb_dim)
                a_k = ac.unsqueeze(-1)                                    # (B, K_fc, 1)
                ctx_3d_k = a_k * ep + (1.0 - a_k) * em                    # (B, K_fc, d)
                ctx_k = ctx_3d_k.reshape(B, -1)                           # (B, K_fc * d)
                fc_logits_list.append(lg)
                fc_acts_list.append(ac)
                fc_ctx_list.append(ctx_k)
            fc_logits = torch.stack(fc_logits_list, dim=1)         # (B, N, K_fc)
            fc_acts   = torch.stack(fc_acts_list,   dim=1)         # (B, N, K_fc)

        # 4) Residual channel (hybrid CBM). One projection per sub-window;
        # the projection module itself is shared.
        if self.residual_proj is not None:
            residual_list = [self.residual_proj(z_per_sw[k])
                              for k in range(self.n_subwindows)]
            if disable_residual:
                residual_list = [torch.zeros_like(r) for r in residual_list]
            residual = torch.stack(residual_list, dim=1)       # (B, N, residual_dim)
        else:
            residual_list = [None] * self.n_subwindows
            residual = x.new_zeros((B, self.n_subwindows, 0))

        # 5) Shared sub-decoder, parameterized by sub-window id.
        sub_forecasts = []
        for k in range(self.n_subwindows):
            sw_idx = torch.full((B,), k, dtype=torch.long, device=device)
            sw_emb = self.sub_window_id_embed(sw_idx)          # (B, sw_id_dim)
            parts = [lb_ctx, fc_ctx_list[k], sw_emb]
            if residual_list[k] is not None:
                parts.append(residual_list[k])                  # (B, residual_dim)
            dec_in = torch.cat(parts, dim=-1)
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
            "residual":  residual,   # (B, N, residual_dim) -- penalize in loss
        }
