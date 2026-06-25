"""
ConceptBottleneckModel — 1 + N concept bottlenecks for interpretable
multivariate time-series forecasting.

Architecture
------------

    x  in (B, T_in + T_out, C)
       |
       +-- lookback_encoder( x[:, :T_in, :] )    # SEES ONLY THE HISTORY
       |     enc_lb in (B, T_in, D)              # target RevIN'd (revin_all)
       |     z_global = enc_lb.mean(dim=1)       # (B, D) -> LOOKBACK bottleneck
       |
       +-- forecast_encoder( x[:, T_in:, :] )    # SEES ONLY THE FORECAST WINDOW
             enc_fc in (B, T_out, D)             # target masked (mask_all)
             z_sw_k            = enc_fc[:, k*T_sw:(k+1)*T_sw, :].mean(dim=1)  # per sub-window
             z_global_forecast = enc_fc.mean(dim=1)   # (B, D) -> GLOBAL-FORECAST bottleneck

    Two SEPARATE-WEIGHT encoders give a HARD lookback/forecast information
    barrier: lookback concepts are provably a function of the history, and
    forecast concepts a function of the forecast window. (use_time_pool=False
    on both; the CBM does the pooling.)
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
        n_global_forecast_concepts: int = 0,
        n_subwindows: int = 4,
        concept_emb_dim: int = 16,
        decoder_hidden: int = 512,
        decoder_depth: int = 3,
        sub_window_id_dim: int = 16,
        residual_dim: int = 16,
        sw_gru_hidden: int = 32,
        sw_summary_dim: int = 64,
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
        self.n_lookback_concepts        = n_lookback_concepts
        self.n_forecast_concepts        = n_forecast_concepts
        self.n_global_forecast_concepts = int(n_global_forecast_concepts)
        self.n_subwindows               = n_subwindows
        self.output_length       = output_length
        self.output_dim          = output_dim
        self.sub_window_length   = output_length // n_subwindows
        self.residual_dim        = int(residual_dim)
        self.concept_emb_dim     = int(concept_emb_dim)
        self.single_forecast_bottleneck = bool(single_forecast_bottleneck)
        self.input_length = int(input_length)

        # ---- Encoder backbones (TWO separate-weight encoders) ----
        # To give a HARD lookback/forecast information barrier, the lookback
        # and forecast concept bottlenecks read from two physically separate
        # encoders, each of which only ever sees its own slice of the window:
        #
        #   lookback_encoder  sees ONLY x[:, :input_length, :]  (the history,
        #       target real -> RevIN over the whole target, target_mode=
        #       "revin_all"). Its mean-pooled output is z_global, fed to the
        #       LOOKBACK bottleneck. So lookback concepts are PROVABLY a
        #       function of the history alone.
        #
        #   forecast_encoder  sees ONLY x[:, input_length:, :]  (the forecast
        #       window; target unknown -> masked, target_mode="mask_all",
        #       use_rev_in=0). Its per-timestep output is sliced per
        #       sub-window (z_sw_k) for the FORECAST sub-window bottlenecks and
        #       mean-pooled (z_global_forecast) for the GLOBAL-FORECAST
        #       bottleneck. So forecast concepts are a function of the forecast
        #       window alone.
        #
        # Both encoders run with use_time_pool=False so they return per-
        # timestep (B, T, D) features; the CBM does the pooling itself.
        # Implemented as a single-window encoder via input_length=0,
        # output_length=window_len (total_length = window_len), with the
        # target handled by target_mode rather than the legacy lookback/
        # forecast split.
        enc_sig = inspect.signature(Embedding_Learning.__init__)
        if "target_mode" not in enc_sig.parameters:
            raise RuntimeError(
                "The two-encoder ConceptBottleneckModel requires an encoder "
                "that supports `target_mode` (e.g. ConvEmbedding_Learning). "
                f"The active Embedding_Learning ({Embedding_Learning.__name__}) "
                "does not. Either switch the encoder alias in model/embedding.py "
                "or restore the single-encoder build."
            )

        def _make_window_encoder(window_len: int, target_mode: str,
                                 use_rev_in_flag: int):
            kw = dict(
                input_dim=input_dim,
                mid_dim=mid_dim,
                ext_dim=ext_dim,
                output_dim=output_dim,
                bottle_dim_p=bottle_dim_p,
                input_length=0,            # whole fed window handled via target_mode
                output_length=window_len,  # total_length == window_len
                num_blocks=num_blocks,
                dropout=dropout,
                norm_type=norm_type,
                use_rev_in=use_rev_in_flag,
                target_mode=target_mode,
            )
            if "use_time_pool" in enc_sig.parameters:
                kw["use_time_pool"] = False
            return Embedding_Learning(**kw)

        self.lookback_encoder = _make_window_encoder(
            window_len=input_length, target_mode="revin_all",
            use_rev_in_flag=use_rev_in,
        )
        self.forecast_encoder = _make_window_encoder(
            window_len=output_length, target_mode="mask_all",
            use_rev_in_flag=0,
        )

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
        # Single lookback concept predictor: encoder feature -> K_lb logits.
        # The lookback is summarized as a whole (no per-sub-window split) --
        # an ablation showed splitting the lookback hierarchically did NOT
        # help and inflated parameter count for redundant supervision, since
        # most lookback concepts are summary-style (season, weather regime)
        # and don't differ between halves of the same 96-hour history.
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

        # ---- Global forecast bottleneck (full CEM, OPTIONAL) ----
        # When n_global_forecast_concepts > 0, an extra bottleneck reads
        # z_global to produce a single set of activations for the WHOLE
        # forecast window. Its concept-context is concatenated into every
        # sub-window's decoder input (same value reused across sub-windows --
        # the global forecast IS one global characterization).
        # When n_global_forecast_concepts == 0, this bottleneck is skipped
        # entirely (no params, no decoder-input growth).
        if self.n_global_forecast_concepts > 0:
            self.global_forecast_concept_predictor = nn.Linear(
                mid_dim, self.n_global_forecast_concepts,
            )
            self.global_forecast_emb_plus  = nn.Linear(
                mid_dim, self.n_global_forecast_concepts * self.concept_emb_dim,
            )
            self.global_forecast_emb_minus = nn.Linear(
                mid_dim, self.n_global_forecast_concepts * self.concept_emb_dim,
            )

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

        # ---- Sub-window context BiGRU (sequence-aware refinement) ----
        # Each forecast sub-window has its own concept context fc_ctx[k] of
        # dim K_fc * d. We project them to a smaller summary, run a
        # bidirectional GRU across the N sub-window axis, and concatenate the
        # per-position output back into the decoder input. This lets the
        # decoder for sub-window k know what's happening in adjacent sub-
        # windows (sequential context) while keeping the per-sub-window
        # concept activations themselves fully interpretable.
        # Disabled (sw_gru_out_dim = 0) when sw_gru_hidden == 0 or when the
        # forecast side has been collapsed to a single global bottleneck
        # (--single_forecast_bottleneck makes the GRU degenerate -- same
        # context at every position).
        lb_ctx_dim  = n_lookback_concepts             * self.concept_emb_dim
        fc_ctx_dim  = n_forecast_concepts             * self.concept_emb_dim
        gfc_ctx_dim = self.n_global_forecast_concepts * self.concept_emb_dim

        self.sw_gru_hidden = int(sw_gru_hidden)
        if self.sw_gru_hidden > 0 and not self.single_forecast_bottleneck:
            self.sw_summary_dim = int(sw_summary_dim)
            self.sw_context_summary = nn.Linear(fc_ctx_dim, self.sw_summary_dim)
            self.sw_context_gru = nn.GRU(
                self.sw_summary_dim, self.sw_gru_hidden,
                bidirectional=True, batch_first=True,
            )
            sw_gru_out_dim = self.sw_gru_hidden * 2
        else:
            sw_gru_out_dim = 0

        # ---- Shared sub-decoder ----
        # Input dims (full-CEM layout):
        #   lookback context        = K_lb         * d   (input-conditional)
        #   forecast (sub-window)   = K_fc         * d   (input-conditional)
        #   sw_gru context          = 2 * sw_gru_hidden  (cross-sub-window aware; 0 if disabled)
        #   forecast (global)       = K_fc_global  * d   (input-conditional; 0 if disabled)
        #   sub-window id           = sw_id_dim
        #   residual                = residual_dim
        # Output: T_sw forecast values. decoder_depth = number of Linear layers (>=1).
        dec_in_dim  = (lb_ctx_dim + fc_ctx_dim + sw_gru_out_dim + gfc_ctx_dim
                       + sub_window_id_dim + self.residual_dim)
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

    def _pool_encoder_features(self,
                               enc_lb: torch.Tensor,
                               enc_fc: torch.Tensor
                               ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Pool the two encoders' outputs into the concept-layer inputs.

        Returns
        -------
        z_global           : (B, D) -- mean over the LOOKBACK encoder's
                             per-timestep features. Fed to the LOOKBACK
                             concept bottleneck. Provably a function of the
                             history alone (the lookback encoder never sees
                             the forecast window).
        z_per_sw           : list of N (B, D) tensors -- mean over each
                             forecast sub-window slice of the FORECAST
                             encoder's features. Fed to the per-sub-window
                             forecast bottlenecks (and the residual channel).
        z_global_forecast  : (B, D) -- mean over the FORECAST encoder's whole
                             output. Fed to the GLOBAL-FORECAST bottleneck.

        Both encoders run with use_time_pool=False, so enc_lb / enc_fc are
        (B, T, D). A 2D fallback is kept for robustness if an encoder ever
        returns a pooled (B, D) vector.
        """
        # ---- lookback summary ----
        if enc_lb.dim() == 2:
            z_global = enc_lb
        elif enc_lb.dim() == 3:
            z_global = enc_lb.mean(dim=1)                 # (B, D)
        else:
            raise ValueError(
                f"lookback encoder output has unexpected ndim={enc_lb.dim()}; "
                f"expected 2 or 3."
            )

        # ---- forecast: per-sub-window + whole-forecast summaries ----
        if enc_fc.dim() == 2:
            if not getattr(self, "_warned_pooled_encoder", False):
                warnings.warn(
                    "ConceptBottleneckModel: forecast encoder returned a 2D "
                    "(B, D) tensor; per-sub-window concept predictors will "
                    "receive identical inputs. Use an encoder that returns "
                    "(B, T_out, D) (e.g. use_time_pool=False).",
                    RuntimeWarning,
                )
                self._warned_pooled_encoder = True
            z_global_forecast = enc_fc
            return z_global, [enc_fc] * self.n_subwindows, z_global_forecast

        if enc_fc.dim() != 3:
            raise ValueError(
                f"forecast encoder output has unexpected ndim={enc_fc.dim()}; "
                f"expected 2 or 3."
            )

        # (B, T, D). If T != output_length, keep the last output_length steps.
        T = enc_fc.shape[1]
        if T != self.output_length:
            enc_fc = enc_fc[:, -self.output_length:, :]

        z_global_forecast = enc_fc.mean(dim=1)            # (B, D)
        sw_len = self.sub_window_length
        z_per_sw = [
            enc_fc[:, k * sw_len : (k + 1) * sw_len, :].mean(dim=1)
            for k in range(self.n_subwindows)
        ]
        return z_global, z_per_sw, z_global_forecast

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

        # 1) Encode with TWO separate encoders over disjoint windows.
        #    Clone the slices: PerChannelEmbedding overwrites the target column
        #    in-place, and cloning prevents the two encoder passes from
        #    aliasing each other's region of x.
        x_lb = x[:, : self.input_length, :].clone()       # lookback window
        x_fc = x[:, self.input_length :, :].clone()       # forecast window
        _, enc_lb = self.lookback_encoder(x_lb)           # (B, T_in,  D)
        _, enc_fc = self.forecast_encoder(x_fc)           # (B, T_out, D)
        z_global, z_per_sw, z_global_forecast = self._pool_encoder_features(
            enc_lb, enc_fc,
        )

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

        # 2b) Global forecast concept layer (OPTIONAL). Reads z_global to
        # produce ONE concept activation pattern characterizing the WHOLE
        # forecast window. Its context is reused across every sub-window's
        # decoding.
        if self.n_global_forecast_concepts > 0:
            # Reads the FORECAST encoder's whole-window summary (not the
            # lookback z_global) so global-forecast concepts derive from the
            # forecast window, matching how their labels are computed.
            gfc_logits = self.global_forecast_concept_predictor(z_global_forecast)   # (B, K_gfc)
            gfc_acts   = torch.sigmoid(gfc_logits)
            g_ep = self.global_forecast_emb_plus(z_global_forecast).reshape(
                B, self.n_global_forecast_concepts, self.concept_emb_dim,
            )
            g_em = self.global_forecast_emb_minus(z_global_forecast).reshape(
                B, self.n_global_forecast_concepts, self.concept_emb_dim,
            )
            a_g  = gfc_acts.unsqueeze(-1)                                    # (B, K_gfc, 1)
            gfc_ctx_3d = a_g * g_ep + (1.0 - a_g) * g_em                     # (B, K_gfc, d)
            gfc_ctx    = gfc_ctx_3d.reshape(B, -1)                           # (B, K_gfc * d)
        else:
            gfc_logits = x.new_zeros((B, 0))
            gfc_acts   = x.new_zeros((B, 0))
            gfc_ctx    = x.new_zeros((B, 0))

        # 3) Forecast concept layer(s).
        if self.single_forecast_bottleneck:
            # Ablation mode: ONE forecast predictor on the FORECAST encoder's
            # whole-window summary (z_global_forecast). The same concept
            # context is reused for every sub-window's decoding.
            lg = self.forecast_concept_predictors[0](z_global_forecast)  # (B, K_fc)
            ac = torch.sigmoid(lg)
            ep = self.forecast_emb_plus(z_global_forecast).reshape(B,
                        self.n_forecast_concepts, self.concept_emb_dim)
            em = self.forecast_emb_minus(z_global_forecast).reshape(B,
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

        # 3b) Sequence-aware refinement: BiGRU over the N sub-window contexts.
        # Projects each fc_ctx[k] to a small summary, runs a bidirectional GRU
        # across sub-windows, and yields a per-sub-window cross-context vector
        # that the decoder consumes alongside fc_ctx[k]. Skipped in single-
        # forecast-bottleneck mode (degenerate -- same context at every k).
        if (self.sw_gru_hidden > 0
                and not self.single_forecast_bottleneck
                and hasattr(self, "sw_context_gru")):
            fc_ctx_stack = torch.stack(fc_ctx_list, dim=1)           # (B, N, K_fc*d)
            sw_summaries = self.sw_context_summary(fc_ctx_stack)      # (B, N, sw_summary_dim)
            sw_enhanced, _ = self.sw_context_gru(sw_summaries)        # (B, N, 2*sw_gru_hidden)
        else:
            sw_enhanced = None

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
            parts = [lb_ctx, fc_ctx_list[k]]
            if sw_enhanced is not None:
                parts.append(sw_enhanced[:, k, :])              # (B, 2*sw_gru_hidden), cross-sub-window context
            if self.n_global_forecast_concepts > 0:
                parts.append(gfc_ctx)                           # (B, K_gfc * d), shared across sub-windows
            parts.append(sw_emb)
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
            "gfc_logits": gfc_logits, # (B, K_gfc); K_gfc=0 -> empty tensor
            "gfc_acts":   gfc_acts,
            "residual":  residual,   # (B, N, residual_dim) -- penalize in loss
        }
