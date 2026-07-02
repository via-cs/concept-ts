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
       +-- (STRICT CBM: no residual bypass. Every input to the sub-decoder --+
       |    is either a CEM concept context or a positional embedding.)     |
       |                                                                    |
       +-- Shared sub-decoder, parameterized by sub-window id k ------------+
           for k:
             sw_emb   = Embedding(N, sw_id_dim)(k)
             dec_in   = concat(lb_ctx, fc_ctx[k], gfc_ctx, sw_gru[k], sw_emb)
             # Residual sub-decoder stack: sum of B block deltas, each a
             # function of dec_in (with feedback: also the running forecast
             # from earlier blocks, itself a function of dec_in).
             running = 0
             for block in decoder_blocks:
                 delta = block(dec_in [+ running.flat if feedback and block>0])
                 running += delta
             sub_y[k] = running                   # (B, T_sw, output_dim)
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
import math
import warnings

import torch
import torch.nn as nn

from model.embedding import Embedding_Learning


# ---------------------------------------------------------------------------
# Sinusoidal hour-of-day encoding
# ---------------------------------------------------------------------------
def _sinusoidal_hour_encoding(hours: torch.Tensor, dim: int) -> torch.Tensor:
    """Deterministic sinusoidal encoding of hour-of-day (0..23).

    Produces sin/cos pairs at ``dim//2`` different periods, giving a rich,
    fixed representation of the hour. This is the recommended way to encode
    a cyclical calendar variable when you want a downstream Linear layer to
    have a clean, non-competing weight group for time-based concepts (a
    concept whose truth depends only on hour just puts its weight on these
    dims; the encoder-side weights stay at zero).

    Periods chosen: [24, 12, 6, 3] hours (daily, half-day, quarter-day,
    3-hour). Adding more periods captures finer harmonics but for hour-of-day
    concepts (which typically care about resolutions of 3-6 hours), 4 periods
    is plenty.

    Args
    ----
    hours : (B, N) int/long tensor of hours in [0, 23]
    dim   : output feature dim. Must be even. Default 8 → 4 sin + 4 cos.

    Returns
    -------
    (B, N, dim) float tensor of sinusoidal features.
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even (sin+cos pairs), got {dim}")
    n_freqs = dim // 2
    # Use the smallest n_freqs periods from [24, 12, 6, 3, 2, ...] -- if dim
    # is bigger than 8 we extend geometrically.
    default_periods = [24.0, 12.0, 6.0, 3.0]
    if n_freqs <= len(default_periods):
        periods = default_periods[:n_freqs]
    else:
        # Extend by halving beyond period 3 (rarely needed).
        periods = default_periods + [
            3.0 / (2 ** (i - len(default_periods) + 1))
            for i in range(len(default_periods), n_freqs)
        ]
    periods_t = torch.tensor(
        periods, device=hours.device, dtype=torch.float32,
    )                                                       # (n_freqs,)
    h = hours.float().unsqueeze(-1)                          # (B, N, 1)
    angles = 2 * math.pi * h / periods_t                     # (B, N, n_freqs)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, N, dim)


# ---------------------------------------------------------------------------
# Residual sub-decoder block (NHiTS-style residual stacking)
# ---------------------------------------------------------------------------
class ResidualDecoderBlock(nn.Module):
    """One block of the residual sub-decoder stack.

    Takes concept-derived context (the same ``dec_in`` the original monolithic
    ``sub_decoder`` used, optionally augmented with the running forecast from
    previous blocks) and produces a ``(T_sw, out_dim)`` DELTA forecast. The
    stack sums deltas across blocks -- each block's job is to correct the
    residual left by the earlier blocks (coarse trend first, mid-scale
    corrections next, fine detail last).

    INTERPRETABILITY GUARANTEE: every input to this block is concept-derived.
    - Block 0 receives ``dec_in``, which is the concatenation of the three
      CEM concept-context vectors (LOOKBACK, FORECAST-SW, FORECAST-GLOBAL) +
      cross-sub-window BiGRU refinement + sub-window id embedding. NO
      residual bypass -- the strict-CBM cleanup removed the residual channel
      entirely.
    - Blocks i>0 (when ``feedback`` is enabled at the stack level) additionally
      receive the RUNNING FORECAST produced by blocks 0..i-1. That running
      forecast is itself a function of concept-derived inputs, so it carries
      no new information -- it just lets the block condition its correction
      on "what has been predicted so far".

    Concretely: ablating any concept propagates linearly through every block.
    NO raw feature / raw future covariate access, so the concept bottleneck
    remains the ONLY information pathway to the forecast.
    """

    def __init__(self, in_dim: int, hidden: int, depth: int,
                 out_len: int, out_dim: int, dropout: float):
        super().__init__()
        if depth < 1:
            raise ValueError(f"ResidualDecoderBlock depth must be >= 1, got {depth}")
        self.out_len = int(out_len)
        self.out_dim = int(out_dim)
        layers: list[nn.Module] = []
        d = int(in_dim)
        for _ in range(depth - 1):
            layers.extend([
                nn.Linear(d, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            d = hidden
        layers.append(nn.Linear(d, self.out_len * self.out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim) -> returns (B, out_len, out_dim)."""
        flat = self.mlp(x)
        return flat.reshape(-1, self.out_len, self.out_dim)


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
        # ---- Residual sub-decoder stack (NHiTS-style) -----------------------
        # The old design was a single monolithic sub-decoder MLP: a big
        # Sequential mapping ``dec_in`` -> flat forecast. The new design is a
        # STACK of ``decoder_n_blocks`` ResidualDecoderBlocks. Each block
        # produces a signed DELTA forecast; the deltas sum to the final
        # sub-window forecast.
        #
        # decoder_n_blocks=1 recovers the OLD behavior exactly (single block,
        # no delta summation to speak of). decoder_n_blocks=3 is the
        # recommended default -- the first block predicts a coarse forecast
        # from the concept context, and subsequent blocks correct the residual.
        #
        # decoder_feedback controls whether blocks i>0 also see the running
        # forecast produced so far. If True (default), each block's input is
        # ``[dec_in, running.reshape(B, -1)]``; each block then specializes on
        # a different residual scale (NHiTS-style hierarchical refinement).
        # If False, every block sees only ``dec_in`` and blocks tend to
        # collapse into "one block does everything else outputs ~0".
        #
        # INTERPRETABILITY: the stack takes ONLY concept-derived inputs. No
        # raw feature access, no bypass around the concept layer. Ablating a
        # concept propagates linearly through every block. Concept-only MAE
        # (residual disabled, everything else on) still measures how much the
        # concepts alone explain -- the delta sum is a function of the concept
        # inputs by construction.
        decoder_n_blocks: int = 3,
        decoder_feedback: bool = True,
        sub_window_id_dim: int = 16,
        # NOTE: the ``residual_dim`` argument has been removed. There is NO
        # encoder-to-decoder residual pathway in this model any more. Every
        # bit of information reaching the sub-decoder passes through one of
        # the three concept bottlenecks (LOOKBACK, FORECAST-SUB-WINDOW,
        # FORECAST-GLOBAL) plus the sub-window id embedding (which carries
        # NO content -- it's a positional placeholder). This is a strict CBM
        # by construction: no bypass exists, so ablating a concept mechanically
        # affects the forecast. See __init__ body for the exact dec_in layout.
        sw_gru_hidden: int = 32,
        sw_summary_dim: int = 64,
        single_forecast_bottleneck: bool = False,
        use_rev_in: int = 1,
        norm_type: str = "B",
        # ---- Sub-window TIME-OF-DAY embedding ------------------------------
        # The forecast encoder's output is mean-pooled over each sub-window
        # before being fed to the per-sub-window concept predictor. For PRSA
        # at 1-hour resolution with pred_len=96 and n_subwindows=4 the
        # sub-window length is 24 -- exactly one daily cycle -- so the mean
        # of the hour-of-day channel over a sub-window is ALWAYS 11.5,
        # regardless of where the sub-window starts. Mean pooling therefore
        # destroys the only signal that distinguishes "morning sub-window"
        # from "evening sub-window", and the concept head literally cannot
        # learn ``is_morning_rush_block``.
        #
        # Fix: extract each sub-window's actual START HOUR from the input
        # tensor and CONCATENATE a small DETERMINISTIC SINUSOIDAL encoding of
        # that hour to ``z_per_sw[k]`` before the forecast-concept predictor
        # sees it.
        #
        # This design is stronger than the original additive-learned fix
        # because:
        #   (a) Concatenation lets the predictor learn an INDEPENDENT weight
        #       group on the hour dims. A pure-hour concept just zeroes its
        #       encoder-side weights and puts all mass on the hour-side
        #       weights -- no entanglement with the encoder features.
        #   (b) Sinusoidal is a FIXED, KNOWN representation of hour. The
        #       "morning direction" is a stable target for the predictor's
        #       weight vector, unlike a randomly-initialised nn.Embedding
        #       whose morning direction drifts throughout training.
        #   (c) 8 dims are enough (sin/cos at periods 24, 12, 6, 3 hours),
        #       much smaller than the previous mid_dim additive vector.
        #
        # ``calendar_hour_channel`` is the channel index of the hour column
        # inside the (B, T, in_dim) input tensor. For PRSA the input layout
        # is [year, month, day, hour, PM10, ..., PM2.5_masked] → channel 3.
        # Set to a negative number to DISABLE the hour augmentation
        # (legacy / ablation; will reintroduce the time-blindness bug).
        calendar_hour_channel: int = 3,
        # ``hour_encoding_dim`` MUST be an even number (sin + cos pairs).
        # Default 8 = 4 sin + 4 cos at periods [24, 12, 6, 3] hours -- covers
        # daily, half-day, quarter-day, and 3-hour cycles which is more than
        # enough resolution for hour-of-day concepts. Only used when
        # ``calendar_hour_channel >= 0``.
        hour_encoding_dim: int = 8,
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
        # fired it; only the input (concept-side input = concat of the
        # sub-window's pooled encoder features and its sinusoidal hour
        # encoding) differs. This mirrors how the static embedding matrix
        # was shared previously.
        #
        # Input dim is ``mid_dim + hour_encoding_dim`` (or just ``mid_dim`` if
        # the hour augmentation is disabled via calendar_hour_channel < 0).
        # Concatenating the sinusoidal hour encoding here rather than adding
        # it to z_per_sw[k] gives the predictor an INDEPENDENT weight group
        # on the hour dimensions -- a pure-hour concept just zeros its
        # encoder-side weights and puts all mass on the hour-side weights.
        # See the ``calendar_hour_channel`` / ``hour_encoding_dim`` docs in
        # __init__.
        n_fc_predictors = 1 if self.single_forecast_bottleneck else n_subwindows
        _use_hour = int(calendar_hour_channel) >= 0
        self.hour_encoding_dim = int(hour_encoding_dim) if _use_hour else 0
        if _use_hour and (self.hour_encoding_dim % 2 != 0):
            raise ValueError(
                f"hour_encoding_dim must be even (sin+cos pairs), got "
                f"{self.hour_encoding_dim}"
            )
        _fc_input_dim = mid_dim + self.hour_encoding_dim
        self.forecast_concept_predictors = nn.ModuleList([
            nn.Linear(_fc_input_dim, n_forecast_concepts)
            for _ in range(n_fc_predictors)
        ])
        self.forecast_emb_plus  = nn.Linear(_fc_input_dim, n_forecast_concepts * self.concept_emb_dim)
        self.forecast_emb_minus = nn.Linear(_fc_input_dim, n_forecast_concepts * self.concept_emb_dim)

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

        # ---- Sub-window TIME-OF-DAY encoding (deterministic sinusoidal) ----
        # The old learned nn.Embedding(24, mid_dim) has been replaced with a
        # deterministic sinusoidal encoding computed on-the-fly in forward().
        # No parameters are needed here. See ``calendar_hour_channel`` /
        # ``hour_encoding_dim`` docs in __init__ and the ``_sinusoidal_hour``
        # helper below.
        self.calendar_hour_channel = int(calendar_hour_channel)
        # self.hour_encoding_dim is already set above where the forecast
        # concept predictors were built with input dim mid_dim + hour_enc_dim.

        # ---- No residual channel (STRICT CBM) ----
        # There is no residual pathway from the encoder to the sub-decoder.
        # The residual_proj / residual_list machinery that used to exist here
        # has been removed entirely. See the "residual_dim was removed"
        # comment at the top of __init__.

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
        # dec_in layout (STRICT CBM -- every entry is either a concept-derived
        # context vector or a positional embedding; NO residual bypass):
        #   lookback CEM context    = K_lb         * d   (input-conditional)
        #   forecast-SW CEM context = K_fc         * d   (input-conditional, one per k)
        #   sw_gru context          = 2 * sw_gru_hidden  (cross-sub-window aware,
        #                                                 derived from fc_ctx; 0 if disabled)
        #   forecast-GLOBAL context = K_fc_global  * d   (input-conditional; 0 if disabled)
        #   sub-window id embedding = sw_id_dim          (positional; no content)
        # Output: T_sw forecast values. decoder_depth = number of Linear layers per block.
        dec_in_dim  = (lb_ctx_dim + fc_ctx_dim + sw_gru_out_dim + gfc_ctx_dim
                       + sub_window_id_dim)
        if decoder_depth < 1:
            raise ValueError(f"decoder_depth must be >= 1, got {decoder_depth}")
        if decoder_n_blocks < 1:
            raise ValueError(
                f"decoder_n_blocks must be >= 1, got {decoder_n_blocks}"
            )
        self.decoder_n_blocks = int(decoder_n_blocks)
        self.decoder_feedback = bool(decoder_feedback)
        _sw_out_size = self.sub_window_length * output_dim  # flat forecast size per block
        block_modules: list[nn.Module] = []
        for i in range(self.decoder_n_blocks):
            # Blocks i>0 with feedback also see the running forecast so far
            # (which is a function of concept-derived inputs). Block 0 sees
            # only dec_in.
            if self.decoder_feedback and i > 0:
                b_in = dec_in_dim + _sw_out_size
            else:
                b_in = dec_in_dim
            block_modules.append(ResidualDecoderBlock(
                in_dim=b_in, hidden=decoder_hidden, depth=decoder_depth,
                out_len=self.sub_window_length, out_dim=output_dim,
                dropout=dropout,
            ))
        self.decoder_blocks = nn.ModuleList(block_modules)

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
        disable_residual : NO-OP. The residual pathway has been removed
            from the model entirely; this argument is kept only for backward
            compatibility with old caller code (concept-only-MAE utilities,
            saved evaluation scripts, etc.). Passing True has zero effect on
            the output because there is no residual channel to disable.
            Legacy docstring for reference: (previously) if True, the residual channel was zeroed before
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

        # ---- Augment z_per_sw[k] with the sub-window's actual START HOUR ----
        # Mean-pooling over a 24-step sub-window destroys time-of-day info
        # (the mean of any cyclical channel over one full daily cycle is
        # constant), so without an extra time signal the forecast concept
        # head cannot tell morning sub-windows from evening ones.
        #
        # We CONCATENATE (not add) a deterministic sinusoidal encoding of
        # each sub-window's start hour to z_per_sw[k]. Concatenation is a
        # big win over the previous additive-learned design:
        #
        #   * Independent weight groups -- the predictor Linear has separate
        #     weights on the encoder-side vs the hour-side dims. A pure-hour
        #     concept just zeros its encoder-side weights and puts all mass
        #     on the hour-side ones; a pure-weather concept does the opposite.
        #     No entanglement, no competing gradients.
        #   * Deterministic sinusoidal -- the "morning direction" in the hour
        #     encoding is a KNOWN, FIXED value (specific sin/cos coefficients),
        #     not a random-init nn.Embedding vector that drifts during
        #     training. Much easier optimization target.
        #   * Tiny -- hour_encoding_dim defaults to 8, much smaller than the
        #     previous mid_dim additive vector, and no learned parameters.
        if (self.calendar_hour_channel >= 0
                and not self.single_forecast_bottleneck
                and self.hour_encoding_dim > 0):
            sw_len = self.sub_window_length
            # Index of each sub-window's FIRST step inside the full input.
            sw_start_idx = self.input_length + torch.arange(
                self.n_subwindows, device=x.device,
            ) * sw_len                                                       # (N,)
            # x[:, sw_start_idx, hour_ch] -> (B, N) integer hours 0..23.
            # The hour column in the input is the RAW (non-z-scored) value
            # straight out of the CSV per LOF_Dataset.get_data, so casting
            # to .long() and clamping is safe.
            start_hours = (
                x[:, sw_start_idx, self.calendar_hour_channel]
                .long().clamp(0, 23)
            )                                                                # (B, N)
            sw_hour_vecs = _sinusoidal_hour_encoding(
                start_hours, self.hour_encoding_dim,
            )                                                                # (B, N, hour_enc_dim)
            z_per_sw = [
                torch.cat([z_per_sw[k], sw_hour_vecs[:, k, :]], dim=-1)
                for k in range(self.n_subwindows)
            ]

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

        # (Residual channel removed -- this is a strict CBM.
        #  ``disable_residual`` is retained as a no-op arg so old callers /
        #  concept-only-MAE eval code paths don't error, but there is no
        #  residual pathway left to disable.)

        # 4) Shared sub-decoder, parameterized by sub-window id.
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
            dec_in = torch.cat(parts, dim=-1)
            # ---- Residual sub-decoder stack (NHiTS-style) ----
            # Each block produces a signed delta forecast; the deltas sum
            # into the running forecast. Every block operates on concept-
            # derived context only, preserving the concept bottleneck as the
            # single information pathway to the forecast.
            running = dec_in.new_zeros(
                (B, self.sub_window_length, self.output_dim)
            )
            for bi, block in enumerate(self.decoder_blocks):
                if self.decoder_feedback and bi > 0:
                    block_in = torch.cat(
                        [dec_in, running.reshape(B, -1)], dim=-1,
                    )
                else:
                    block_in = dec_in
                delta = block(block_in)                          # (B, T_sw, O)
                running = running + delta
            sub_y = running
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
            # (residual removed -- strict CBM, no bypass to regularize)
        }
