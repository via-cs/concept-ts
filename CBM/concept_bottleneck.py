"""
ConceptBottleneckModel — 1 + N concept bottlenecks for interpretable
multivariate time-series forecasting.

Architecture
------------

    x  in (B, T_in + T_out, C)
       |
       +-- lookback_encoder( x[:, :T_in, :] )    # SEES ONLY THE HISTORY
       |     enc_lb in (B, T_in, D)              # target RevIN'd (revin_all)
       |     z_global = AttnPool(enc_lb)         # (B, D) -> LOOKBACK bottleneck
       |
       +-- forecast_encoder( x[:, T_in:, :] )    # SEES ONLY THE FORECAST WINDOW
             enc_fc in (B, T_out, D)             # target masked (mask_all)
             z_sw_k            = AttnPool(enc_fc[:, k*T_sw:(k+1)*T_sw, :])  # per sub-window
             z_global_forecast = AttnPool(enc_fc)     # (B, D) -> GLOBAL-FORECAST bottleneck

    Two SEPARATE-WEIGHT encoders give a HARD lookback/forecast information
    barrier: lookback concepts are provably a function of the history, and
    forecast concepts a function of the forecast window. (use_time_pool=False
    on both; the CBM does the pooling.)

    Temporal pooling is a LEARNED, ORDER-DEPENDENT attention pool (AttnPool,
    additive Bahdanau attention over the time axis) -- NOT a plain mean --
    so order-sensitive concepts (is_pm25_increasing / _declining, etc.) are
    visible to the concept predictors. Training also uses CEM Random-
    Intervention (randint_p): during training only, each concept activation
    is swapped for its ground-truth label with prob randint_p before entering
    the +/- embedding mix, reducing information leakage through the concept
    activation channel.
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
# Attention-weighted temporal pool
# ---------------------------------------------------------------------------
class AttentionPool(nn.Module):
    """Bahdanau-style additive attention pool over the TIME axis.

    Replaces a permutation-invariant ``x.mean(dim=1)`` with a learned
    weighted average where the weights are computed from x itself. This
    lets downstream layers see an ORDER-DEPENDENT summary of a temporal
    slice -- the same slice with its timesteps shuffled would yield the
    same mean pool, but a different attention pool.

    Why this matters here: the CBM's concept predictors sit on top of
    these pooled vectors. Concepts whose truth values depend on ORDER
    (e.g. ``is_pm25_increasing``, ``is_pm25_declining``,
    ``peak_in_first_half``) cannot be reliably decoded from a mean pool
    because two temporal slices with identical means and different
    orderings look identical to the predictor. An attention pool can
    up-weight the endpoints, or the trend-changing sections, so the
    pooled vector actually reflects the ordering.

    Design
    ------
    scores = W2 · tanh(W1 · x)           # (B, T, 1)
    alpha  = softmax(scores, dim=T)      # (B, T, 1)  positional weights
    pooled = sum_t alpha_t * x_t         # (B, D)

    ``hidden`` is the intermediate width of the attention MLP. Small by
    design (default 32) -- this module is a pooling layer, not a mixer.

    ``mask`` (optional) can be used to exclude timesteps (e.g. padding).
    True marks a VALID position; False marks a masked-out one.
    """

    def __init__(self, d_model: int, hidden: int = 32, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(int(d_model), int(hidden)),
            nn.Tanh(),
            nn.Linear(int(hidden), 1),
        )
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else None

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, D) -> pooled: (B, D)
        scores = self.proj(x)                                                    # (B, T, 1)
        if mask is not None:
            # mask: (B, T) with True = valid
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(scores, dim=1)                                     # (B, T, 1)
        if self.dropout is not None:
            alpha = self.dropout(alpha)
        pooled = (alpha * x).sum(dim=1)                                          # (B, D)
        return pooled


# ---------------------------------------------------------------------------
# Static (input-independent) concept embedding head
# ---------------------------------------------------------------------------
class _StaticConceptEmb(nn.Module):
    """Input-INDEPENDENT concept embedding head for the CEM ± mixing.

    Drop-in replacement for ``nn.Linear(in_dim, K*d)``: it accepts the encoder
    feature but IGNORES it, returning a learned ``(K*d)`` parameter broadcast
    over the batch. So the per-concept context

        ĉ_i = a_i · e+_i + (1 - a_i) · e-_i
            = e-_i + a_i · (e+_i - e-_i)

    depends on the input ONLY through the scalar activation ``a_i`` (a fixed
    vector plus a scalar times a fixed direction). This converts the full CEM
    -- whose input-conditional ``e+/e-(z)`` form a K*d-dimensional continuous
    channel of encoder information straight to the decoder -- into a
    classic-CBM-style bottleneck whose only input-dependent signal is the K
    concept activations. Used to test / remove the embedding-path leakage.

    Output shape ``(B, K*d)`` matches the Linear path exactly, so callers can
    ``.reshape(B, K, d)`` unchanged.
    """

    def __init__(self, n_concepts: int, emb_dim: int):
        super().__init__()
        self.out_dim = int(n_concepts) * int(emb_dim)
        # Small random init so the K*d columns (and e+ vs e-) start distinct.
        self.emb = nn.Parameter(torch.randn(self.out_dim) * 0.1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z is used only for batch size / device; its VALUES are ignored.
        return self.emb.unsqueeze(0).expand(z.shape[0], -1)     # (B, K*d)


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
        # ---- Experiment mode -----------------------------------------------
        # "future_aware" (default): current behavior. Forecast encoder sees
        #     x[:, T_in:, :] (future covariates). Three concept bottlenecks:
        #     LOOKBACK (global), FORECAST-SUB-WINDOW, FORECAST-GLOBAL.
        # "lookback_only": strict no-future-features setting. Forecast
        #     encoder is skipped entirely. Two concept bottlenecks:
        #     LOOKBACK (global) and LOOKBACK-SUB-WINDOW (new). Decoder still
        #     produces per-forecast-sub-window forecasts, but everything it
        #     conditions on is derived from the lookback alone.
        experiment_mode: str = "future_aware",
        # LOOKBACK sub-window bottleneck config (only used when
        # experiment_mode='lookback_only'). n_lookback_subwindows defaults to
        # n_subwindows for symmetry. n_lookback_subwindow_concepts=0 skips
        # the LB-SW bottleneck (in that mode, only LOOKBACK-GLOBAL remains).
        n_lookback_subwindows: int = None,
        n_lookback_subwindow_concepts: int = 0,
        concept_emb_dim: int = 16,
        # When True, the CEM ± embedding heads (e+/e-) are INPUT-INDEPENDENT
        # learned vectors (see _StaticConceptEmb) instead of Linear(z -> K*d).
        # This closes the embedding-path leak: the decoder's only input-
        # dependent signal from each bottleneck becomes the K concept
        # activations (classic-CBM bottleneck). Default False = full CEM.
        static_concept_emb: bool = False,
        # Optional hybrid-CBM residual channel (Option 1). When residual_dim>0,
        # a small learned Linear(encoder feature -> residual_dim) runs from the
        # LOOKBACK encoder (z_global; carries the recent PM2.5 level) and, in
        # future_aware mode, from each FORECAST sub-window (z_per_sw[k]; carries
        # that block's covariate magnitudes). These are concatenated into the
        # decoder input and are meant to be L2-penalized by the trainer
        # (alpha_residual_reg) so the decoder uses them ONLY for the continuous
        # level/scale the concepts can't encode. residual_dim=0 => strict CBM
        # (no bypass). forward(..., disable_residual=True) zeroes them to give
        # the concept-only forecast.
        residual_dim: int = 0,
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
        # ---- Attention-weighted temporal pool ------------------------------
        # Replaces the sequence-destroying ``mean(dim=1)`` pool that used to
        # feed the concept predictors. Each pooled tensor (LOOKBACK global,
        # FORECAST global, per-FORECAST-SW, per-LOOKBACK-SW) now goes through
        # an AttentionPool module which computes learned per-timestep weights
        # via a two-layer additive-attention MLP (Bahdanau-style). This gives
        # the concept predictors an ORDER-DEPENDENT summary of each temporal
        # slice, so concepts like ``is_pm25_increasing`` are no longer
        # invisible to the concept head at pool time. ``attn_pool_hidden`` is
        # the intermediate width of the attention MLP -- small by design.
        attn_pool_hidden: int = 32,
        # ---- Random-Intervention (RandInt) training -------------------------
        # CEM training trick from Zarlenga et al. 2022. During training only,
        # each concept activation is REPLACED with its ground-truth label
        # with probability ``randint_p`` (independently per concept, per
        # sample, per training step). The intervened activation flows into
        # the CEM ± embedding mixing, so the decoder learns to correctly
        # respond to concept-activation VALUES (0 vs 1) rather than to
        # information leaked through the two embedding heads.
        #
        # Set ``randint_p = 0.0`` to disable and recover the original
        # CEM-without-intervention behavior. Typical values are 0.1 - 0.5;
        # the CEM paper defaults to 0.25. Interventions are only applied
        # when ``concept_labels`` is passed to ``forward``, so evaluation
        # never intervenes -- only training does.
        randint_p: float = 0.0,
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
        self.static_concept_emb  = bool(static_concept_emb)
        self.single_forecast_bottleneck = bool(single_forecast_bottleneck)
        self.input_length = int(input_length)

        # Experiment mode + lookback-sub-window config.
        if experiment_mode not in ("future_aware", "lookback_only"):
            raise ValueError(
                f"experiment_mode must be 'future_aware' or 'lookback_only', "
                f"got {experiment_mode!r}."
            )
        self.experiment_mode = experiment_mode
        _n_lb_sw = (n_lookback_subwindows if n_lookback_subwindows is not None
                    else n_subwindows)
        if input_length % _n_lb_sw != 0:
            raise ValueError(
                f"input_length ({input_length}) must be divisible by "
                f"n_lookback_subwindows ({_n_lb_sw})."
            )
        self.n_lookback_subwindows = int(_n_lb_sw)
        self.lookback_sub_window_length = input_length // self.n_lookback_subwindows
        self.n_lookback_subwindow_concepts = int(n_lookback_subwindow_concepts)

        # In lookback_only mode, forecast-side concept counts are ZEROED out
        # regardless of what the constructor was passed. All forecast-side
        # sub-modules will not be constructed below.
        self._forecast_side_active = (self.experiment_mode == "future_aware")
        if not self._forecast_side_active:
            self.n_forecast_concepts        = 0
            self.n_global_forecast_concepts = 0

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
        if self._forecast_side_active:
            self.forecast_encoder = _make_window_encoder(
                window_len=output_length, target_mode="mask_all",
                use_rev_in_flag=0,
            )
        else:
            # experiment_mode='lookback_only': no forecast encoder at all.
            # The lookback encoder alone supplies all concept-side signal.
            self.forecast_encoder = None

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
        def _make_concept_emb(in_dim: int, n_concepts: int) -> nn.Module:
            """CEM ± embedding head: input-conditional Linear (full CEM) or an
            input-independent static head (--static_concept_emb). Both return
            (B, K*d) so downstream reshape/mixing is identical."""
            if self.static_concept_emb:
                return _StaticConceptEmb(n_concepts, self.concept_emb_dim)
            return nn.Linear(in_dim, n_concepts * self.concept_emb_dim)

        self.lookback_concept_predictor = nn.Linear(mid_dim, n_lookback_concepts)
        self.lookback_emb_plus  = _make_concept_emb(mid_dim, n_lookback_concepts)
        self.lookback_emb_minus = _make_concept_emb(mid_dim, n_lookback_concepts)

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
        _use_hour = int(calendar_hour_channel) >= 0
        self.hour_encoding_dim = int(hour_encoding_dim) if _use_hour else 0
        if _use_hour and (self.hour_encoding_dim % 2 != 0):
            raise ValueError(
                f"hour_encoding_dim must be even (sin+cos pairs), got "
                f"{self.hour_encoding_dim}"
            )
        _fc_input_dim = mid_dim + self.hour_encoding_dim

        # Forecast sub-window / global concept predictors ONLY in future_aware.
        if self._forecast_side_active:
            n_fc_predictors = 1 if self.single_forecast_bottleneck else n_subwindows
            self.forecast_concept_predictors = nn.ModuleList([
                nn.Linear(_fc_input_dim, n_forecast_concepts)
                for _ in range(n_fc_predictors)
            ])
            self.forecast_emb_plus  = _make_concept_emb(_fc_input_dim, n_forecast_concepts)
            self.forecast_emb_minus = _make_concept_emb(_fc_input_dim, n_forecast_concepts)
        else:
            self.forecast_concept_predictors = nn.ModuleList()
            self.forecast_emb_plus  = None
            self.forecast_emb_minus = None

        # LOOKBACK-SUB-WINDOW concept predictors ONLY when lookback_only + K_lb_sw>0.
        # Mirrors the forecast-side wiring but reads from z_per_lb_sw[k] (per
        # lookback sub-window pooled feature) + sinusoidal hour encoding of
        # the lookback sub-window's start hour.
        if (self.experiment_mode == "lookback_only"
                and self.n_lookback_subwindow_concepts > 0):
            self.lookback_sw_concept_predictors = nn.ModuleList([
                nn.Linear(_fc_input_dim, self.n_lookback_subwindow_concepts)
                for _ in range(self.n_lookback_subwindows)
            ])
            self.lookback_sw_emb_plus  = _make_concept_emb(
                _fc_input_dim, self.n_lookback_subwindow_concepts,
            )
            self.lookback_sw_emb_minus = _make_concept_emb(
                _fc_input_dim, self.n_lookback_subwindow_concepts,
            )
        else:
            self.lookback_sw_concept_predictors = nn.ModuleList()
            self.lookback_sw_emb_plus  = None
            self.lookback_sw_emb_minus = None

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
            self.global_forecast_emb_plus  = _make_concept_emb(
                mid_dim, self.n_global_forecast_concepts,
            )
            self.global_forecast_emb_minus = _make_concept_emb(
                mid_dim, self.n_global_forecast_concepts,
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

        # ---- Attention-weighted temporal pools ----
        # One pool per temporal-scale × experiment-mode combination:
        #   * LOOKBACK-global pool  -- always built (used in both modes)
        #   * FORECAST-global pool  -- future_aware only
        #   * FORECAST-SW pool      -- future_aware only, SHARED across the N
        #                              forecast sub-windows (they are all the
        #                              same task, applied to same-length slices)
        #   * LOOKBACK-SW pool      -- lookback_only only, SHARED across the
        #                              N_lb lookback sub-windows
        # Sharing the sub-window pools across positions keeps the parameter
        # count low; each pool learns a POSITION-INDEPENDENT rule for "which
        # timesteps within a sub-window matter most" and lets the concept
        # predictor at each sub-window see its own attention-weighted summary.
        self.attn_pool_hidden = int(attn_pool_hidden)
        self.attn_pool_lookback_global = AttentionPool(mid_dim, hidden=self.attn_pool_hidden)
        if self._forecast_side_active:
            self.attn_pool_forecast_global = AttentionPool(mid_dim, hidden=self.attn_pool_hidden)
            self.attn_pool_forecast_sw     = AttentionPool(mid_dim, hidden=self.attn_pool_hidden)
        else:
            self.attn_pool_forecast_global = None
            self.attn_pool_forecast_sw     = None
        if (self.experiment_mode == "lookback_only"
                and self.n_lookback_subwindow_concepts > 0):
            self.attn_pool_lookback_sw = AttentionPool(mid_dim, hidden=self.attn_pool_hidden)
        else:
            self.attn_pool_lookback_sw = None

        # ---- RandInt state ----
        # Intervention probability; forward() reads this to gate the
        # per-concept swap of predicted activation with ground-truth label.
        self.randint_p = float(randint_p)
        if not (0.0 <= self.randint_p <= 1.0):
            raise ValueError(
                f"randint_p must be in [0, 1], got {self.randint_p}"
            )

        # ---- Optional hybrid residual channel (Option 1) ----
        # A small learned bypass from the encoder(s) to the sub-decoder that
        # carries the continuous level/scale the (near-binary) concepts cannot.
        #   residual_lb : Linear(mid_dim -> r) on z_global  -> shared across
        #                 sub-windows (the recent PM2.5 level).
        #   residual_fc : Linear(mid_dim -> r) on z_per_sw[k] -> per forecast
        #                 sub-window (that block's covariate magnitudes);
        #                 future_aware only.
        # Penalized in the loss (alpha_residual_reg * ||residual||^2) and
        # zeroable at eval (disable_residual) for the concept-only forecast.
        self.residual_dim = int(residual_dim)
        if self.residual_dim > 0:
            self.residual_lb_proj = nn.Linear(mid_dim, self.residual_dim)
            if self._forecast_side_active:
                self.residual_fc_proj = nn.Linear(mid_dim, self.residual_dim)
            else:
                self.residual_fc_proj = None
        else:
            self.residual_lb_proj = None
            self.residual_fc_proj = None
        # Residual contribution to the decoder input width:
        #   future_aware : lb residual + per-sw fc residual = 2 * r
        #   lookback_only: lb residual only               =     r
        if self.residual_dim > 0:
            residual_in_dim = (2 * self.residual_dim
                               if self._forecast_side_active
                               else self.residual_dim)
        else:
            residual_in_dim = 0

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
        lb_ctx_dim    = n_lookback_concepts                     * self.concept_emb_dim
        fc_ctx_dim    = self.n_forecast_concepts                * self.concept_emb_dim
        gfc_ctx_dim   = self.n_global_forecast_concepts         * self.concept_emb_dim
        lb_sw_ctx_dim = self.n_lookback_subwindow_concepts      * self.concept_emb_dim

        # ---- Sub-window context BiGRU (sequence-aware refinement) ----
        # In future_aware mode: BiGRU walks over FORECAST sub-window concept
        #     contexts. Its refined output per forecast sub-window is fed into
        #     the decoder for that sub-window.
        # In lookback_only mode: BiGRU walks over LOOKBACK sub-window concept
        #     contexts. The refined output is FINAL-STATE-POOLED and reused as
        #     a single shared "lookback sequence summary" for every forecast
        #     sub-window's decoding, because forecast sub-windows have no
        #     per-position concept context of their own in this mode.
        self.sw_gru_hidden = int(sw_gru_hidden)
        _bigru_input_ctx_dim = (
            lb_sw_ctx_dim if self.experiment_mode == "lookback_only"
            else fc_ctx_dim
        )
        _bigru_enabled = (
            self.sw_gru_hidden > 0
            and not self.single_forecast_bottleneck
            and _bigru_input_ctx_dim > 0
        )
        if _bigru_enabled:
            self.sw_summary_dim = int(sw_summary_dim)
            self.sw_context_summary = nn.Linear(_bigru_input_ctx_dim, self.sw_summary_dim)
            self.sw_context_gru = nn.GRU(
                self.sw_summary_dim, self.sw_gru_hidden,
                bidirectional=True, batch_first=True,
            )
            sw_gru_out_dim = self.sw_gru_hidden * 2
        else:
            sw_gru_out_dim = 0

        # ---- Shared sub-decoder ----
        # dec_in layout (STRICT CBM -- every entry is either a concept-derived
        # context vector or a positional embedding; NO residual bypass).
        #
        # In experiment_mode='future_aware':
        #   lookback CEM context     = K_lb         * d   (input-conditional)
        #   forecast-SW CEM context  = K_fc         * d   (input-conditional, one per k)
        #   sw_gru context (per k)   = 2 * sw_gru_hidden  (BiGRU over FORECAST sub-windows)
        #   forecast-GLOBAL context  = K_fc_global  * d   (input-conditional; 0 if disabled)
        #   sub-window id embedding  = sw_id_dim          (positional)
        #
        # In experiment_mode='lookback_only':
        #   lookback CEM context     = K_lb         * d   (input-conditional)
        #   sw_gru context (shared)  = 2 * sw_gru_hidden  (BiGRU over LOOKBACK sub-windows,
        #                                                  final-state summary reused for
        #                                                  every forecast sub-window)
        #   sub-window id embedding  = sw_id_dim          (positional)
        #   (no forecast-SW / forecast-GLOBAL contexts -- those bottlenecks are
        #    disabled in lookback_only mode)
        #
        # Output: T_sw forecast values. decoder_depth = number of Linear layers per block.
        dec_in_dim  = (lb_ctx_dim + fc_ctx_dim + sw_gru_out_dim + gfc_ctx_dim
                       + sub_window_id_dim + residual_in_dim)
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
        z_global           : (B, D) -- ATTENTION-weighted pool over the
                             LOOKBACK encoder's per-timestep features. Fed
                             to the LOOKBACK concept bottleneck. Provably a
                             function of the history alone (the lookback
                             encoder never sees the forecast window).
        z_per_sw           : list of N (B, D) tensors -- ATTENTION-weighted
                             pool over each forecast sub-window slice of
                             the FORECAST encoder's features. Fed to the
                             per-sub-window forecast bottlenecks.
        z_global_forecast  : (B, D) -- ATTENTION-weighted pool over the
                             FORECAST encoder's whole output. Fed to the
                             GLOBAL-FORECAST bottleneck.

        Both encoders run with use_time_pool=False, so enc_lb / enc_fc are
        (B, T, D). A 2D fallback is kept for robustness if an encoder ever
        returns a pooled (B, D) vector; in that fallback the attention pool
        is skipped (there is nothing to attend over).
        """
        # ---- lookback summary ----
        if enc_lb.dim() == 2:
            z_global = enc_lb
        elif enc_lb.dim() == 3:
            z_global = self.attn_pool_lookback_global(enc_lb)   # (B, D)
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

        z_global_forecast = self.attn_pool_forecast_global(enc_fc)  # (B, D)
        sw_len = self.sub_window_length
        z_per_sw = [
            self.attn_pool_forecast_sw(
                enc_fc[:, k * sw_len : (k + 1) * sw_len, :]
            )
            for k in range(self.n_subwindows)
        ]
        return z_global, z_per_sw, z_global_forecast

    def _pool_lookback_encoder(self, enc_lb: torch.Tensor
                                ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Pool the LOOKBACK encoder's per-timestep features into:

          z_global    : (B, D) ATTENTION-weighted pool over the whole lookback
                        -> LOOKBACK-GLOBAL concept bottleneck
          z_per_lb_sw : list of N_lb_sw (B, D) tensors, one ATTENTION-weighted
                        pool per lookback sub-window slice
                        -> LOOKBACK-SUB-WINDOW concept bottleneck

        Only called in experiment_mode='lookback_only'. Symmetric to how
        _pool_encoder_features slices the forecast encoder in future_aware
        mode, but the slicing here is over the LOOKBACK axis (input_length,
        n_lookback_subwindows).

        If ``self.attn_pool_lookback_sw`` was not built (because
        ``n_lookback_subwindow_concepts == 0``), the per-sub-window list
        falls back to a mean pool so callers still get well-formed tensors.
        """
        if enc_lb.dim() == 2:
            # Encoder returned pooled (B, D). Can't split by sub-window.
            return enc_lb, [enc_lb] * self.n_lookback_subwindows
        if enc_lb.dim() != 3:
            raise ValueError(
                f"lookback encoder output has unexpected ndim={enc_lb.dim()}; "
                f"expected 2 or 3."
            )
        T = enc_lb.shape[1]
        if T != self.input_length:
            enc_lb = enc_lb[:, -self.input_length:, :]

        z_global = self.attn_pool_lookback_global(enc_lb)                        # (B, D)

        lb_sw_len = self.lookback_sub_window_length
        if self.attn_pool_lookback_sw is not None:
            z_per_lb_sw = [
                self.attn_pool_lookback_sw(
                    enc_lb[:, k * lb_sw_len : (k + 1) * lb_sw_len, :]
                )
                for k in range(self.n_lookback_subwindows)
            ]
        else:
            # No LB-SW concept bottleneck configured; downstream code will
            # produce empty concept tensors, so this list is unused. Keep the
            # legacy mean pool as a well-defined placeholder.
            z_per_lb_sw = [
                enc_lb[:, k * lb_sw_len : (k + 1) * lb_sw_len, :].mean(dim=1)
                for k in range(self.n_lookback_subwindows)
            ]
        return z_global, z_per_lb_sw

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # RandInt helper
    # ------------------------------------------------------------------
    def _gate_activation(self, acts: torch.Tensor,
                         labels: torch.Tensor | None,
                         intervention: str | None = None) -> torch.Tensor:
        """Decide which activation feeds the CEM ± embedding mixing.

        Priority:

        1. Eval-time DETERMINISTIC INTERVENTION (leak diagnostic), when
           ``intervention`` is not None -- the activation of EVERY concept is
           replaced (probability 1.0):
             * "oracle" → ground-truth labels          (needs labels)
             * "flip"   → 1 - labels                    (needs labels)
             * "random" → Bernoulli(0.5)                (label-independent)
           If a layer has no labels this batch (labels is None) the "oracle"/
           "flip" modes leave the prediction unchanged for that layer.

           Only the activation CHANNEL is intervened; the ± embeddings
           e+/e-(z) are NOT touched. So comparing forecast MAE under these
           modes vs. baseline measures how much the forecast actually depends
           on the concept-activation channel rather than leaking through the
           embeddings: a large "flip"/"random" degradation ⇒ concepts matter
           (leak contained); near-zero degradation ⇒ the decoder bypasses the
           activations via e+/e- (leak).

        2. Training RandInt (Zarlenga et al. 2022): probabilistic per-element
           swap with ground-truth labels at rate ``self.randint_p``.

        3. Otherwise: the model's own activations, unchanged.

        Never alters the raw acts/logits the model returns as outputs.

        Args
        ----
        acts         : (B, ..., K) sigmoid activations in [0, 1]
        labels       : (B, ..., K) binary {0, 1} labels, or None
        intervention : None | "oracle" | "flip" | "random"
        """
        if intervention is not None:
            if intervention == "random":
                return torch.bernoulli(torch.full_like(acts, 0.5))
            if labels is None:
                return acts
            labels = labels.to(acts.dtype)
            if intervention == "oracle":
                return labels
            if intervention == "flip":
                return 1.0 - labels
            raise ValueError(
                f"unknown intervention {intervention!r}; expected "
                f"None | 'oracle' | 'flip' | 'random'."
            )
        if (not self.training
                or labels is None
                or self.randint_p <= 0.0):
            return acts
        mask = torch.bernoulli(torch.full_like(acts, float(self.randint_p)))
        return mask * labels.to(acts.dtype) + (1.0 - mask) * acts

    def forward(self, x: torch.Tensor, disable_residual: bool = False,
                concept_labels: dict | None = None,
                intervention: str | None = None) -> dict:
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
        concept_labels : optional dict enabling CEM RandInt training.
            When present AND the model is in ``.train()`` mode AND
            ``self.randint_p > 0``, per-concept ground-truth labels are
            swapped into the corresponding activations with probability
            ``randint_p`` before they enter the CEM ± embedding mixing.
            Keys (all optional -- any missing key skips RandInt for that
            layer):

              * "lb"    : (B, K_lb)                lookback concept labels
              * "fc"    : (B, N_sw,  K_fc)         per-forecast-SW labels
              * "gfc"   : (B, K_gfc)               global-forecast labels
              * "lb_sw" : (B, N_lb_sw, K_lb_sw)    per-lookback-SW labels

            Callers pass this dict during TRAINING only; pass None during
            eval / inference (or simply omit it) so the model uses its own
            predicted activations end-to-end.
        """
        if concept_labels is None:
            concept_labels = {}
        B = x.shape[0]
        device = x.device

        # 1) Encode. In future_aware: two separate encoders over disjoint
        #    windows. In lookback_only: only the lookback encoder is used;
        #    the forecast encoder was not even constructed.
        x_lb = x[:, : self.input_length, :].clone()       # lookback window
        _, enc_lb = self.lookback_encoder(x_lb)           # (B, T_in,  D)

        if self._forecast_side_active:
            x_fc = x[:, self.input_length :, :].clone()   # forecast window
            _, enc_fc = self.forecast_encoder(x_fc)       # (B, T_out, D)
            z_global, z_per_sw, z_global_forecast = self._pool_encoder_features(
                enc_lb, enc_fc,
            )
            # z_per_lb_sw not needed in future_aware mode.
            z_per_lb_sw = []
        else:
            # lookback_only mode: no forecast encoder, no forecast features.
            z_global, z_per_lb_sw = self._pool_lookback_encoder(enc_lb)
            z_per_sw          = []
            z_global_forecast = z_global   # unused; kept for API symmetry

        # ---- Augment per-sub-window features with sinusoidal HOUR encoding ---
        # In future_aware mode: augment z_per_sw (per FORECAST sub-window),
        # extracting the hour from the FORECAST slice of x.
        # In lookback_only mode: augment z_per_lb_sw (per LOOKBACK sub-window),
        # extracting the hour from the LOOKBACK slice of x.
        # See __init__ for the design rationale (independent weight groups +
        # deterministic sinusoidal + tiny param footprint).
        if (self.calendar_hour_channel >= 0
                and not self.single_forecast_bottleneck
                and self.hour_encoding_dim > 0):
            if self._forecast_side_active:
                # forecast sub-window start hours
                sw_len_ = self.sub_window_length
                n_sw_   = self.n_subwindows
                base_offset = self.input_length
                sw_start_idx = base_offset + torch.arange(
                    n_sw_, device=x.device,
                ) * sw_len_                                                    # (N,)
                start_hours = (
                    x[:, sw_start_idx, self.calendar_hour_channel]
                    .long().clamp(0, 23)
                )                                                              # (B, N)
                sw_hour_vecs = _sinusoidal_hour_encoding(
                    start_hours, self.hour_encoding_dim,
                )                                                              # (B, N, hour_enc_dim)
                z_per_sw = [
                    torch.cat([z_per_sw[k], sw_hour_vecs[:, k, :]], dim=-1)
                    for k in range(n_sw_)
                ]
            else:
                # lookback sub-window start hours (from the LOOKBACK slice)
                lb_sw_len_ = self.lookback_sub_window_length
                n_lb_sw_   = self.n_lookback_subwindows
                lb_sw_start_idx = torch.arange(
                    n_lb_sw_, device=x.device,
                ) * lb_sw_len_                                                  # (N_lb_sw,)
                lb_start_hours = (
                    x[:, lb_sw_start_idx, self.calendar_hour_channel]
                    .long().clamp(0, 23)
                )                                                              # (B, N_lb_sw)
                lb_sw_hour_vecs = _sinusoidal_hour_encoding(
                    lb_start_hours, self.hour_encoding_dim,
                )                                                              # (B, N_lb_sw, hour_enc_dim)
                z_per_lb_sw = [
                    torch.cat([z_per_lb_sw[k], lb_sw_hour_vecs[:, k, :]], dim=-1)
                    for k in range(n_lb_sw_)
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
        # RandInt: per-concept swap of activation with ground-truth label
        # during training; identity during eval. lb_acts (the model's own
        # prediction) is what we RETURN, but lb_acts_used flows into the CEM
        # mixing so the decoder learns to interpret activation VALUES.
        lb_acts_used = self._gate_activation(lb_acts, concept_labels.get("lb"), intervention)
        a_lb = lb_acts_used.unsqueeze(-1)                       # (B, K_lb, 1)
        lb_ctx_3d = a_lb * lb_eplus + (1.0 - a_lb) * lb_eminus  # (B, K_lb, d)
        lb_ctx    = lb_ctx_3d.reshape(B, -1)                    # (B, K_lb * d)

        # 2b) Global forecast concept layer (OPTIONAL, future_aware only).
        # Reads the FORECAST encoder's whole-window summary to produce one
        # concept activation pattern characterizing the WHOLE forecast window.
        # Skipped entirely in lookback_only mode.
        if self._forecast_side_active and self.n_global_forecast_concepts > 0:
            gfc_logits = self.global_forecast_concept_predictor(z_global_forecast)   # (B, K_gfc)
            gfc_acts   = torch.sigmoid(gfc_logits)
            g_ep = self.global_forecast_emb_plus(z_global_forecast).reshape(
                B, self.n_global_forecast_concepts, self.concept_emb_dim,
            )
            g_em = self.global_forecast_emb_minus(z_global_forecast).reshape(
                B, self.n_global_forecast_concepts, self.concept_emb_dim,
            )
            # RandInt on global-forecast activations.
            gfc_acts_used = self._gate_activation(gfc_acts, concept_labels.get("gfc"), intervention)
            a_g  = gfc_acts_used.unsqueeze(-1)                               # (B, K_gfc, 1)
            gfc_ctx_3d = a_g * g_ep + (1.0 - a_g) * g_em                     # (B, K_gfc, d)
            gfc_ctx    = gfc_ctx_3d.reshape(B, -1)                           # (B, K_gfc * d)
        else:
            gfc_logits = x.new_zeros((B, 0))
            gfc_acts   = x.new_zeros((B, 0))
            gfc_ctx    = x.new_zeros((B, 0))

        # 2c) LOOKBACK-SUB-WINDOW concept layer (lookback_only only).
        # Per-lookback-sub-window predictor + shared CEM ± MLPs. Structure
        # mirrors the forecast sub-window layer. Produces:
        #   lb_sw_logits : (B, N_lb_sw, K_lb_sw)  -- for BCE loss
        #   lb_sw_ctx_list: list of N_lb_sw (B, K_lb_sw * d) tensors --
        #                   fed to the BiGRU
        lb_sw_ctx_list: list = []
        if (self.experiment_mode == "lookback_only"
                and self.n_lookback_subwindow_concepts > 0):
            lb_sw_logits_list = []
            lb_sw_acts_list   = []
            lb_sw_labels_all = concept_labels.get("lb_sw")  # (B, N_lb_sw, K_lb_sw) or None
            for k in range(self.n_lookback_subwindows):
                lg = self.lookback_sw_concept_predictors[k](z_per_lb_sw[k])    # (B, K_lb_sw)
                ac = torch.sigmoid(lg)                                          # (B, K_lb_sw)
                ep = self.lookback_sw_emb_plus(z_per_lb_sw[k]).reshape(
                    B, self.n_lookback_subwindow_concepts, self.concept_emb_dim,
                )
                em = self.lookback_sw_emb_minus(z_per_lb_sw[k]).reshape(
                    B, self.n_lookback_subwindow_concepts, self.concept_emb_dim,
                )
                # RandInt on per-lookback-SW activations using the k-th slice
                # of the (B, N_lb_sw, K_lb_sw) label tensor.
                labels_k = (
                    lb_sw_labels_all[:, k, :] if lb_sw_labels_all is not None else None
                )
                ac_used = self._gate_activation(ac, labels_k, intervention)
                a_k = ac_used.unsqueeze(-1)                                     # (B, K_lb_sw, 1)
                ctx_3d_k = a_k * ep + (1.0 - a_k) * em                          # (B, K_lb_sw, d)
                ctx_k = ctx_3d_k.reshape(B, -1)                                 # (B, K_lb_sw * d)
                lb_sw_logits_list.append(lg)
                lb_sw_acts_list.append(ac)
                lb_sw_ctx_list.append(ctx_k)
            lb_sw_logits = torch.stack(lb_sw_logits_list, dim=1)                # (B, N_lb_sw, K_lb_sw)
            lb_sw_acts   = torch.stack(lb_sw_acts_list,   dim=1)                # (B, N_lb_sw, K_lb_sw)
        else:
            lb_sw_logits = x.new_zeros((B, 0, 0))
            lb_sw_acts   = x.new_zeros((B, 0, 0))

        # 3) Forecast concept layer(s). Only active in future_aware mode.
        if not self._forecast_side_active:
            # In lookback_only, no forecast SW concept computation happens.
            fc_ctx_list = []
            fc_logits   = x.new_zeros((B, 0, 0))
            fc_acts     = x.new_zeros((B, 0, 0))
        elif self.single_forecast_bottleneck:
            # Ablation mode: ONE forecast predictor on the FORECAST encoder's
            # whole-window summary (z_global_forecast). The same concept
            # context is reused for every sub-window's decoding.
            lg = self.forecast_concept_predictors[0](z_global_forecast)  # (B, K_fc)
            ac = torch.sigmoid(lg)
            ep = self.forecast_emb_plus(z_global_forecast).reshape(B,
                        self.n_forecast_concepts, self.concept_emb_dim)
            em = self.forecast_emb_minus(z_global_forecast).reshape(B,
                        self.n_forecast_concepts, self.concept_emb_dim)
            # RandInt on the single (global-collapsed) forecast bottleneck.
            # In --single_forecast_bottleneck mode concept_labels.build_window_
            # label_tables produces fc labels on the WHOLE forecast window with
            # shape (B, K_fc) -- already the right target shape, so use them
            # as-is. Guard against a stray (B, N_sw, K_fc) tensor (e.g. if a
            # future caller passes per-sub-window labels here) by mean-collapsing
            # the sub-window axis only when it is present.
            fc_labels_full = concept_labels.get("fc")
            if fc_labels_full is None:
                fc_labels_collapsed = None
            elif fc_labels_full.dim() == 3:
                fc_labels_collapsed = fc_labels_full.mean(dim=1)             # (B, K_fc)
            else:
                fc_labels_collapsed = fc_labels_full                        # (B, K_fc)
            ac_used = self._gate_activation(ac, fc_labels_collapsed, intervention)
            a_g = ac_used.unsqueeze(-1)                               # (B, K_fc, 1)
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
            fc_labels_all  = concept_labels.get("fc")  # (B, N_sw, K_fc) or None
            for k in range(self.n_subwindows):
                lg = self.forecast_concept_predictors[k](z_per_sw[k])    # (B, K_fc)
                ac = torch.sigmoid(lg)                                    # (B, K_fc)
                ep = self.forecast_emb_plus(z_per_sw[k]).reshape(B,
                            self.n_forecast_concepts, self.concept_emb_dim)
                em = self.forecast_emb_minus(z_per_sw[k]).reshape(B,
                            self.n_forecast_concepts, self.concept_emb_dim)
                # RandInt on per-forecast-SW activations using the k-th
                # slice of the (B, N_sw, K_fc) label tensor.
                labels_k = (
                    fc_labels_all[:, k, :] if fc_labels_all is not None else None
                )
                ac_used = self._gate_activation(ac, labels_k, intervention)
                a_k = ac_used.unsqueeze(-1)                               # (B, K_fc, 1)
                ctx_3d_k = a_k * ep + (1.0 - a_k) * em                    # (B, K_fc, d)
                ctx_k = ctx_3d_k.reshape(B, -1)                           # (B, K_fc * d)
                fc_logits_list.append(lg)
                fc_acts_list.append(ac)
                fc_ctx_list.append(ctx_k)
            fc_logits = torch.stack(fc_logits_list, dim=1)         # (B, N, K_fc)
            fc_acts   = torch.stack(fc_acts_list,   dim=1)         # (B, N, K_fc)

        # 3b) Sequence-aware refinement: BiGRU over per-sub-window contexts.
        # In future_aware mode: BiGRU walks FORECAST sub-window contexts
        #     (fc_ctx_list). Per-position output is fed into that sub-window's
        #     decoder input.
        # In lookback_only mode: BiGRU walks LOOKBACK sub-window contexts
        #     (lb_sw_ctx_list). Final-position output is a SINGLE VECTOR
        #     summarising the whole lookback sequence, reused as-is for
        #     every FORECAST sub-window's decoder input.
        sw_enhanced_forecast_side: torch.Tensor | None = None   # per-forecast-sw
        sw_enhanced_lookback_shared: torch.Tensor | None = None  # single vector

        if (self.sw_gru_hidden > 0
                and not self.single_forecast_bottleneck
                and hasattr(self, "sw_context_gru")):
            if self._forecast_side_active and fc_ctx_list:
                fc_ctx_stack = torch.stack(fc_ctx_list, dim=1)           # (B, N, K_fc*d)
                sw_summaries = self.sw_context_summary(fc_ctx_stack)      # (B, N, sw_summary_dim)
                sw_enhanced, _ = self.sw_context_gru(sw_summaries)        # (B, N, 2*sw_gru_hidden)
                sw_enhanced_forecast_side = sw_enhanced
            elif (not self._forecast_side_active) and lb_sw_ctx_list:
                lb_sw_ctx_stack = torch.stack(lb_sw_ctx_list, dim=1)      # (B, N_lb_sw, K_lb_sw*d)
                lb_summaries = self.sw_context_summary(lb_sw_ctx_stack)   # (B, N_lb_sw, sw_summary_dim)
                lb_enhanced, _ = self.sw_context_gru(lb_summaries)        # (B, N_lb_sw, 2*sw_gru_hidden)
                # Pool across the LOOKBACK sub-window axis to get a single
                # shared vector reused for every forecast-side decoding.
                # Mean-pool works fine here because we already stacked the
                # BiGRU's per-position outputs; mean over N_lb_sw retains the
                # sequence summary.
                sw_enhanced_lookback_shared = lb_enhanced.mean(dim=1)      # (B, 2*sw_gru_hidden)

        # 3c) Optional hybrid residual channel (Option 1). Small learned bypass
        # carrying the continuous level/scale the concepts can't encode.
        #   res_lb      : (B, r) from z_global, shared across sub-windows.
        #   res_fc_list : list of (B, r), one per forecast sub-window (from
        #                 z_per_sw[k]); None in lookback_only mode.
        # disable_residual zeroes them (concept-only forecast) while keeping the
        # decoder-input width fixed. The (non-zeroed) magnitudes are returned as
        # "residual" for the trainer's L2 penalty.
        res_lb, res_fc_list = None, None
        residual_for_penalty: list[torch.Tensor] = []
        if self.residual_dim > 0:
            res_lb = self.residual_lb_proj(z_global)                     # (B, r)
            residual_for_penalty.append(res_lb)
            if self._forecast_side_active and self.residual_fc_proj is not None:
                res_fc_list = [self.residual_fc_proj(z_per_sw[k])
                               for k in range(self.n_subwindows)]         # N x (B, r)
                residual_for_penalty.extend(res_fc_list)
            if disable_residual:
                res_lb = torch.zeros_like(res_lb)
                if res_fc_list is not None:
                    res_fc_list = [torch.zeros_like(r) for r in res_fc_list]

        # 4) Shared sub-decoder, parameterized by sub-window id.
        sub_forecasts = []
        for k in range(self.n_subwindows):
            sw_idx = torch.full((B,), k, dtype=torch.long, device=device)
            sw_emb = self.sub_window_id_embed(sw_idx)          # (B, sw_id_dim)

            if self._forecast_side_active:
                parts = [lb_ctx, fc_ctx_list[k]]
                if sw_enhanced_forecast_side is not None:
                    parts.append(sw_enhanced_forecast_side[:, k, :])
                if self.n_global_forecast_concepts > 0:
                    parts.append(gfc_ctx)
                parts.append(sw_emb)
                if self.residual_dim > 0:
                    parts.append(res_lb)             # shared lookback-level residual
                    parts.append(res_fc_list[k])     # per-sub-window forecast residual
            else:
                # lookback_only: dec_in = lb_ctx + (shared LB-BiGRU summary) + sw_emb
                parts = [lb_ctx]
                if sw_enhanced_lookback_shared is not None:
                    parts.append(sw_enhanced_lookback_shared)
                parts.append(sw_emb)
                if self.residual_dim > 0:
                    parts.append(res_lb)             # lookback-level residual
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
            # LOOKBACK-SUB-WINDOW concept outputs (lookback_only mode only).
            # Shapes: (B, N_lb_sw, K_lb_sw). Empty tensor in future_aware mode.
            "lb_sw_logits": lb_sw_logits,
            "lb_sw_acts":   lb_sw_acts,
            # Hybrid residual channel magnitudes (Option 1) for L2 penalty.
            # (B, r * n_residual_projs); empty tensor when residual_dim == 0.
            "residual": (torch.cat([r.reshape(B, -1) for r in residual_for_penalty], dim=-1)
                         if residual_for_penalty else x.new_zeros((B, 0))),
        }
