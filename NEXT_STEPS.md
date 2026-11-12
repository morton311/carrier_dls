# Outstanding work and next steps

Last updated 2026-11-05. Paper draft: `papers/scitech_2027/main.tex` (gitignored; red `\todo{}` markers in the PDF).

## Model and method

1. **Full-coverage DLS is unstable in long rollouts at Re 30k (decision pending).**
   All four full-coverage runs (`configs/ldc/*_30k_10k_fc*.json`, latent `ldc_30k_10k_dls_p99m10_fc`) are worse than
   the original mesh. Projection-in-the-loop seeds 1/2/3: skill 0.22/0.31/0.28 (original 0.35/0.34/0.30), long-run
   RMS 75/47/277 % (original 33/27/28 %). Seed 3 blows up, and the energy growth starts in the newly covered
   strip (strip energy x1271 by step 2000). The compression floor is fine (6.43 % vs 6.25 %), so this is a forecasting
   problem in the cut elements. Suspected cause: the loss ridge for cut elements is scaled by the masked trace
   (`lib/decode.py::element_gram_roots`), while the assembly ridge uses the full-element trace, so the weakly observed
   cut-element directions are barely penalized in training. Options:
   - scale the loss ridge by the full-element trace, as the assembly does (one-line change), then retrain; or
   - remove the DOFs of nodes that lie entirely outside the domain.
   Until resolved, the paper keeps the original-mesh Re 30k numbers.
   **Cut-element regularization (2026-11-02, finite cell method style).** `latent_params.cut_weight` (alpha) gives
   the points of cut elements outside the domain weight alpha, fit to an extension of the field
   (`cut_extension`: 'edge' or 'odd' reflection); the loss Gram ridge of a regularized latent uses the full-element
   trace; scoring stays strictly in-domain (`element_masks(strict=True)`). `latent_params.basis_from` reuses another
   latent's local modes (the basis is random-patch SVD, so this isolates the mesh change). Fixed on the way: the
   compression right-hand side ignored the element masks (harmless with zero-padded data, wrong with an extension).
   Diagnosis: at alpha 0 the out-of-domain DOFs hold 35 % (Re 30k), 71 % (Re 15k) and 49 % (cavity) of the
   coefficient energy. alpha 0.01 (odd): cavity 0.49 -> 0.17 and mass-matrix condition 1.6e9 -> 4.5e7 with in-domain
   error unchanged (0.1819 -> 0.1821); Re 30k condition is set by the basis itself (1.7e9 even uncut), and the
   cut-strip compression error rises 0.141 -> 0.185 (the alpha-0 fit uses the free outside DOFs). Runs (seed 1):
   `ldc/spatial_gram_rollout_galerkin_30k_10k_fca{0.01,0.1}`, `ldc/spatial_gram_rollout_galerkin_heldout_s1_fc{,a0.01}`,
   `challenge/spatial_gram_rollout_galerkin_fc_all_d128_ca0.01` (result dirs use `fca0_01odd`).
   Results (seed 1): Re 15k full coverage beats the truncated mesh on 30-step error with the lid strip included
   (alpha 0: 2.72 %, alpha 0.01: 2.75 %, truncated 3.81 +- 0.2 % over 3 seeds); alpha 0.01 fixes the long-run
   statistics that alpha 0 loses in the strip (RMS error 9.0 -> 6.0 %, TKE 2.7 -> 1.7 %, final field error
   0.28 -> 0.16; truncated 5.6 %, 2.1 %, 0.25). Cavity: Challenge NMSE 0.786 vs 0.771 (better through step 10,
   worse at 20-30), multi-start error the same (0.864 vs 0.869) with a smaller start spread (0.061 vs 0.081),
   long-run energy retention worse (0.17 vs 0.40). Compression floor identical (0.036).
   Re 30k seed 1, 30-step error / skill / long-run RMS error: truncated 0.246 / 0.353 / 32.6 %, fc alpha 0
   0.297 / 0.218 / 74.7 %, alpha 0.01 0.253 / 0.333 / 46.4 % (damped: energy retention 0.42), alpha 0.1
   0.248 / 0.346 / 29.7 % (best worst-start 0.318 and start spread 0.034 of every Re 30k run; TKE error 39.8 %).
   alpha 0.1 recovers the truncated mesh's accuracy with the whole domain covered.
   **alpha 0.1 (odd) is the choice.** Re 30k over 3 seeds: 30-step error 0.256 +- 0.009 (truncated 0.256 +- 0.011,
   fc alpha 0 0.278 +- 0.018 with seed 3 blowing up), worst start 0.328 (truncated 0.347), all 3 stable (seed 2
   damped: RMS error 55 %). Re 15k seed 1: 2.76 % with the best long-run statistics of any Re 15k run (TKE error
   1.3 %, final field error 0.137, late spectrum 0.11). Cavity seed 1: Challenge NMSE 0.742, the best of any model
   (alpha 0: 0.771, hybrid ms8: 0.772), better at every step; multi-start 0.850 (0.869); long-run energy
   retention 0.23 (0.40). Seeds 1-3 (2026-11-03): Re 15k 2.71 +- 0.08 % (truncated 3.81 +- 0.20 %), worst start
   3.4 % (4.4 %), final field error 0.138 (0.219), long-run RMS error 6.3 % (5.6 %). Cavity Challenge NMSE
   0.742 / 0.789 / 0.763 = 0.765 +- 0.024 (alpha 0, one seed: 0.771), so seed 1 was lucky; multi-start 0.853 +-
   0.003 (0.869) and worst start 1.00 (1.19) hold up; long-run energy retention 0.21 (0.40). Next: retrain the
   paper models (hybrid diffusion ms8) on the alpha-0.1 latents; seeds 2-3 of cavity alpha 0 to settle the cavity.
   Hybrids on alpha 0.1 (2026-11-03, `*_fca0.1{,_ms8}`, `*_ca0.1{,_ms8}`, mean model = alpha-0.1 seed 1):
   Re 15k ms8 30-step 1.56 % (old-mesh ms8 2.11 %, new det 2.76 %), skill 0.952, autocorr 0.017, late spectrum
   0.044 (all best of any Re 15k model), RMS error 6.0 % (old 4.9 %). Re 30k: one-step hybrid 0.245 vs det 0.248
   with better long-run stats (spectrum 0.14 vs 0.23); ms8 adds nothing (0.248, spectrum 0.50). Cavity ms8:
   multi-start 0.875 (det 0.850), restores long-run energy 0.23 -> 0.41, spectrum 4.8 -> 0.97.
2. **Re 15k headline on a full-coverage basis** (the Re 15k mesh is truncated too: 512x512 grid, 491x491 covered), so the lid strip is included (after item 1 is fixed).
3. **Diffusion-based element-token transformer** (motivation: MSE-trained models damp toward the conditional mean;
   cavity keeps 40 % and Re 30k 65 % of the fluctuating energy). Plan:
   - diffuse the next-frame increment in whitened global coefficients (`M_lambda^{1/2}`), so samples stay
     consistent across elements and no noise enters near-null directions;
   - denoiser = spatial transformer + diffusion-time embedding (adaLN, DiT-style); predict per-element x0/v and apply
     the mass-weighted projection after every denoising step;
   - one-step training (v-parameterization, cosine schedule, Gram-weighted loss, noise on the conditioning frames);
     10-30 DDIM/EDM sampling steps;
   - evaluation: ensembles (~8 samples) in multistart, ensemble-mean NMSE + spread/skill, long-run statistics per sample;
   - code: `lib/diffusion.py`, `DiffusionSpatialTransformer` in `lib/models.py`, new `model_type` branch in
     `lib/runner.py`, ensemble support in pred/eval/`lib/multistart.py`, a `diffusion` config block.
   Start on Re 15k, then the Challenge 2.1 cavity.
   **Implemented 2026-10-19** (`model_type: tr_spatial_diff`, `model_params.diffusion`; see the `lib/diffusion.py`
   docstring). Choices that differ from the plan above: the default diffusion space is per-DOF standardized global
   increments (`space: dof`), not `M_lambda^{1/2}`-whitened ones (`space: gram`, kept as an option). In gram space the
   y-variance per DOF spans 0.03-119 and white y-noise reaches 4e4x the increment variance in near-null coefficient
   directions on the Re 15k latent. The loss is still the field-energy (A-metric) x0 error, weighted min(SNR+1, 5).
   EMA weights (0.999) are what eval runs. `scripts/challenge_eval.py` scores the ensemble mean and one sample.
   Runs: `configs/ldc/spatial_diff_heldout_s1.json`, `configs/challenge/spatial_diff_fc_all_d128.json`
   (+ `_long`: resumed from epoch 200 with `es_min_delta` 0.002 and the new `lr_threshold` 0.005; it stopped at
   313 and is within noise of the epoch-204 model, so the cavity loss has plateaued).
   **First results (2026-10-19), 8-member ensembles, one-step training only:**
   - Re 15k: 30-step error 1.97 % vs 3.60-4.01 % deterministic (3 seeds), skill 0.94 vs 0.88, one sample 1.99 %.
     The long rollout is unstable: one corner grows from about step 150 and doubles every ~25 steps
     (RMS error 1204 %, energy x150 by step 500). Spread-skill 0.22 (far too little spread).
   - Cavity: long-run RMS error 29 % vs 48 %, energy retention 0.59 vs 0.40, spectrum log-ratio 0.53 vs 3.7,
     autocorrelation error 0.13 vs 0.61. Short horizon is worse: Challenge NMSE 0.914 vs 0.771 (DMD 1.09); better
     only at step 1 (0.29 vs 0.35), crosses 1 at step 17. Spread-skill 0.72.
   **Robustness fine-tunes (Re 15k, from the s1 model; `_cni`, `_pf4`, `_cn`):** none fixes the blow-up.
   Input-only window noise (cni, 0.01) 1.88 % / RMS 1031 %; pushforward k<=4 (pf4) 1.92 %, spread-skill 0.40 /
   RMS 1091 %. Window noise with the true frame as target (cn, 0.02) and pushforward k<=8 diverged in training:
   an error the model cannot infer is learned as extra randomness. Four independent 500-step samples of the base
   model all cross 3x the true energy at steps 290-293: an unstable mode of the learned mean map, not sampling noise.
   **Residual mode (`diffusion.mean_model`)**: frozen rollout-trained deterministic forecast + diffused residual
   (CorrDiff-style). Runs: `configs/ldc/spatial_diff_res_heldout_s1.json`,
   `configs/challenge/spatial_diff_res_fc_all_d128.json`.
   - Re 15k hybrid: **stable** (energy 0.8-1.25 of truth over 500 steps), RMS 5.12 % (det 5.60 %), spectrum
     log-ratio 0.035/0.133 (det 0.060/0.199), autocorrelation 0.041 (det 0.075); 30-step error 2.61 % (det 3.81 %,
     pure diffusion 1.97 %). The residual model overfits (best epoch 40; train 0.09 vs val 0.56): the mean model's
     one-step error is 4.4 % of the increment in-sample vs 5.8-6.1 % held out. Dropout 0.3 / d128: no gain.
     Out-of-fold residuals (2 folds, `spatial_gram_rollout_galerkin_heldout_s1_f{0,1}of2` via
     `train_data[].train_fold`; hybrid `_oof`) made it **worse**: 30-step 5.05 %, RMS 9.5 %. Each fold model has only
     900 snapshots, so its out-of-fold residual energy (0.21) is ~6x the full model's held-out residuals; the
     residual model then over-corrects. More folds (or a cheaper inflation of the in-sample residuals) untested.
   - Cavity d256 hybrid (5.3 M): same as d128 within noise; capacity is not the limit.
   - Re 30k (10k snapshots, original mesh): pure diffusion 30-step 0.210 vs det 0.246-0.268 (3 seeds), skill 0.45 vs
     0.30-0.35, but unstable (energy x2.7 by step 100). Hybrid stopped at epoch 3 and equals det (0.2457): the det
     model overfits (held-out residuals ~25x the in-sample ones), so in-sample residuals carry no usable signal.
   - **Multi-step energy-score training (`diffusion.ms_steps`, 2026-10-20)**: K sampled steps with gradients through a
     4-step DDIM sampler, 2 members, fair energy score in the field-energy norm + the one-step loss (see the
     `lib/diffusion.py` docstring). 4 DDIM steps already match 20 on the one-step-trained model (Re 15k 1.86 % vs
     1.97 %), so training and inference use 4. Re 15k fine-tunes from the pure model: `_ms8`, `_ms16`. At epoch 40,
     `_ms8` (raw weights): 30-step 1.70 %, long-rollout energy x18 at step 500 (base x110), doubling every ~64
     steps instead of ~25: slower growth, not yet stable. Final `_ms8` (epoch 100): **30-step 1.58 %** (det 3.81 %),
     step 1 0.11 %, skill 0.951; long rollout still diverges, from every start point tried after 330-420 steps.
     `_ms16` paused at epoch ~70 (tracking `_ms8`; checkpoint kept).
   - **Thermalizer** (`diffusion.target: state` model `spatial_diff_state_heldout_s1` + `diffusion.thermalizer`,
     Pedersen et al. 2025): applied every step it stabilizes 1,000-step rollouts but wrecks accuracy (30-step 15-31 %);
     triggered by a detector calibrated on real snapshots (max score, every 10 steps, t=0.3): 30-step 1.84 %, stable,
     but long-run statistics mediocre (RMS 14.8 %, TKE 15 %, autocorrelation 0.24 vs hybrid 5.1 %, 2.3 %, 0.04).
     Run `spatial_diff_heldout_s1_ms8_th`.
   - **Multi-step hybrids (best overall so far)**: the unroll feeds the mean model the hybrid's own samples, so the
     residuals it trains on are no longer in-sample.
     - Re 15k `spatial_diff_res_heldout_s1_ms8` (epoch 18 by val ES; epoch-40 raw weights scored 1.97 %): 30-step
       2.11 % (det 3.81 %), stable, best long-run statistics of all models: RMS 4.87 %, spectrum log-ratio
       0.017/0.057 (det 0.060/0.199), autocorrelation 0.024 (det 0.075).
     - Cavity `challenge/spatial_diff_res_fc_all_d128_ms8` (32 members): Challenge NMSE **0.772 vs det 0.771**, better
       through step 10 (0.273/0.566/0.741 vs 0.350/0.593/0.793), worse at 20-30; multi-start NMSE 0.791 vs 0.793 with
       the lowest start-to-start spread (0.096 vs 0.156); RMS 33.5 % (det 48 %) but energy retention fell to 0.46.
     - Model selection by the fixed-draw validation energy score stops early and not at the best multi-start epoch;
       consider selecting on multi-start error over the test windows.
     - Running: Re 30k `spatial_diff_res_30k_10k_ms8`, `spatial_diff_30k_10k_ms8`.
   - Multi-start Challenge NMSE (same 50 starts, 32 members; `scripts/fig_multistart_nmse.py`,
     `figures/multistart_nmse.pdf`): det 0.793, hybrid 0.816, pure 0.872, DMD 1.107, persistence 1.956.
   - Cavity hybrid (1.4 M): Challenge NMSE 0.843 (8 members) / **0.805 (32 members)** vs det 0.771 and pure diffusion
     0.914 / 0.878; better than det to step ~10 (step 10: 0.759 vs 0.793), worse after (step 30: 0.991 vs 0.867),
     never above 1. Long-run statistics as pure diffusion (RMS 29.5 %, energy 0.55). Testing d256
     (`configs/challenge/spatial_diff_res_fc_all_d256.json`).
   - Full-coverage latents carry most of their coefficient norm in out-of-domain DOFs (cavity truth ~4e4 vs ~3.7e3
     for deterministic predictions at the same field RMS): the 1e-8 compression ridge leaves them nearly free.
     Possibly relevant to item 1 (inputs dominated by unconstrained, invisible coefficients).
   Earlier plan: (a) training-time robustness for the Re 15k blow-up (`cond_noise_std`, or pushforward training on the
   model's own one-step samples); (b) multi-step-aware training for the cavity's short-horizon gap, or a hybrid with
   the BPTT-trained deterministic model as the mean and diffusion for the residual (CorrDiff-style); (c) more
   ensemble members for the Challenge score (late-step NMSE of an M-member mean saturates near 1 + 1/M).
4. **Challenge 2.1**: three seeds of the 1.9 M model (`configs/challenge/spatial_gram_rollout_galerkin_fc_all_d128.json`);
   official-format submission for the 32 test sequences (fill the field from the full-coverage latent).
5. **Longer Re 30k record** (e.g. 30,000 snapshots): data quantity is the only lever that has helped so far.

## Paper (`papers/scitech_2027/main.tex`)

- POD-latent baseline (Re 30k started 2026-11-03: `ldc/pod_30k_10k_m400`, global temporal transformer of
  Table arch, basis fit to the training block only via `latent_params.fit_train_only`). Held-out projection
  error of a training-block POD floors at 0.150 by ~400 modes per component (200: 0.173, 100: 0.221); the
  DLS fca0.1 latent (1,584 per component) reaches 0.134. Result: 30-step ms_err 0.431 (worst start 0.62,
  step 1 0.37, skill -0.12) vs DLS 0.256 +/- 0.009; RMS 20 %, energy retention 0.90. Overfits from epoch 2
  (train 0.23 -> 0.0055, held-out 0.42 -> 0.88). Regularized retry `pod_30k_10k_m400_reg` (d64, AdamW 0.1,
  rollout noise 0.05) is worse: 0.490, skill -0.28. Fewer modes (`pod_30k_10k_m100`): in-span 0.407, skill -0.08,
  same overfit (best epoch 3). `pod_30k_10k_m50` (done 2026-11-05): in-span 0.352, skill +0.04, best epoch 3
  (train 0.107 -> 0.0004, held-out 0.25 -> 0.41), projection floor 0.302, RMS 28.6 % (its RMS floor is only 8.4 %,
  so this is not truncation), energy 0.83. The +0.04 is not a real win: in-span scoring gets easier as modes are
  removed, because persistence is scored in the same truncated space too (0.384, 0.376, 0.366). Raw-field score
  (`scripts/raw_field_score.py`, includes truncation; `results/raw_field_score.json`): POD 400 0.453, POD 100 0.455,
  POD 50 0.452, DLS 0.294 +/- 0.007, hybrid 0.285, raw persistence 0.384. So fewer modes gain nothing against the
  real field, and every POD model stays worse than persistence (raw skill -0.18; DLS +0.23). Not yet in the
  paper's tables.
- Frequency-extrapolation baseline under the current protocol, done 2026-11-05 (`scripts/extrap_baseline.py`,
  `results/extrap_baseline.json`; all frequencies kept = training record repeated, on the alpha-0.1 DLS coefficients).
  30-step error 1.91 (Re 15k) and 1.68 (Re 30k), far worse than persistence (0.325, 0.380) because it ignores the
  current state. RMS error 9.8 % (Re 15k, det 6.3 %) and 23.2 % (Re 30k, det 29-55 %, POD 20.0 %), so at Re 30k it
  still beats the transformers on long-run RMS. Floors over the same windows: DLS 5.8 % / 6.9 %, POD 400 3.8 %.
  Not yet in the paper. Still open: same architecture as the spatial model; wall-clock cost per forecast step,
  local vs global.
- Introduction now has related work (FEONet, DG-FEONet, MINN, FE hypergraph and graph ROM, GMsFEM + DL, DD-PDCNN) and a
  metric-precedent paragraph (2026-11-05). Results sections, abstract and Table latent still hold the pre-alpha-0.1
  numbers; the 2026 progress report (`papers/reports/Accomplishments_2026.docx`) already uses the alpha-0.1 ones. Cavity
  seeds are 0.765 +/- 0.023 from the unrounded values (0.024 above came from rounded ones).
- Long-horizon statistics figure (RMS fields, TKE history, probe PSDs) for the Re 15k model.
- Re-run the architecture, training-recipe and compression studies on the held-out split (split v2), with seeds,
  or keep the footnote.
- lambda sweep of the mass-weighted projection on the held-out split.
- p99m5: three seeds on split v2; optionally a p99m20 point.
- Decide how to compare with the 2026 results (re-evaluate the 2026 model with the current protocol?).
- Add `figures/full_cover.pdf` to Sec. II-D (LDC Re 30k: full-grid reconstruction error 23.4 % -> 12.3 %).
- Cite the original LDC DNS; decide on the POD-on-DLS paragraph; drop the airwake acknowledgment; note the scope
  change from the abstract (no airwake, no temporal RoPE).
- Terminology: "mass-weighted projection", never "Galerkin" for the assembly (see memory note).

## Code and housekeeping

- **Uncommitted**: early-stopping hold (`model_params.es_start_epoch`), metrics JSON fields, input-history /
  model-size / full-coverage configs, figure scripts (`fig_assembly_schematic.py`, `fig_assembly_example.py`,
  `fig_full_cover.py`), relabelled figures, on-demand decoding (`lib/recfield.py`, `dls_2d.decode_fields`,
  runner/plotting/script changes), this file.
- Disk: `/home/HDD` filled up on 2026-10-14 (36 in-sample `*_train_data_pred_rec.h5` deleted, 109 GB).
  **Done 2026-10-19:** decoded fields are no longer stored by default. Config key `save_rec` (top level, default
  false): pred writes only `*_pred.h5` coefficients; eval decodes on demand through `lib/recfield.py`
  (`register` / `open_rec`, cached TKE/RMS in a `*_pred_stats.h5` sidecar); the codec reference is decoded from
  `latent_coeff.h5`. `dls_2d.decode_fields` is a vectorized, bit-identical replacement for the element loop (~1.8x).
  Verified: re-evaluating two runs reproduced all 47 metrics (differences <= 1e-6, GPU nondeterminism only);
  the Re 30k 2,000-step eval took 3.2 min with no large writes.
  **Deleted 2026-10-19** (user OK): the 64 stored `*_pred_rec.h5` (71.6 GB) and 11 shared
  `results/<latent>/*_codec_ref_rec.h5` (66.7 GB); lists in the session scratchpad. `/home/HDD` now has 219 GB free
  (97 % used); `results/` is 23 GB.
- `main.py -m latent` ends with `ImportError: latent_eval` from `lib/dls.py` after the latent is written (pre-existing).
- `pl.plot_RMS` crashes on 3D data (`IndexError: index 20 is out of bounds for axis 2 with size 3`, z-slice), which aborts
  eval before the metrics JSON is written. Found re-scoring the airwake run (`configs/carrier/distributed_refix.json`,
  `results/0aoa_dls_p21m25/distributed_refix`, a copy of the May weights with the fixed inference code): RMS error 64.3 %
  (May: 60.7 %), TKE error 83.8 %. Fix before returning to airwake data.
- `train_data.sample_train` samples windows with replacement (about 13 % duplicates at 3,000 of 10,348).

## Annual report metrics (2026-11-06)

Advisor feedback on `papers/reports/Accomplishments_2026.docx`: the program officer does not know the challenge, so the
report now uses one measure for all three flows (50 forecasts x 30 steps, relative field error, skill against
persistence) and Table 2 defines each column. NMSE no longer appears in the report. Open cavity on that protocol
(`scripts/cavity_baselines.py`, `results/cavity_baselines.json`): DLS 85.2 +/- 0.3 %, skill 0.38; hybrid 87.5 %, one sample
100.4 %; DMD-25 103.3 %, skill 0.25; mean flow 100 %, skill 0.27; persistence 137.4 %. Same ranking as the official NMSE
(0.765, 0.788, 1.09, 1.0, 1.89). The paper still reports NMSE for the challenge; decide whether it should change too.
`scripts/raw_field_score.py` now also covers Re 15k and the cavity: compression error 8.5 % and 18.8 %; DLS against the raw
field 8.9 % and 85.9 %.

## SLICK comparison on Challenge 2.1 (2026-11-06)

Chu & Schmidt, Proc. R. Soc. A 481:20250270 (SLICK, a stochastic Koopman ROM on SPOD coordinates). The paper itself uses its own
protocol; the comparable benchmark is in their repo (`challenge21/`, commit d61ba83, added 2026-09-21). Schmidt also leads the challenge, so
reviewers will know it. `scripts/slick_baseline.py` is my Python port (running their MATLAB was blocked by the permission system); it reproduces
their saved results to 2e-14 per episode on their split. On our 30 held-out sequences, same 10,368 training snapshots, hyperparameters
selected on our validation block: SLICK 0.863 (0.608 step 1, 0.925 step 30), DLS 0.765 +/- 0.023 (lower at all 30 steps, 27/30 sequences,
difference -0.098, exploratory 95 % block-bootstrap interval -0.132 to -0.061), hybrid 0.791. SLICK's own 20 episodes: DLS 0.742 vs 0.843, or
0.737 vs 0.850 on the 13 inside our held-out block. DLS keeps more energy than SLICK at every reported step, so the gap is not from damping.
`scripts/slick_comparison.py report` prints all of this (`results/slick_comparison.json`). Only SLICK's mean forecast is scored: its long-run
statistics, which it is designed to preserve, are untested here and may well beat ours.
To do if approved: `main.tex:412` and `:462` say our model is the only one below the mean flow throughout, which SLICK also does (max 0.946);
add a SLICK row and curve, cite it, reword. Report: needs SLICK on the 50-start protocol. Their repo has no license file, so check with the authors
before publishing the port. Also: `data/Challenge2_1_train.h5`'s mean uses all 12,800 frames including our held-out block
(rms 0.22 from the training-only mean, field rms about 93); negligible, but a fit-only mean would be cleaner.

## NMSE against turbulent fluctuations (2026-11-09; for the SciTech paper and the journal article)

The advisor's study is separate and will not be shared (user, 2026-11-10): do not use or cite it. Our own sweep of enrichment, tokenization and losses is
planned in docs/tradeoff_study.md (staged, about 145 GPU-h in all; a SciTech subset of about 17 GPU-h). SLICK stays out of the paper until the user
understands its implementation and assumptions.
Framework: NMSE = 1 + E - 2 c sqrt(E), exact per frame (`scripts/fig_tradeoff.py`, residual 2e-14; law of cosines, Taylor 2001). It is the energy-only case of the
distortion-perception function, D(P) = D* + ((P* - P)+)^2 (Freirich et al. 2021, Thm 1), with D(0) <= 2 D* (Blau & Michaeli 2018, Thm 3). So the existence of the tradeoff is known; the contribution
has to be how the enrichment and tokenization move it. Cavity, 30 held-out sequences: DLS 0.765 at E 0.30, c 0.47 (3 seeds). A per-lead gain set on 7 validation-block
episodes takes it to NMSE 1.00 at E 0.89 with no training, the same move as the advisor's; SLICK needs 1.24 for that. That is the baseline a statistical loss must beat.
What a gain cannot do (`scripts/tradeoff_scales.py`, seed 1, energy kept for wavelengths >31 / 8-31 / <8 points, truth shares 84/13/3 %): DLS as is 42/4/2 %; energy-matched 120/9/5 %, RMS-map error 45 -> 21 %;
hybrid one sample 59/35/16 % at NMSE 1.08. Basis ceiling (projection of the truth): 5 modes 100/95/33 %, 10 modes 100/98/50 %, 20 modes 100/99/74 %, so the enrichment caps small-scale recovery.
Caveats: gains act open loop on the 30-step window only, not on free rollouts. Element-local and global-temporal runs have stale energies (pre-inference-fix), so tokenization is untested here; re-evaluate them.
Same model, d128 tau 4: long-run energy 0.40 at alpha 0 (one seed) vs 0.20-0.24 at alpha 0.1 (three seeds): run alpha 0 seeds 2-3. Generative rows mix products (NMSE from the ensemble mean, statistics from one
sample); on a Pareto plot use one sample for both (hybrid: 1.06, not 0.79).

## SciTech subset running (launched 2026-11-10; paper draft due in 1-2 weeks, final manuscript about 2026-12-01)

Four queues via `scripts/run_queue.sh` (train, pred, eval; prints EXIT per config): 10-mode alpha-0.1 model x 3 seeds
(latent results/cavity_dls_p13m10_fca0_1odd, 5,940 coefficients, compression NMSE 0.022); energy term beta = 0.1, 0.3, 1, 3
(`frame_energy` in lib/losses.py, checked against the identity); alpha 0 seeds 2-3. Four at once fill the GPU, so queues will
pass the 2 h background limit: resume from checkpoint. After they finish: challenge_eval.py on the new runs, then
fig_tradeoff.py, tradeoff_scales.py and flow_stats.py (add the new runs to their run lists).
Proposed statistical measures (energy ratio, RMS map, spatial and frequency spectrum shape, PDF shape, POD energy distance,
each over its sampling floor): docs/tradeoff_study.md; awaiting the user's agreement.

## Loss ridge dominates the Gram loss (2026-11-10)

- With gram_lambda 1e-2 the ridge holds 75-88 % of the truth's metric norm on the Challenge cavity and 99.6-99.8 % on
  the LDC p99m10 latents; on trained models' errors the loss gradient is 39-51 deg (cavity) and 73-84 deg (LDC) from the
  field-energy gradient. Paper Sec. on the metric said "within about 11 deg": fixed 2026-11-11 (see below).
  Details and numbers: docs/tradeoff_study.md; check: scripts/gram_ridge_check.py.
- Running: cavity gram_lambda 1e-3, 1e-4, 0 (configs/challenge/*_ca0.1_gl*.json); Re 15k gram_lambda 1e-6
  (configs/ldc/spatial_gram_rollout_galerkin_heldout_s1_fca0.1_gl1e-6.json). Assembly lambda unchanged (1e-2).
- The energy-term (frame_energy) beta sweep is invalid under the 1e-2 ridge (it inflated the field 1.5-2.5x). Rerun it
  at the chosen small gram_lambda.
- Results so far: alpha 0 vs 0.1 indistinguishable over 3 seeds. 10 modes, 3 seeds: 30-step NMSE 0.766 / 0.803 / 0.787
  (mean 0.785) against 5 modes 0.742 / 0.789 / 0.763 (mean 0.765); 1-step correlation 0.785 against 0.81. The basis
  ceiling improves (projection shape_x 3.6 dB against 5.4 dB) but the deterministic rollouts do not use it (shape_x
  15-17 dB, pdf_shape 0.25 either way). Enrichment beyond 5 modes buys nothing for a deterministic model on this data.
- Re 15k, gram_lambda 1e-6 (one seed): worse than 1e-2 (3 seeds). 30-step error 0.033 vs 0.026-0.028; step 30 0.065 vs
  0.035-0.037; the long rollout blows up (energy 21x by the end, RMS error 171 % vs 6.2 %). The ridge does real work:
  the near-null directions feed back through the inputs, so the loss must hold them. Keep the base lambda at 1e-2.
- frame_energy now takes metric 'field' (default): scored in pure field energy (lambda 0, the loss metric's scale) while
  the other terms keep lambda 1e-2 (losses.apply_metric). Verified: matches grid energy to 0.5 %, combo = mse_lambda +
  beta * fen_field exactly, configs without the term unchanged. Runs before 2026-11-11 used the old behavior ('loss').
- Running: beta sweep with the field term (configs/challenge/*_fen{0.1,0.3,1,3}_field.json); Re 15k gram_lambda 1e-4.

### Paper edits (2026-11-11)
- Sec. 2.2 (metric): the loss is now described as field-energy error plus a ridge that carries most of the loss (99.8 % of the
  Re 15k truth norm, 77 % on Challenge) and is needed for stable rollouts (lambda 1e-6 result). The MSE-on-coefficients
  angle was re-measured on the paper's models with `scripts/gram_ridge_check.py`: 83-86 deg, against 75-81 deg (Re 15k)
  and 39-51 deg (Challenge) for the Gram loss. On Re 15k the Gram loss is only slightly better aligned than a plain MSE.
- Sec. 4.3: the five floor-normalized statistics (energy ratio, RMS map, spatial and frequency spectrum shape, PDF shape).
- New Sec. 6.8 "Forecast error and fluctuation recovery" (label sec:tradeoff) with Table 8: the Murphy / Taylor identity,
  deterministic models near the NMSE optimum (1 - c^2 is 0.718 of the 0.765 mean NMSE), the gain baseline, alpha and
  enrichment over 3 seeds, the energy-term method. Todos: field-energy beta sweep rows and a figure; tokenization sweep;
  whether the hybrid enters.
- Flagged (todo, not changed): Table 6 / abstract quote seed 1 of the alpha-0 1.9 M model (energy 0.40, RMS 48 %); three
  seeds give 0.25 +- 0.13 and 59 +- 9 %. The width/energy claim in Sec. 6.7 and the Discussion is within that spread.
- Discussion: metric sentence, enrichment limitation (10 modes did not help) and the link to sec:tradeoff.
- Bibliography: murphy1988skill, taylor2001diagram.

### Loss ridge sweep, first results (2026-11-11)
- Re 15k, gram_lambda 1e-4, seed 1: 30-step mean error 0.0153 against 0.026-0.028 (1e-2, 3 seeds); skill 0.953 against
  0.915-0.919; long rollout stable (energy 0.93 at the end) but RMS 7.0 % vs 6.2 %, TKE 3.9 % vs 1.3-1.9 %.
  1e-6 blew up, so there is an optimum in between. Running (queues 11-12): 1e-4 s2, s3; 1e-3 and 1e-5 s1.
- Challenge cavity, single seeds: 1e-4 NMSE 0.732, 0 0.783 (1e-2: 0.742-0.789); both stable; better spatial spectrum
  shape (13.3-13.7 dB vs 16.3-17.2) and more small-scale energy. 1e-3 running.
- If the Re 15k seeds hold: rerun the paper's headline recipe at the chosen lambda (assembly lambda is separate and
  stays 1e-2 unless tested), and revise Sec. 2.2 (todo in the draft).
- Re 15k gram_lambda 1e-4, 3 seeds (done 2026-11-11 12:02): 30-step error 0.0172 +- 0.0028 vs 0.0271 +- 0.0008, skill
  0.947 vs 0.917; but RMS 8.4 +- 1.8 % vs 6.27 %, TKE 5.0 % vs 1.6 %, final energy 0.92 vs 1.02. A trade-off, not a free
  win: decide whether the paper keeps 1e-2 (long-run statistics, consistency) or reports both. 1e-3, 1e-5 running.
- Field-energy beta sweep: E matches the per-frame optimum (0.40/0.43, 0.56/0.64, 0.75/0.75); NMSE 0.78 / 0.88 / 0.97;
  moves along the gain curve (vs gain -0.015 / +0.004 / +0.036). beta 0.1 resuming; beta 3 rollout stats pending.
- Queues now run detached (setsid; scripts/run_steps.sh; logs in results/queue_logs/) so they survive a session exit.

### Resume bug (found and fixed 2026-11-11)
`_model_fit` on resume took the checkpoint epoch's current weights as the "best" model and never restored the
checkpoint's `best_model`, while keeping the true best loss. A resumed run that never beat its pre-resume best therefore
ended with the checkpoint-epoch weights. Fixed in lib/runner.py (restore best_model, best epoch from the loss history,
replay the patience reference). Of 12 resumed runs, 3 had the wrong weights:
- cavity_dls_p13m5_fc/spatial_gram_rollout_galerkin_fc_all_d128 (the paper's 1.9 M Challenge row, alpha 0 s1): epoch 20
  used, best epoch 12; checkpoint overwritten, so retraining (queue 15). Its outlier free-rollout energy (0.40 vs
  0.17-0.18 for s2, s3) and the paper's width/energy claim may be artifacts of this.
- ..._fc_all_d128_tl16 (tau 16, 7.4 M row of the Challenge table): epoch 15 used, best 4; retraining (queue 15).
- ..._ca0_1_fen0_1_field: recovered exactly from the checkpoint (epoch 12): NMSE 0.744, E 0.373, c 0.505.
Old outputs are kept in each run's resume_bug_backup/. After the retrains: rerun scripts/challenge_eval.py,
flow_stats.py, tradeoff_runs.py (drop the L0 cache entries), and update the paper's Challenge table.
- Re 15k ridge sweep complete for seed 1 (1e-2 .. 1e-6): 30-step error 0.0276 / 0.0223 / 0.0153 / 0.0219 / 0.0330;
  RMS 6.2 / 6.5 / 7.0 / 47 / 171 %. Short horizon best at 1e-4; long-run statistics worsen monotonically; unstable below
  1e-4. Options: seeds at 1e-3 (middle point), a lambda figure for the paper.
- Do not edit scripts/run_steps.sh while a queue is executing it (bash reads scripts incrementally; queue 14's log was
  garbled by such an edit, though its runs completed).
- Resume-bug retrains done (2026-11-11 13:55). Correction: the tl16 run is the "tau = 16, 1.9 M" row, not 7.4 M.
  1.9 M (alpha 0 s1, best epoch 9): 30-step mean 0.771 -> 0.777, step 30 0.87 -> 0.94; free-rollout energy 0.40 -> 0.18,
  RMS 48.3 -> 60.5 %. tau 16, 1.9 M: 0.897 -> 0.816 (no longer crosses the mean flow), energy 0.19 -> 0.31, RMS 63.9 -> 49.9 %.
  alpha 0 three seeds now agree: NMSE 0.775 +- 0.004, free-rollout energy 0.174 +- 0.005 (the 0.17-0.40 "seed spread"
  was the bug). Paper updated: abstract and conclusions 0.78, Challenge table rows, correlation/energy by lead, removed
  the width/energy claim, tau-16 sentence, Table 8 alpha-0 row; figures/challenge_heldout.pdf regenerated. Backups of
  the old draft in the session scratchpad, old run outputs in each run's resume_bug_backup/.

## Compression study, Challenge 2.1 (2026-11-11)

`scripts/compression_study.py` (cache `results/compression_study/cavity.json`), `scripts/fig_compression.py`
(`figures/compression_study.pdf`). 102 DLS bases (patch 7-41, modes 0-100, full cover, alpha 0.1 odd, up to 8,000
coefficients) and global POD (fit on the training block) project the held-out block; no forecasting model. The projection
path reproduces the stored p13m5 latent exactly (`--check`).
- **The PIV content below ~10-point wavelengths is white in time.** Frame-to-frame (lag-1, phase-aligned) coherence by
  streamwise band: 0.93 / 0.91 / 0.67 / 0.26 at wavelengths 154-77 / 77-38 / 38-19 / 19-10 points, then 0.044 / 0.028 /
  0.026 at 10-5 / 5-2.4 / <2.4 against 0.025 for white noise. Those bands hold ~5 % of the fluctuation energy; the spectrum
  flattens to a plateau below ~3 points. Measurement noise (or decorrelation within one frame): unforecastable either way.
  Noise share estimates 2 % (spatial plateau) to 7-11 % (temporal extrapolation).
- **p13m5 discards only that white content.** Its residual (NMSE 0.036) has lag-1 correlation 0.003 even along the
  convection (best shift dx = -4 points/frame), and a flat frequency spectrum (17.2 dB, bins 4-63). Coherent-field NMSE
  (from lag-1 covariances) 1e-4. p13m5 over-resolves: it keeps 66 % of the white 10-5 point band.
- **Element size barely matters at a fixed coefficient count above ~1,000**; the curves collapse. Below ~1,000, global POD
  keeps more coherent content (POD 700: 3e-3 vs best DLS 6e-3); above, DLS keeps more (3,240: DLS 1e-4, POD 8e-4) because
  POD spends coefficients on noise (keeps 44 % of the 5-2.4 point band vs 13 %). Local smooth bases act as a spatial noise filter.
- **Half the coefficients lose nothing coherent:** p13m2 (1,620), p17m4 (1,680), p21m8 (1,836): coherent NMSE 4-9e-4,
  coherent RMS map 0.12-0.16 of the sampling floor (p13m5 0.10).
- Basis sampling (3 seeds) changes nothing (spread < 1 %); modes from 200 training snapshots instead of snapshot 0: no gain
  (slightly worse for large patches). About 75 % of every element's directions lie under the 1e-2 ridge for every basis.
- **Paper statistic affected:** flow_stats `shape_x_db` includes the white bands. On bands above 4.8 points the deterministic
  rollouts are 30-40x the sampling floor instead of 50-57x (hybrid one sample 4.6x instead of 16x); the conclusion holds.
  Better: restrict spectral shape to the coherent bands (> ~10 points) or use lag-1 statistics, and say why.
- **Forecasting test running (2026-11-11 ~15:45, queues 16-19, one seed each, the paper's recipe):** p9m2 (3,360, 507 tokens),
  p13m2 (1,620, 234), p17m4 (1,680, 140), p21m5 (1,224, 80), p21m8 (1,836, 80), p21m15 (3,264, 80);
  configs `challenge/spatial_gram_rollout_galerkin_fc_all_d128_ca0.1_p*m*.json`. Then challenge_eval.py and flow_stats.py.

Forecasting test finished 2026-11-11 (one seed each, paper recipe, only the basis changed; scratch scores in
results/cavity_challenge_eval.json). 1-30 NMSE / rollout energy ratio / rms_map (x floor):
p13m5 (3 seeds) 0.765 +/- 0.023 / 0.20-0.23 / 8.0-8.3; p9m2 0.774 / 0.31 / 6.9; p13m2 0.781 / 0.17 / 9.2;
p17m4 0.772 / 0.23 / 8.2; p21m5 0.813 / 0.09 / 10.3; p21m8 0.848 / 0.09 / 10.1; p21m15 0.874 / 0.01 / 12.6.
Reading: halving the latent (p13m2, p17m4) forecasts the same within seed spread; 21-point elements are worse at every
size and worst with the most modes; p9m2 at matched size keeps the most rollout energy (single seed, unconfirmed).
Decision (user, 2026-11-11): paper statistics stay on all bands; note where differences sit in the white bands.


## Element energy coordinates (orthonormal tokens), NOT pursued (user, 2026-11-11: a complication over the current implementation)

Question from the user: would orthogonalizing the basis help stability or cost? Prior analysis: papers/random/sgfem_surrogate.tex
(2026-10-02) recommended removing the near-null subspace from what the network sees and proposed a retraining test that
was never run. Check on the current latents (scratchpad ortho_check.py, truth over the held-out block, no training):
- LDC p99m10: element mass matrix has 4 of 44 directions above 1e-2 x mean eigenvalue; the 40 below hold 13 % of the
  field energy, the 30 below 1e-4 hold 1.1 %, the 20 below 1e-6 hold 0.1 %. 24 data-ordered orthonormal directions per
  element lose 0.04 %, 16 lose 0.2-0.3 %.
- Cavity p13m5: 8 of 24 above 1e-2; the 16 below hold 1.4 %, the 11 below 1e-4 hold 0.2 %. 16 data-ordered directions
  lose 0.07 %, 12 lose 0.3 %.
- Assembled M: cond 2.3e10 (LDC), 2.4e11 (cavity); only ~10 global directions below 1e-8 x mean.
Test to run: tokens and loss in z_e = Lambda^(1/2) V^T d_e (truncated, one global input scale, plain MSE, no loss ridge;
assembly right-hand side sum G^T V Lambda^(1/2) z_e with a 1e-8 ridge). Compare with lambda 1e-2 and 1e-4 on Re 15k
(30-step error and 500-step statistics) and on the cavity. No computational gain expected.

## Draft trim (2026-11-11)

Conference-scope trim applied to papers/scitech_2027/main.tex (19 pp, body ends p. 15; was 23 pp). Copies:
main_2026-11-11_before_trim.tex (original) and main_2026-11-11_figs_cut.tex (after Figs 4-5 were cut).
- Cut: Figs 4-5, training-recipe table, compression-level section and p99m5 rows, POD-on-coefficients paragraph,
  gain baseline / Delta_g / alpha and 10-mode rows / beta sweep (all for the journal paper), most Challenge ablation rows.
- New figures: figures/datasets.pdf (scripts/fig_datasets.py), figures/ldc_statistics.pdf and
  figures/cavity_statistics.pdf (scripts/fig_statistics.py, cache results/fig_statistics.npz).
- Challenge headline is now the alpha 0.1 reference over three seeds (0.765 +/- 0.023; steps 1/5/10/30 =
  0.34/0.58/0.76/0.90); figures/challenge_heldout.pdf replots its seed 1 (PLOT_KEYS in scripts/challenge_eval.py).
- Finding from the new cavity figure: free rollouts collapse within ~30 steps onto a single tone (0.94 kHz for seeds 1
  and 3, ~1.5 kHz for seed 2) with ~0.2 of the energy and no broadband content. Now stated in Sec. 6.5 and the abstract.
- Defaults I chose that the user should confirm: lambda = 1e-2 stated as the paper's value with the trade-off in one
  paragraph; "only model" claim removed (SLICK still not in the paper).
- Still open in the draft: DNS citation TODO (Sec. 5.1); architecture table is single-seed on the old split; no POD or
  frequency-extrapolation baseline in it (results/extrap_baseline.json exists).

Later on 2026-11-11 (draft now 20 pp): added the frequency-extrapolation baseline on the paper's truncated-mesh bases
(`scripts/extrap_baseline.py --paper` -> results/extrap_baseline_paper.json): Re 15k error 1.91 at every lead, RMS 9.0 %,
TKE 14.8 %; Re 30k error 1.68, RMS 23.4 % (lower than the transformer's 29.0 %), energy ratio 1.13. Rows in Tables arch
and re30k. Rollout tones on Challenge 2.1 identified as Rossiter modes 2 and 3 (rollout 965/945 Hz and 1520 Hz; PIV
peaks 938 and 1538 Hz; modified Rossiter equation with Zhang et al.'s constants and U = 200 m/s gives 960 and 1552 Hz;
U assumed from Mach 0.6 at room stagnation temperature, not given in Zhang et al.). DNS solver citation added
(Mittal et al. 2008). No TODO markers remain. Copy before these edits: main_2026-11-11_trimmed.tex.
