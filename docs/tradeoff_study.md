# Study plan: forecast error against turbulent fluctuations

Draft, 2026-11-10. This is our own study for the SciTech 2027 paper and the journal article that follows. It is independent of
the advisor's separate study, which is not shared and must not be used or cited.

## Question

A forecast trained on mean-square error drifts toward the conditional mean and loses fluctuation energy as predictability is lost. Recovering
the fluctuations costs error. We want to know how three design choices move that tradeoff:

1. the GFEM enrichment space (how many local modes, how large the elements, how cut elements are treated);
2. the tokenization (which coefficients form one token, and how time enters);
3. the training loss (the metric, statistical terms, and generative training).

## What is already known, and what is not

For any forecast F of a mean-removed field q, with E = |F|^2/|q|^2 (energy kept) and c the pattern correlation,

    NMSE = 1 + E - 2 c sqrt(E)        (exact for every frame; checked to 2e-14 on the cavity)

At a fixed c, the error is lowest at E = c^2 (NMSE = 1 - c^2), and matching the energy costs 2(1 - c). This is the law of cosines behind
Taylor diagrams (Taylor 2001), and the energy-only case of the distortion-perception function D(P) = D* + ((P* - P)+)^2 (Freirich et al.
2021, Thm 1), with D(0) <= 2 D* (Blau & Michaeli 2018, Thm 3). **The paper must not claim the tradeoff.** The open questions are how each design
choice changes c (which lowers the whole curve) and which scales the recovered energy sits in.

Results so far on the cavity's 30 held-out sequences (`scripts/fig_tradeoff.py`, `scripts/tradeoff_scales.py`):

- DLS transformer: NMSE 0.765, E 0.30, c 0.47 (3 seeds).
- A gain per lead, set on 7 validation-block episodes, raises it to NMSE 1.00 at E 0.89 with no training. **Any statistical loss must beat this.**
- The gain does not restore the right scales. Energy kept for wavelengths over 31 / 8-31 / under 8 grid points (the truth's energy split is
  84 / 13 / 3 %): DLS 42 / 4 / 2 %, energy-matched DLS 120 / 9 / 5 %, a hybrid sample 59 / 35 / 16 % at NMSE 1.08.
- The enrichment caps what any forecast can hold. The basis's projection of the truth keeps 100 / 95 / 33 % with 5 modes, 100 / 98 / 50 %
  with 10, and 100 / 99 / 74 % with 20. The deterministic model is far below even the 5-mode ceiling, so the ceiling binds only once a loss
  or a generative model recovers the small scales.
- With the same architecture, long-run energy is 0.40 with cut elements unregularized (one seed) and 0.20-0.24 with alpha = 0.1 (three seeds).
- Earlier statistical terms hurt: `traj_energy` + `traj_var` on Re 15k (`spatial_gram_new_losses`) gave a long-run RMS error of 123 %.

## Measurement protocol

Every point on a tradeoff plot must be **one forecast product**. Generative rows currently report NMSE from the ensemble mean and statistics
from one sample; the study scores the sample and the mean as separate products.

For each run, on the official-format 30 sequences and the 50-start protocol:

| Measure | Why |
|---|---|
| 30-step NMSE, E, c (per lead and averaged) | position on the identity |
| NMSE minus the parent's rescaled curve at the same E | did the change beat a free gain? (negative = yes) |
| Energy kept per wavelength band, and as a fraction of the basis ceiling | which scales were recovered, and how much was possible |
| RMS-map error over the 30-step window | spatial distribution, not just total energy |
| Long-run energy kept, RMS error, spectrum distance, stability | free-rollout statistics, which a gain cannot fix |

Three seeds for any point that a claim rests on (cavity energies vary 0.20-0.24 across seeds; Re 30k RMS error varies 29-55 %).
Implementation: fold `fig_tradeoff.py` and `tradeoff_scales.py` into one scorer that takes a run list and writes these per run.

## Axis A: enrichment

The DLS knobs are the number of local modes m, the patch (element) size, set by n_skip, and the cut-element weight alpha. More modes also means more
coefficients, so the two effects need separating:

- **Richness at fixed element size** (n_skip 6): m = 2, 5, 10, 20.
- **Fixed coefficient budget** (about 3,240): small elements with few modes against large elements with many: about (n_skip 4, m 2),
  (6, 5) and (9, 12). The exact counts depend on the full-coverage mesh, so compute them when the latents are built.
- **Cut elements**: alpha = 0 and 0.1 on the winner (alpha 0 seeds 2-3 are already pending).

For each latent, record the compression NMSE and the per-band ceiling before training. Those cost minutes and show what each space can hold.

## Axis B: tokenization

Implemented today: one token per time step for the whole latent (global temporal, `tr_enc`), per-element temporal tokens without
cross-element attention (element-local), and one token per element with the last tau frames stacked and attention over space (`tr_spatial`, the
paper's model). The existing global and element-local runs are on the old split and predate the inference fix, so retrain them on split 2.

Proposed generalization: **a tokenization is a set of gather operators G_t** (which global coefficients form token t). Overlapping tokens are
combined by the mass-weighted projection we already use (sum G_t^T M_t G_t). That makes these one code path:

- nodes (one token per node, its m + 1 coefficients);
- elements (current);
- 2x2 element patches (overlapping at shared nodes);
- the whole field (one token).

Time is the second choice: frames stacked in the token (current) against separate space-time tokens (about tau times more tokens per step).
Hold the parameter count and training windows fixed when comparing.

## Axis C: losses

| Family | Options | Status |
|---|---|---|
| Base metric | raw coefficients, mass matrix M, regularized M_lambda | implemented (`loss_metric`) |
| Energy term | L = NMSE + beta (sqrt(E) - 1)^2, beta in {0, 0.1, 0.3, 1, 3} | new, simple |
| Scale term | band energies as quadratic forms c^T B_k c, with B_k = D^T F_k^T F_k D (decoder D, band filter F_k) | new |
| Rollout statistics | `traj_energy`, `traj_var`, `traj_dt` | implemented; hurt Re 15k before |
| Generative | hybrid diffusion (mean + diffused residual), multi-step | implemented |

The energy term has an exact reference: at fixed c, the per-frame optimum is sqrt(E) = (c + beta)/(1 + beta). A model trained with weight beta
that lands below that point has raised c, and above it the term only rescaled. The scale term follows the project's theme: like M, the
B_k are precomputed, so band energies are evaluated on coefficients without decoding. The local-mode index cannot stand in for scale, because
on the 7x7-point element every enrichment column has about the same spatial frequency (0.18-0.21 cycles per point). Check the literature before calling
the band matrices new.

## Design and cost

One GPU. Measured wall times: cavity deterministic about 1.5 h, cavity hybrid about 4.5 h, Re 30k deterministic about 2.7 h.

| Stage | Runs | Cost |
|---|---|---|
| 0. Scorer; latent ceilings; retrain global and element-local tokenizations on split 2 | 6 latents built, 2 tokenizations x 3 seeds | about 10 GPU-h |
| 1. Enrichment at fixed model and loss (cavity) | 6 latents x 3 seeds | about 30 GPU-h |
| 2. Tokenization at the chosen enrichment | node and patch tokens implemented; 5 tokenizations x 3 seeds | 1-2 days of code, about 25 GPU-h |
| 3. Losses at the chosen enrichment and tokenization | beta and band sweeps, rollout terms, hybrids | about 50 GPU-h |
| 4. Interactions and confirmation on Re 30k | best of each axis crossed; winners on Re 30k | about 30 GPU-h |

Run the cavity first: it is cheap and has the strongest tradeoff. Re 15k is the control, where predictability is high and the tradeoff should be small.

## What fits the SciTech paper

Assuming a final-manuscript deadline in early December (check it):

1. The identity, the rescaling baseline and the per-band analysis on existing runs (done; figure `figures/tradeoff_cavity.pdf`).
2. Basis ceilings for m = 5, 10, 20 (done for the older 10- and 20-mode latents; rebuild them on the full-coverage mesh).
3. The energy-term beta sweep, one seed, five runs (about 8 GPU-h).
4. One enrichment step: m = 10 with the paper's model, three seeds (about 6 GPU-h).
5. Alpha 0 seeds 2-3 (about 3 GPU-h).

That gives a short subsection, "Forecast error and fluctuation recovery", with one figure: the identity with our models, the beta front,
and energy per band against the ceilings. Tokenization and the full sweep go to the journal article.

## Risks

- A sweep over three axes is large; stage it, and do not cross axes until each is understood alone.
- Energy per band depends on the windowing of a non-periodic PIV field; keep the band edges fixed across the study.
- A gain is applied open loop; it is a baseline for the 30-step forecast, not for free rollouts.
- The cavity is short and noisy; Re 30k is the check that conclusions carry over.

## Statistical measures (proposed 2026-11-10)

RMS and TKE errors are not enough. The TKE error compares two time histories, so once the forecast decorrelates (about 10
steps on the cavity) it scores timing, not statistics. The RMS map says where the energy is, but not at which scales, at
which frequencies, or with what distribution. Proposed set, all computed on the free rollout over the held-out block by
`scripts/flow_stats.py`:

| Measure | Question | Ideal | Sampling floor (cavity) |
|---|---|---|---|
| Energy ratio | how much fluctuation energy | 1 | 0.96 +/- 0.04 |
| RMS-map error | where the energy is | 0 | 0.072 |
| Spatial spectrum shape, dB | which scales (amplitude removed) | 0 | 0.30 |
| Frequency spectrum shape, dB | which time scales and tones (amplitude removed) | 0 | 0.37 |
| One-point PDF shape (standardized W1) | skewness and tails (amplitude removed) | 0 | 0.054 |
| Energy distance in POD space | the joint state distribution | 0 | 0.080 |

Each distance is also divided by its sampling floor: the distance between the held-out truth and equally long segments of
the training record. A ratio near 1 means the rollout matches the truth as well as another stretch of the true flow does. The
basis's projection of the truth is a second reference: the spatial-spectrum shape has a 5.4 dB floor for the 5-mode basis
(the small scales it cannot hold), and every other measure is below the sampling floor.
Amplitude is measured once, by the energy ratio; the spectral and PDF measures are normalized so they score shape only.
The integral time scale was dropped: its ratio between two stretches of the true flow is 1.38 +/- 0.37.

First results (cavity, existing runs): deterministic DLS models score 16-17 dB in spatial shape, 18-22 dB in frequency shape and
about 5x the floor in PDF shape, whatever the architecture or cut-element treatment. Hybrid and diffusion samples score 4.6-4.8 dB,
0.6-1.2 dB and 1.3-1.5x the floor, close to what the basis allows. Seeds of one deterministic model spread by 5 dB in frequency
shape and by 2x in energy distance, so claims need three seeds.

## First SciTech-subset results (2026-11-10, evening)

Scored by `scripts/tradeoff_runs.py` (30 held-out sequences) and `scripts/flow_stats.py` (free rollout).

| Run | NMSE | E | c | vs gain | Bands kept (>31 / 8-31 / <8 pts) |
|---|---|---|---|---|---|
| alpha 0.1, 3 seeds | 0.742-0.789 | 0.28-0.34 | 0.44-0.50 | -0.03 to +0.03 | 33-42 / 3-4 / 1-2 % |
| alpha 0, 3 seeds | 0.770-0.777 | 0.28-0.39 | 0.45-0.49 | -0.01 to +0.01 | 35-50 / 4-5 / 2 % |
| 10 modes, s1 | 0.766 | 0.30 | 0.47 | +0.00 | 39 / 4 / 3 % |
| energy term beta 0.1 / 0.3 / 1 / 3 | 0.876 / 0.861 / 1.41 / 2.13 | 0.53 / 0.68 / 1.51 / 2.55 | 0.44-0.49 | -0.03 to +0.05 | up to 301 / 21 / 11 % |

- alpha 0 and 0.1 are indistinguishable with three seeds (mean NMSE 0.773 and 0.765; seed spread about 0.02).
- 10 modes do not help the deterministic model: it does not use the extra small-scale capacity (ceiling 50 % under 8 pts).
- The energy-term runs did not raise c and lie on the parent's rescaled curve. They overshoot the energy (2.5x at beta 3,
  in sample as well), almost all of it at large scales, and their spectrum and PDF shapes stay at 15-18 dB and about 4x
  the floor. **They are invalid as a test of the energy term**, because of the ridge finding below; rerun after the
  lambda test.

## The loss ridge dominates (found 2026-11-10)

The Gram loss uses M_e + lambda (tr M_e / n) I with lambda = 1e-2. The true coefficients are large along ill-conditioned
directions of M_e (a modest field component along a near-null direction needs a coefficient of order 1/sqrt(mu)), so the
ridge, which should be a small floor, carries most of the norm (`scripts/gram_ridge_check.py`):

| Latent | Ridge share of the truth's norm | Ridge share of forecast errors | Gradient angle to field energy |
|---|---|---|---|
| Challenge cavity, 5 modes | 75-78 % | 85-91 % | 39-51 deg |
| Challenge cavity, 10 modes | 88 % | | |
| LDC Re 15k / 30k, p99m10 | 99.6-99.8 % | 95-99.6 % | 73-84 deg |

So the training loss on the lid-driven cavity is almost entirely the ridge, and its gradient is nearly orthogonal to the
field-energy gradient (the raw latent metric is about 89 deg). The paper's "within about 11 deg" (Sec. on the metric) does
not hold for these latents. The energy term matched a norm that was mostly ridge, which the model cannot predict, so it
inflated the visible field instead. Without the ridge, the metric reproduces the grid energy ratio exactly (0.68 / 0.28 /
0.18 at steps 1 / 10 / 30 for the paper's cavity model).

Running: loss lambda 1e-3, 1e-4 and 0 on the cavity, 1e-6 on Re 15k (predicted truth ridge shares 23 %, 3 %, 0 and 2 %);
the assembly keeps lambda 1e-2. If a small loss lambda is stable and at least as good, rerun the beta sweep with it,
and consider a better-conditioned latent (drop or shrink the near-null directions in the compression) for the journal.

Update (2026-11-10, night). A small loss lambda is not the fix. On Re 15k, gram_lambda 1e-6 (one seed) is worse than
1e-2 (three seeds): 30-step error 0.033 against 0.026-0.028, and the long rollout blows up (energy 21x the truth by the
end). The near-null directions are nearly invisible in the field but feed back through the network's inputs, so the loss
must constrain them. The base lambda stays 1e-2, and the energy term is now scored in field energy alone, in the loss's
units (`frame_energy` with `metric: field`), so its analytic reference applies to the field part of the loss. The beta
sweep is rerunning that way (`*_fen{beta}_field`), with Re 15k gram_lambda 1e-4 and the cavity lambda sweep alongside.
For the paper: describe the loss as the field-energy error plus a ridge that, on these latents, carries most of the loss
and is needed for stable rollouts, rather than as a slightly regularized field energy.

Enrichment, 3 seeds each (2026-11-11). 10 modes per element against 5, both alpha 0.1, 30 held-out sequences: 30-step
NMSE 0.785 (0.766 / 0.803 / 0.787) against 0.765 (0.742 / 0.789 / 0.763); energy kept 0.28 against 0.30; 1-step
correlation 0.785 against 0.81. In free rollout the two are indistinguishable (shape_x 15-17 dB, shape_t 16-21 dB,
pdf_shape 0.25, all about 50x the sampling floor), although the 10-mode basis alone is closer to the truth (shape_x 3.6
dB against 5.4 dB, pdf_shape 0.016 against 0.018). The richer space raises the ceiling; an MSE-trained deterministic
model does not reach into it, so enrichment and loss have to be studied together.

Loss ridge on the Challenge cavity, single seeds (2026-11-11; assembly lambda stays 1e-2). 30-step NMSE: lambda 1e-4
0.732, lambda 0 0.783, against 0.742-0.789 for three seeds at 1e-2, so no NMSE effect yet. Both trained and rolled out
stably (1,276 free steps; energy 0.15 and 0.30 against 0.20-0.23), unlike Re 15k at 1e-6: the cavity basis is better
conditioned (ridge 77 % of the truth norm, not 99.8 %). Both have a better spatial spectrum shape than every 1e-2 seed
(13.3 and 13.7 dB against 16.3-17.2) and more energy below 8 points (0.03-0.04 against 0.01-0.02); lambda 1e-4 also has
the best frequency shape of any deterministic run (12.4 dB against 17.2-21.2). The ridge holds back small-scale content.
Pending: lambda 1e-3, seeds if the shape gain holds, and Re 15k at 1e-4. The paper's "lambda 0 drifts" sentence needs a
basis qualifier if this holds (it is true for the p99m10 lid-driven latents and the p13m20 cavity basis).

Re 15k at loss lambda 1e-4 (one seed, 2026-11-11, full-coverage alpha 0.1 latent): much better at short horizon. 30-step
mean error 0.0153 against 0.026-0.028 for three seeds at 1e-2 (skill 0.953 against 0.915-0.919); step 1 0.0031 against
0.0043-0.0051; step 30 0.024 against 0.035-0.037. The long rollout is stable but slightly worse in statistics: energy
0.99 mean, 0.93 at the end; RMS error 7.0 % against 6.2-6.3 %; TKE error 3.9 % against 1.3-1.9 %. So lambda has an
optimum between 1e-6 (blows up) and 1e-2 (ridge is 99.8 % of the loss). Running: 1e-4 seeds 2 and 3, and 1e-3 and 1e-5
(seed 1) on Re 15k. If the seeds hold, the paper's training lambda should change and the headline numbers be rerun.

Field-energy term (lambda 0), first two betas (2026-11-11, one seed each). beta 0.3: NMSE 0.781, E 0.435 (predicted
0.396 from its own c), c 0.48, vs gain -0.015. beta 1: NMSE 0.879, E 0.637 (predicted 0.559), c 0.47, vs gain +0.004.
The term now does what the per-frame optimum says and moves the model along the gain curve; correlation unchanged.
Free rollout at beta 1: energy 0.71, RMS map 4.0x floor (best deterministic), but shape_x 58x, shape_t 40x, pdf_shape
4.3x, the same as the reference. Amplitude can be trained in; the distribution of the energy across scales cannot, by
an energy term. The hybrid (one sample) reaches 16x / 3.3x / 1.5x. Cavity loss ridge 1e-3: NMSE 0.755, statistics as
1e-2; so on the cavity no consistent ridge effect (1e-4 0.732, 0 0.783), spectral-shape gain only at 1e-4 and 0.

2026-11-11, midday. The background queues died with the previous session at about 07:48 (not the 2 h limit); restarted
detached (setsid) with scripts/run_steps.sh, logs in results/queue_logs/. Field energy beta 3: NMSE 0.967, E 0.753
(predicted 0.750), c 0.456, vs gain +0.036. Predicted vs actual E over the sweep: 0.40/0.43, 0.56/0.64, 0.75/0.75.
Re 15k loss lambda 1e-4, seed 3: 30-step mean error 0.0204 (skill 0.937), against 0.0153 for seed 1 and 0.026-0.028 for
1e-2; RMS error 7.7 %, TKE 4.7 % (1e-2: 6.2-6.3 %, 1.3-1.9 %). The short-horizon gain holds; long-run statistics are
slightly worse in both seeds.

Re 15k loss lambda 1e-4, all three seeds (2026-11-11, 12:02). Against 1e-2 (3 seeds): 30-step mean error 0.0172 +- 0.0028
against 0.0271 +- 0.0008 (-37 %); skill 0.947 +- 0.009 against 0.917 +- 0.002; step 30 0.026 against 0.036. Long rollout:
RMS error 8.4 +- 1.8 % against 6.27 +- 0.05 %; TKE error 5.0 +- 1.3 % against 1.6 +- 0.3 %; final energy 0.92 +- 0.10
(seed 2 0.81) against 1.02. The loss ridge sets a short-horizon accuracy vs long-horizon statistics trade-off, and the
smaller ridge is 3-4x less consistent across seeds. This is the same NMSE-vs-statistics tension, on the lid-driven
cavity, controlled by the loss metric. lambda 1e-3 and 1e-5 (seed 1) are running.

Field-energy sweep complete (one seed each; beta 0.1 recovered from its checkpoint after the resume bug). 30-step:
NMSE 0.744 / 0.781 / 0.879 / 0.967; E 0.37 / 0.43 / 0.64 / 0.75 against predicted 0.35 / 0.40 / 0.56 / 0.75; c 0.51 /
0.48 / 0.47 / 0.46; vs gain -0.035 / -0.015 / +0.004 / +0.036 (reference seeds -0.029 to +0.027). Free rollout: energy
0.30 / 0.41 / 0.71 / 0.71; RMS map 7.5 / 6.2 / 4.0 / 5.5 x floor; shape_x 56 / 59 / 58 / 49 x; shape_t 54 / 50 / 40 / 32 x;
pdf_shape 4.7 / 4.7 / 4.3 / 3.6 x (reference 55 / 54 / 4.8; hybrid sample 16 / 3.3 / 1.5). Paper Table 8 updated.

Re 15k ridge sweep, seed 1 (complete 2026-11-11 12:48). lambda 1e-2 / 1e-3 / 1e-4 / 1e-5 / 1e-6: 30-step mean error
0.0276 / 0.0223 / 0.0153 / 0.0219 / 0.0330; RMS error 6.2 / 6.5 / 7.0 / 47 / 171 %; TKE error 1.3 / 2.3 / 3.9 / 82 / 729 %;
final energy 1.02 / 1.00 / 0.93 / 3.4 / 21. The short horizon improves down to 1e-4; the long-run statistics worsen
monotonically, and below 1e-4 the rollout grows. 1e-3 is a middle point (-19 % 30-step error, RMS +0.3 points, TKE +1
point). A figure of 30-step error and RMS/TKE error against lambda would show the trade-off; seeds at 1e-3 would firm it up.

SLICK on the trade-off (2026-11-11; scripts/slick_tradeoff.py, our port's deterministic forecast fit to our split; not
for the paper until the user has reviewed SLICK). 30 sequences: NMSE 0.863, E 0.244, c 0.378 (ours 0.765, 0.30, 0.47).
By lead, c 0.63 / 0.52 / 0.37 / 0.38 / 0.30 at steps 1 / 5 / 10 / 20 / 30 (ours 0.81 / 0.65 / 0.52 / 0.38 / 0.32); it also
sits at the NMSE-optimal amplitude (step 1: E 0.394, c^2 0.395; amplitude term 0.062 of 0.863). Against our reference's
gain curve at its energy: +0.103. At full energy (E = 1) by gain: SLICK 1.25, ours 1.07. Free rollout: the linear
operator (spectral radius 0.996) decays to the mean flow (energy 0.41 / 0.17 / 0.06 / 0.006 at steps 1 / 10 / 100 / 300),
so its deterministic forecast has no long-run statistics; a statistics comparison needs SLICK's stochastic forcing.

SLICK from the authors' MATLAB (2026-11-11, at the user's request; scripts/matlab/run_slick_ours.m calling release
SLICK_v1.0.1 / v2.0.1, commit d61ba838, unmodified, on our split). Their selection over all 81 candidates picked nDFT 32,
rank 4, lambda1 1e-2, lambda2 1e-4 (validation NMSE 0.85742, spectral radius 0.9958); their test NMSE on our 30 sequences
is 0.86212 (their normalization by the fit-block mean; 0.863 with the dataset mean, as we score everything); DMD 1.0926.
Their forecasts match our port to 3.4e-8 relative, so the trade-off numbers above are unchanged and now come from the
official code: E 0.244, c 0.378, +0.103 against our gain curve, 1.25 at full energy (ours 1.07), free rollout decays to
the mean. Outputs in results/slick_challenge21/matlab_ours/; scripts/slick_tradeoff.py scores them.

Resume-bug retrain (2026-11-11): the paper's 1.9 M alpha-0 seed 1, rerun with its best checkpoint (epoch 9), has
free-rollout energy 0.18 (was 0.40 with the epoch-20 weights) and RMS error 61 % (was 48 %); 30-step NMSE 0.777 (was
0.771). The three alpha-0 seeds now agree (energy 0.17-0.18). A side observation: the later, more-trained checkpoint
kept more than twice the energy at nearly the same NMSE; checkpoint selection by test loss picks the most damped model.

## Compression study (2026-11-11)

See NEXT_STEPS.md, "Compression study, Challenge 2.1". In short: on the cavity PIV everything below ~10-point wavelengths
is white in time (lag-1 coherence at the white-noise level), p13m5 discards only that, element size barely matters at a
fixed coefficient count above ~1,000, and half of p13m5's coefficients (p13m2, p17m4, p21m8) keep the coherent field to
< 1e-3. Statistics that compare spectra should use the coherent bands; noise bands inflate the floor ratios.
