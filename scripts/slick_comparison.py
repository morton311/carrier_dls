"""Challenge 2.1: our models against SLICK on matching sequences, official NMSE.

    python scripts/slick_comparison.py infer     # our models' forecasts -> results/slick_challenge21/models.npz
    python scripts/slick_comparison.py report    # table -> results/slick_comparison.json

Two sets of sequences (70 frames given, 30 forecast):
    theirs  20 episodes every 100 frames in the last 2,000 snapshots, the published SLICK benchmark.
            SLICK's scores are the authors' own saved numbers. Forecast windows before snapshot 11,520 lie in
            the block our runs used for early stopping (never for gradients), so 'clean' keeps only the
            13 episodes wholly inside our held-out block. SLICK was fit to 8,800 snapshots here, ours to 10,368.
    ours    30 sequences every 40 frames in our held-out block, as in scripts/challenge_eval.py. SLICK comes from
            the port in scripts/slick_baseline.py, fit to the same 10,368 snapshots as our models.
"""
import json
import logging
import os
import sys

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

from challenge_eval import model_forecasts   # noqa: E402

DATA = 'data/Challenge2_1_train.h5'
OUT = 'results/slick_challenge21'
PUBLISHED = 'papers/challenge_2.1/SLICK/published_results/rank25/results_summary.json'
LAT = 'cavity_dls_p13m5_fca0_1odd/'
SEEDS = [LAT + 'spatial_gram_rollout_galerkin_fc_all_d128_ca0_1' + s for s in ('', '_s2', '_s3')]
HYBRID = LAT + 'spatial_diff_res_fc_all_d128_ca0_1_ms8'
THEIRS = np.arange(10800, 12701, 100)
OURS = np.arange(11520, 12701, 40)
CLEAN = THEIRS >= 11500      # forecast and seed frames all in our held-out block
N_IN, N_OUT = 70, 30


def field_metrics(pred, true, denom=None):
    """Per sequence and step, for (n, 30, nx, ny, 2) mean-removed arrays."""
    ax = (2, 3, 4)
    e = ((pred - true) ** 2).sum(ax)
    t2 = (true ** 2).sum(ax)
    p2 = (pred ** 2).sum(ax)
    return dict(nmse=e / (t2 if denom is None else denom), corr=(pred * true).sum(ax) / np.sqrt(p2 * t2 + 1e-30), energy=p2 / t2)


def load_truth(starts):
    with h5py.File(DATA, 'r') as f:
        mean = f['mean'][:].astype(np.float64)
        lo = int(starts.min())
        q = f['UV'][lo:].astype(np.float64) - mean
    return np.stack([q[s - lo + N_IN:s - lo + N_IN + N_OUT] for s in starts]), mean, q.shape[1:3]


def fit_mean(n):
    with h5py.File(DATA, 'r') as f:
        return f['UV'][:n].mean(0)


def infer():
    logging.basicConfig(level=logging.WARNING)
    starts = np.array(sorted(set(THEIRS) | set(OURS)))
    true, mean, shape = load_truth(starts)
    mu = fit_mean(8800)
    out = {'starts': starts}
    for name, spec, members in [(f's{i + 1}', s, None) for i, s in enumerate(SEEDS)] + [('hybrid', HYBRID, 32)]:
        fc, _, _ = model_forecasts(spec, starts, shape, members)
        for tag, F in fc.items():
            key = name + ('_sample' if 'sample' in tag else '')
            m = field_metrics(F.astype(np.float64), true)
            for k, v in m.items():
                out[f'{key}_{k}'] = v
            # their normalization: the same error over the energy about the 8,800-snapshot mean
            shift = (mean - mu)[None, None]
            out[f'{key}_nmse_fit8800'] = ((F - true) ** 2).sum((2, 3, 4)) / ((true + shift) ** 2).sum((2, 3, 4))
            print(key, 'window NMSE on the union of sequences', round(float(m['nmse'].mean()), 4), flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez(f'{OUT}/models.npz', **out)


def slick_on_ours():
    """The port's forecasts on our 30 sequences, as mean-removed fields like the other models."""
    from slick_baseline import load_model, predict
    true, mean, shape = load_truth(OURS)
    model = load_model()
    with h5py.File(DATA, 'r') as f:
        lo = int(OURS.min())
        uv = f['UV'][lo:].astype(np.float32)
    flat = lambda x: x.transpose(3, 0, 1, 2).reshape(2, x.shape[0], -1).transpose(0, 2, 1).reshape(-1, x.shape[0])
    q = flat(uv)
    F = np.stack([predict(model, q[:, s - lo:s - lo + N_IN]).T.reshape(N_OUT, 2, *shape).transpose(0, 2, 3, 1) for s in OURS])
    return field_metrics(F - mean, true)


def report():
    z = np.load(f'{OUT}/models.npz')
    starts = z['starts']
    res = {}
    idx = {'theirs': np.searchsorted(starts, THEIRS), 'ours': np.searchsorted(starts, OURS)}
    pub = json.load(open(PUBLISHED))
    slick_eps = 1 + np.array(pub['comparisons']['trainingMean']['sequenceDifferences'])
    print(f"SLICK as published on their 20 episodes: {slick_eps.mean():.4f}; on the 13 inside our held-out block: {slick_eps[CLEAN].mean():.4f}")
    dls = {'theirs': [], 'ours': []}
    for key in ('s1', 's2', 's3', 'hybrid', 'hybrid_sample'):
        for name, nkey in (('theirs', f'{key}_nmse_fit8800'), ('ours', f'{key}_nmse')):
            per_seq = z[nkey][idx[name]].mean(1)
            row = dict(window=float(per_seq.mean()), step1=float(z[nkey][idx[name]][:, 0].mean()))
            if name == 'theirs':
                row['clean13'] = float(per_seq[CLEAN].mean())
                row['wins_vs_slick'] = int((per_seq < slick_eps).sum())
                row['wins_vs_slick_clean13'] = int((per_seq[CLEAN] < slick_eps[CLEAN]).sum())
            res.setdefault(key, {})[name] = row
            if key.startswith('s'):
                dls[name].append(per_seq)
        print(key, res[key])
    for name in dls:
        d = np.mean(dls[name], axis=0)
        print(f"mean of DLS seeds on {name}: {d.mean():.4f}" + (f"; clean 13: {d[CLEAN].mean():.4f}" if name == 'theirs' else ''))
    # head to head on our 30 sequences: same training data, validation block, sequences and normalization
    if os.path.exists(f'{OUT}/ours_model.npz'):
        sl = slick_on_ours()
        o = idx['ours']
        d_seq = np.mean([z[f's{i}_nmse'][o].mean(1) for i in (1, 2, 3)], axis=0)
        s_seq = sl['nmse'].mean(1)
        rng = np.random.default_rng(0)
        L = 4       # block bootstrap over adjacent sequences, which are not independent
        boot = []
        for _ in range(5000):
            b = rng.integers(0, len(d_seq), int(np.ceil(len(d_seq) / L)))
            ii = np.mod(b[:, None] + np.arange(L), len(d_seq)).ravel()[:len(d_seq)]
            boot.append((d_seq - s_seq)[ii].mean())
        res['head_to_head'] = dict(
            slick_window=float(s_seq.mean()), dls_window=float(d_seq.mean()),
            dls_seeds=[float(z[f's{i}_nmse'][o].mean()) for i in (1, 2, 3)],
            hybrid_window=float(z['hybrid_nmse'][o].mean()),
            diff=float((d_seq - s_seq).mean()), diff_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            dls_wins=int((d_seq < s_seq).sum()), n=len(d_seq),
            seed_wins=[int((z[f's{i}_nmse'][o].mean(1) < s_seq).sum()) for i in (1, 2, 3)],
            slick_by_step=sl['nmse'].mean(0).tolist(), dls_by_step=np.mean([z[f's{i}_nmse'][o].mean(0) for i in (1, 2, 3)], axis=0).tolist(),
            hybrid_by_step=z['hybrid_nmse'][o].mean(0).tolist())
        for lab, m in (('SLICK', sl), ('DLS', {k: np.mean([z[f's{i}_{k}'][o] for i in (1, 2, 3)], axis=0) for k in ('nmse', 'corr', 'energy')}),
                       ('hybrid', {k: z[f'hybrid_{k}'][o] for k in ('nmse', 'corr', 'energy')})):
            row = {f'{k}@{st}': round(float(m[k][:, st - 1].mean()), 3) for k in ('nmse', 'corr', 'energy') for st in (1, 5, 10, 20, 30)}
            print(lab, round(float(m['nmse'].mean()), 4), row)
            res['head_to_head'][lab + '_at'] = row
        print({k: v for k, v in res['head_to_head'].items() if not k.endswith(('_by_step', '_at'))})
    json.dump(res, open('results/slick_comparison.json', 'w'), indent=1)


if __name__ == '__main__':
    {'infer': infer, 'report': report}[sys.argv[1]]()
