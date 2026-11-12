"""SLICK on the forecast-energy trade-off: (E, c) by lead on our 30 sequences, its own gain curve, free-rollout statistics.

Scores the forecasts written by the authors' own MATLAB benchmark (scripts/matlab/run_slick_ours.m) by default;
--port uses our Python port instead (scripts/slick_baseline.py), which is only a cross-check and not for publication.
When both exist, the largest relative difference between the two sets of forecasts is printed.

    python scripts/slick_tradeoff.py [--port]
"""
import json, os, sys
import numpy as np, h5py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + '/scripts'); os.chdir(ROOT)
from slick_baseline import load_model, predict, read_q
from slick_comparison import CLEAN, OURS, THEIRS, field_metrics, load_truth
import flow_stats as fs
from tradeoff_runs import gain_curve, THETA

MATLAB = 'results/slick_challenge21/matlab_ours/slick_forecasts.mat'
CAL = THEIRS[~CLEAN]
use_port = '--port' in sys.argv
q, shape = read_q()
nx, ny = shape
to_field = lambda out: out.reshape(2, nx, ny, -1).transpose(3, 1, 2, 0)        # (space, H) -> (H, nx, ny, 2)
model = load_model() if (use_port or not os.path.exists(MATLAB)) else None
if not use_port:
    if not os.path.exists(MATLAB):
        sys.exit(f'{MATLAB} not found: run scripts/matlab/run_slick_ours.m first, or pass --port')
    with h5py.File(MATLAB, 'r') as f:                                       # MATLAB v7.3: dimensions reversed
        raw = {k: f[k][:] for k in ('test', 'cal', 'rollout')}
        lo_m, hi_m = int(f['lo'][0, 0]), int(f['hi'][0, 0])
        assert np.array_equal(f['ours'][:].ravel(), OURS) and np.array_equal(f['calib'][:].ravel(), THEIRS[~CLEAN])
    by_start = {**{s: raw['test'][k].T for k, s in enumerate(OURS)}, **{s: raw['cal'][k].T for k, s in enumerate(THEIRS[~CLEAN])}}
print('forecasts from', 'our Python port (cross-check only)' if use_port else 'the authors\' MATLAB benchmark', flush=True)

def raw_forecast(s):
    return predict(model, q[:, s:s + 70]) if use_port else by_start[s].astype(np.float64)

def forecasts(starts, mean):
    return np.stack([to_field(raw_forecast(s)) - mean[None] for s in starts])

if not use_port:
    port = load_model()
    d = max(np.abs(by_start[s] - predict(port, q[:, s:s + 70])).max() / np.abs(by_start[s]).max() for s in OURS)
    print(f'largest relative difference, MATLAB vs port forecasts on the 30 sequences: {d:.1e}', flush=True)

res = {}
for part, starts in (('', OURS), ('cal_', CAL)):
    true, mean, _ = load_truth(starts)
    F = forecasts(starts, mean)
    for k, v in field_metrics(F, true).items():
        res[part + k] = v
m = {k: res[k] for k in ('nmse', 'corr', 'energy')}
mc = {k: res['cal_' + k] for k in ('nmse', 'corr', 'energy')}
print(f"SLICK mean forecast, 30 sequences: NMSE {m['nmse'].mean():.3f}  E {m['energy'].mean():.3f}  c {m['corr'].mean():.3f}")
for k in (0, 4, 9, 19, 29):
    print(f"   step {k + 1:2d}: NMSE {m['nmse'][:, k].mean():.3f}  E {m['energy'][:, k].mean():.3f}  c {m['corr'][:, k].mean():.3f}  c^2 {m['corr'][:, k].mean() ** 2:.3f}")
ex = ((m['corr'] - np.sqrt(m['energy'])) ** 2).mean()
print(f"   1 - c^2 term {(1 - m['corr'] ** 2).mean():.3f}, amplitude term {ex:.3f}")
# position against our reference's gain curve, and SLICK's own gain curve
ours = json.load(open('results/tradeoff_runs.json'))
curve = np.array(ours['gain_curve'])
E = m['energy'].mean()
print(f"   vs our reference's gain curve at E {E:.3f}: {m['nmse'].mean() - np.interp(E, curve[:, 0], curve[:, 1]):+.3f}")
own = gain_curve(m, mc)
i = int(np.argmin(own[:, 1]))
print(f"   SLICK's own gain curve: best NMSE {own[i, 1]:.3f} at E {own[i, 0]:.3f} (theta {THETA[i]:.2f}); at E 1: {np.interp(1.0, own[:, 0], own[:, 1]):.3f}")
print(f"   our reference's gain curve: best NMSE {curve[:, 1].min():.3f} at E {curve[np.argmin(curve[:, 1]), 0]:.3f}; at E 1: {np.interp(1.0, curve[:, 0], curve[:, 1]):.3f}")

# free rollout over the held-out block, as flow_stats does for our models
with h5py.File(fs.DATA, 'r') as f:
    mean = f['mean'][:].astype(np.float32)
with h5py.File(fs.DATA, 'r') as f:
    train = (f['UV'][0:fs.TRAIN_END:5] - mean[None]).reshape(-1, mean.size).astype(np.float64)
_, _, Vt = np.linalg.svd(train, full_matrices=False)
basis = Vt[:fs.K_POD].astype(np.float32); del train
lo, hi = fs.BLOCK[0] + fs.TL, fs.BLOCK[1]
ref = fs.Reference(fs.truth(lo, hi, mean), basis)
if use_port:
    roll = to_field(predict(model, q[:, lo - 70:lo], horizon=hi - lo)).astype(np.float32) - mean[None]
else:
    assert (lo_m, hi_m) == (lo, hi)
    roll = to_field(raw['rollout'].T.astype(np.float64)).astype(np.float32) - mean[None]
sc = ref.score(roll)
fl = json.load(open('results/flow_stats_cavity.json'))['floor']
keys = ['energy', 'rms_map', 'shape_x_db', 'shape_t_db', 'pdf_shape', 'edist_pod', 't_int']
print('SLICK free rollout (1,276 steps):', ' '.join(f'{k} {sc[k]:.3f}' + ('' if k in ('energy', 't_int') else f' ({sc[k] / fl[k]:.1f}x)') for k in keys))
e_t = (roll.reshape(len(roll), -1).astype(np.float64) ** 2).sum(1) / (ref.q.reshape(len(roll), -1).astype(np.float64) ** 2).sum(1).mean()
print('   energy ratio by rollout step 1/10/30/100/300/1000:', ' '.join(f'{e_t[s]:.3f}' for s in (0, 9, 29, 99, 299, 999)))
np.savez('results/slick_challenge21/tradeoff' + ('_port' if use_port else '') + '.npz', **{k: v for k, v in res.items()}, energy_t=e_t)
