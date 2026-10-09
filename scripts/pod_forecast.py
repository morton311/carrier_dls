"""Forecast the global POD coefficients of a decomposed dataset.

The hierarchical model needs a *predicted* global branch, not the true one:
scoring the fine-scale branch against exact POD coefficients measures an upper
bound, since 68% of the fluctuation energy is then handed over for free.

Trains an autoregressive transformer on the (T, K) coefficient record written by
scripts/pod_decompose.py and writes `coeffs_pred` back into the same file: a
copy of the true record with each evaluated window overwritten by a rollout
seeded from `time_lag` true snapshots. Windows are taken from the DLS branch's
split_ids.pkl so both branches are aligned snapshot for snapshot.

Usage:
    python scripts/pod_forecast.py data/Challenge2_1_train_pod25_global.h5 \
        --splits results/cavity_res25_dls_p13m5/hybrid_pod25/split_ids.pkl
"""
from __future__ import annotations

import argparse
import copy
import logging
import os
import pickle
import sys

import h5py
import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lib.models as models

logger = logging.getLogger(__name__)


def make_windows(coeff, idx, time_lag, train_ahead):
    """(N, time_lag, K) inputs and (N, train_ahead, K) targets from window starts."""
    x, y = [], []
    for t in idx:
        t = int(t)
        if t + time_lag + train_ahead > len(coeff):
            continue
        x.append(coeff[t:t + time_lag])
        y.append(coeff[t + time_lag:t + time_lag + train_ahead])
    return torch.from_numpy(np.stack(x)).float(), torch.from_numpy(np.stack(y)).float()


def rollout(model, seed, steps, device):
    """Free-running autoregressive rollout. seed is (time_lag, K)."""
    model.eval()
    inp = torch.from_numpy(seed).float().unsqueeze(0).to(device)
    out = np.zeros((steps, seed.shape[1]), dtype=np.float64)
    with torch.no_grad():
        for n in range(steps):
            nxt = model(inp)
            out[n] = nxt[0].cpu().numpy()
            inp = torch.cat((inp[:, 1:, :], nxt.unsqueeze(1)), dim=1)
    return out


def limit_total(total, max_snapshots=None, snapshot_frac=None):
    """Leading-snapshot cap, so the series stays contiguous."""
    limit = total
    if snapshot_frac is not None:
        limit = min(limit, max(1, int(round(total * float(snapshot_frac)))))
    if max_snapshots is not None:
        limit = min(limit, int(max_snapshots))
    return limit


def train(path, splits, time_lag, train_ahead, d_model, layers, epochs,
          batch_size, lr, patience, grad_clip, device,
          max_snapshots=None, snapshot_frac=None):
    with h5py.File(path, 'r') as f:
        coeff = f['coeffs'][:].astype(np.float64)
    n_snaps, K = coeff.shape
    keep = limit_total(n_snaps, max_snapshots, snapshot_frac)
    if keep < n_snaps:
        logger.info("using %d of %d snapshots (%.1f%%) to fit the forecaster",
                    keep, n_snaps, keep / n_snaps * 100)
    logger.info("POD coefficients: %d snapshots x %d modes", n_snaps, K)

    # per-mode standardization: leading modes carry orders of magnitude more
    # energy than trailing ones, so a single global scale would hide the tail
    mean, std = coeff.mean(0), coeff.std(0)
    std = np.clip(std, std.mean() * 1e-6, None)
    norm = (coeff - mean) / std

    with open(splits, 'rb') as f:
        snaps = pickle.load(f)
    name = next(iter(snaps['train_data']))
    tr_idx = np.asarray(snaps['train_data'][name]['train_indices'])
    te_idx = np.asarray(snaps['train_data'][name]['test_indices'])
    tr_idx = tr_idx[tr_idx + time_lag + train_ahead <= keep]
    te_idx = te_idx[te_idx + time_lag + train_ahead <= keep]
    logger.info("windows: %d train, %d test", len(tr_idx), len(te_idx))

    Xtr, Ytr = make_windows(norm, tr_idx, time_lag, train_ahead)
    Xte, Yte = make_windows(norm, te_idx, time_lag, train_ahead)

    model = models.TransformerEncoderModel(
        time_lag=time_lag, input_dim=K, d_model=d_model, ff_dim=3 * d_model,
        nhead=4, num_layers=layers, activation='swiglu', pre_norm=True).to(device)
    logger.info("model: %d parameters", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    crit = nn.MSELoss()

    best, best_state, since = float('inf'), None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        run = 0.0
        for lo in range(0, len(Xtr), batch_size):
            sel = perm[lo:lo + batch_size]
            inp = Xtr[sel].to(device)
            tgt = Ytr[sel].to(device)
            opt.zero_grad()
            total = None
            # gradients carried across the rollout, as in the DLS branch
            for n in range(train_ahead):
                out = model(inp)
                step = crit(out, tgt[:, n, :])
                total = step if total is None else total + step
                inp = torch.cat((inp[:, 1:, :], out.unsqueeze(1)), dim=1)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            run += float(total.detach()) / train_ahead
        train_loss = run / max(1, (len(Xtr) + batch_size - 1) // batch_size)

        model.eval()
        with torch.no_grad():
            inp, tgt = Xte.to(device), Yte.to(device)
            tot = 0.0
            for n in range(train_ahead):
                out = model(inp)
                tot += float(crit(out, tgt[:, n, :]))
                inp = torch.cat((inp[:, 1:, :], out.unsqueeze(1)), dim=1)
            test_loss = tot / train_ahead
        sched.step(test_loss)

        if test_loss < best * (1 - 0.01):
            best, since = test_loss, 0
        else:
            since += 1
        if test_loss < best or best_state is None:
            best = min(best, test_loss)
            best_state = copy.deepcopy(model.state_dict())
        if ep % 10 == 0 or ep == epochs - 1:
            logger.info("epoch %4d | train %.4e | test %.4e | lr %.2e | patience %d/%d",
                        ep + 1, train_loss, test_loss, opt.param_groups[0]['lr'], since, patience)
        if since >= patience:
            logger.info("early stopping at epoch %d", ep + 1)
            break

    model.load_state_dict(best_state)
    logger.info("best test loss %.4e", best)
    return model, mean, std, norm, snaps, name


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('global_file')
    p.add_argument('--splits', required=True, help="the DLS branch's split_ids.pkl")
    p.add_argument('--time-lag', type=int, default=24)
    p.add_argument('--train-ahead', type=int, default=8)
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--patience', type=int, default=40)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--train-seed', choices=('holdout', 'in_sample'), default='holdout',
                   help="where the train-window rollout is seeded; 'in_sample' "
                        "replays fitted data and is not a forecast")
    p.add_argument('--max-snapshots', type=int, default=None,
                   help='use at most this many leading snapshots of the record')
    p.add_argument('--snapshot-frac', type=float, default=None,
                   help='use at most this fraction of the record')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    np.random.seed(0)

    model, mean, std, norm, snaps, name = train(
        args.global_file, args.splits, args.time_lag, args.train_ahead,
        args.d_model, args.layers, args.epochs, args.batch_size, args.lr,
        args.patience, args.grad_clip, device,
        args.max_snapshots, args.snapshot_frac)

    with h5py.File(args.global_file, 'r') as f:
        pred = f['coeffs'][:].astype(np.float64).copy()

    # Replace each evaluated window with a free-running rollout.
    #
    # The train-window rollout is seeded from inside the held-out test block by
    # default: seeding it at the start of the training range makes it an
    # in-sample replay of data the forecaster fitted, which reports an
    # implausibly good rollout (measured: relative error 0.765 over 11,464 free
    # steps of a broadband flow) and makes the train/eval gap an artefact.
    test_ids = np.asarray(snaps['train_data'][name].get('test_indices', []))
    for group, key in (('train_data', 'train_indices'), ('eval_data', 'val_indices')):
        if group not in snaps or name not in snaps[group]:
            continue
        ids = np.asarray(snaps[group][name][key])
        if len(ids) <= args.time_lag:
            continue
        seed_ids = ids[:args.time_lag]
        if group == 'train_data' and args.train_seed == 'holdout':
            if len(test_ids) >= args.time_lag:
                seed_ids = test_ids[:args.time_lag]
                logger.info("train_data seeded from the held-out block at snapshot %d",
                            int(seed_ids[0]))
            else:
                logger.warning("no held-out block long enough to seed train_data; "
                               "its rollout stays in-sample and should not be quoted")
        seed = norm[seed_ids]
        steps = len(ids) - args.time_lag
        out = rollout(model, seed, steps, device)
        pred[ids[args.time_lag:]] = out * std + mean
        logger.info("%s: rolled out %d steps from snapshot %d",
                    group, steps, int(seed_ids[0]))

    with h5py.File(args.global_file, 'r+') as f:
        if 'coeffs_pred' in f:
            del f['coeffs_pred']
        f.create_dataset('coeffs_pred', data=pred)
    logger.info("wrote coeffs_pred to %s", args.global_file)


if __name__ == '__main__':
    main()
