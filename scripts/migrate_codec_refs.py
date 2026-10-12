"""Replace per-run codec-reference copies with links into one shared file per latent.

One-off migration for results produced before lib/codec_ref existed. For every
results/<latent>/<run>/pred/<name>_<split>_codec_ref_rec.h5 it reads the run's
snapshot ids from split_ids.pkl, builds/extends results/<latent>/<name>_codec_ref_rec.h5,
and swaps the copy for a link -- only after a spot check shows the copy holds the
same reconstruction (lib/codec_ref.link). Copies that fail the check are kept.

Run from the repo root:  python scripts/migrate_codec_refs.py [latent_dir ...]
"""
import glob
import logging
import os
import pickle
import sys

import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lib.codec_ref as codec_ref  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger('migrate')
SPLIT_KEY = {'train_data': 'train_indices', 'eval_data': 'val_indices'}


def recon_fn(latent_dir, name, cfg):
    """Reconstruction of the true coefficients, matching runner._codec_reference."""
    if hasattr(cfg, 'nz_t'):
        import lib.dls as dls
        three_d = True
    else:
        import lib.dls_2d as dls
        three_d = False

    def recon(lo, hi):
        with h5py.File(os.path.join(latent_dir, 'latent_coeff.h5'), 'r') as f:
            g = f[name]
            du, dv = g['dof_u'][lo:hi], g['dof_v'][lo:hi]
            dw = g['dof_w'][lo:hi] if three_d else None
        if three_d:
            return dls.gfem_recon_flexible(rec_target=None, config=cfg, dof_u=du, dof_v=dv, dof_w=dw)
        return dls.gfem_recon_flexible(rec_target=None, config=cfg, dof_u=du, dof_v=dv)
    return recon


def migrate(latent_dir):
    copies = sorted(glob.glob(os.path.join(latent_dir, '*', 'pred', '*_codec_ref_rec.h5')))
    if not copies:
        return 0
    if '_pod_' in os.path.basename(latent_dir.rstrip('/')):
        log.info("%s: POD latent, skipped", latent_dir)
        return 0
    cfg = pickle.load(open(os.path.join(latent_dir, 'latent_config.pkl'), 'rb'))
    freed = 0
    for path in copies:
        run_dir = os.path.dirname(os.path.dirname(path))
        fname = os.path.basename(path)
        group = 'train_data' if fname.endswith('_train_data_codec_ref_rec.h5') else 'eval_data'
        name = fname[:-len(f'_{group}_codec_ref_rec.h5')]
        try:
            ids = pickle.load(open(os.path.join(run_dir, 'split_ids.pkl'), 'rb'))[group][name][SPLIT_KEY[group]]
        except (FileNotFoundError, KeyError) as e:
            log.info("  %s: no split ids (%s), kept", path, e)
            continue
        before = os.path.getsize(path)
        codec_ref.link(path, codec_ref.shared_path(latent_dir, name), ids, recon_fn(latent_dir, name, cfg))
        after = os.path.getsize(path)
        freed += before - after
        log.info("  %-70s %8.2f GB -> %6.1f KB", os.path.relpath(path, latent_dir), before / 1e9, after / 1e3)
    shared = sorted(glob.glob(os.path.join(latent_dir, '*_codec_ref_rec.h5')))
    added = sum(os.path.getsize(s) for s in shared)
    log.info("%s: freed %.1f GB in copies, shared file(s) %.1f GB -> net %.1f GB",
             latent_dir, freed / 1e9, added / 1e9, (freed - added) / 1e9)
    return freed - added


if __name__ == '__main__':
    targets = sys.argv[1:] or sorted(glob.glob('results/*/'))
    net = sum(migrate(t) for t in targets)
    log.info("total net space freed: %.1f GB", net / 1e9)
