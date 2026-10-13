import logging
import os
import sys
import h5py
import pickle
import copy
import hashlib
import random

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import time

import lib.codec_ref as codec_ref
import lib.init as init
import lib.pod as pod
import lib.models as models
import lib.datas as datas
import lib.decode as decode
import lib.losses as losses
import lib.multistart as multistart
import lib.plotting as pl
from lib.metrics import l2_err_norm

logger = logging.getLogger(__name__)

GRAD_CLIP_MAX_NORM = 0.2
# Relative improvement required to reset the early-stopping counter. Bare `<`
# lets plateau noise reset it indefinitely, so a run never stops.
ES_MIN_DELTA = 0.01
# Shorter patience once the LR scheduler has hit lr_min and cannot reduce further.
LR_FLOOR_PATIENCE = 20
CHECKPOINT_INTERVAL = 5
# Reprint the training-table header every N epochs so a long log stays readable.
EPOCH_HEADER_INTERVAL = 25
MAX_ERROR_HORIZONS = 100


class _RolloutAdapter:
    """Per-model-type glue for the shared rollout driver.

    The three model families differ in only four ways; this normalizes them:
        time_axis   axis of ``inputs`` holding the time window (1 or 2)
        unpack      batch -> (inputs, targets, state); state is a mutable dict
        call        (inputs, state) -> outputs for one rollout step
        after_step  optional per-step side effect (scheduled sampling)
    """

    def __init__(self, time_axis, unpack, call, after_step=None):
        self.time_axis = time_axis
        self.unpack = unpack
        self.call = call
        self.after_step = after_step


class runner(nn.Module):
    def __init__(self, config):
        super(runner, self).__init__()
        self.config = config
        # check if overwrite, mode, name, and log are in config, if not set to default values
        if 'overwrite' not in self.config:
            self.config['overwrite'] = 'x'
        if 'mode' not in self.config:
            self.config['mode'] = 'test'
        if 'name' not in self.config:
            self.config['name'] = 'test'
        if 'log' not in self.config:
            self.config['log'] = 'terminal'
        if 'device' not in self.config:
            self.config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        if 'distributed' not in self.config:
            self.config['distributed'] = False

        self.device = config['device']
        self.paths_bib = self._init_paths_and_logging(config)
        

        self._log_config()

        self._get_grid()

        
        logger.info(self.dim)
        if self.dim == 3:
            logger.info("Using 3D DLS for latent coefficient computation")
            import lib.dls as dls
        else:
            logger.info("Using 2D DLS for latent coefficient computation")
            import lib.dls_2d as dls
        self.dls = dls

        # get model info
        self._get_data()
        # model_params.seed fixes the weight initialization and the training
        # shuffles; set after the latent is loaded so it never changes how a new
        # latent space is compressed (the data split is seeded separately)
        seed = self.config['model_params'].get('seed')
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            logger.info(f"Seed: {seed}")
        self._get_model()
        self._compile_model()

        
        

    def _init_paths_and_logging(self, config):
        is_init_path, paths = init.init_path(config)
        self.data_sources = ['train_data', 'eval_data']
        self.config['group_names'] = paths.data_dict
        if config['mode'] != 'compare' and config['log'] == 'file':
            log_path = paths.log_path
            logging.info("To follow the log in real time, run 'tail -f %s'", log_path)
            file_handler = logging.FileHandler(log_path, mode='w')
            formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(formatter)

            root_logger = logging.getLogger()
            for handler in list(root_logger.handlers):
                root_logger.removeHandler(handler)
            root_logger.addHandler(file_handler)
            root_logger.setLevel(logging.INFO)

            logger.handlers = []
            logger.addHandler(file_handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False

            logger.info("Logging to %s", log_path)
        logger.info('Using device: %s', self.device)
        return paths
    
    def _get_grid(self):
        """Load the grids, in the [x, y(, z)] layout the DLS modules work in.

        A 2D source stored [y, x] is transposed on read by lib/dls_2d, so its
        grids are transposed here too; ``source_is_yx`` records that, since
        fields read straight from the source file are still [y, x].
        """
        import lib.dls_2d as dls_2d

        with h5py.File(self.paths_bib.source_path, 'r') as f:
            self.dim = 3 if 'z_grid' in f else 2
            self.x_grid = f['x_grid'][:]
            self.y_grid = f['y_grid'][:]
            self.z_grid = f['z_grid'][:] if self.dim == 3 else None

            self.source_is_yx = (
                self.dim == 2 and dls_2d._is_yx_ordered(self.x_grid, self.y_grid)
            )
            if self.source_is_yx:
                logger.info("Source is stored [y, x]; transposing grids to [x, y].")
                self.x_grid = self.x_grid.T
                self.y_grid = self.y_grid.T

            if 'x' in f:
                self.x = f['x'][:]
                self.y = f['y'][:]
                self.z = f['z'][:] if 'z' in f else None
            else:
                self.x = self.x_grid.shape[0]
                self.y = self.y_grid.shape[1]
                self.z = self.z_grid.shape[2] if self.dim == 3 else None

    def _to_xy(self, arr: np.ndarray, spatial_axis: int) -> np.ndarray:
        """Swap the two spatial axes of a source-layout array into [x, y]."""
        if not self.source_is_yx:
            return arr
        perm = list(range(arr.ndim))
        perm[spatial_axis], perm[spatial_axis + 1] = perm[spatial_axis + 1], perm[spatial_axis]
        return arr.transpose(perm)

    def _xy_crop(self, nx: int, ny: int) -> tuple:
        """Spatial slices cropping a source-layout array to (nx, ny) in [x, y]."""
        if self.source_is_yx:
            return (slice(None, ny), slice(None, nx))
        return (slice(None, nx), slice(None, ny))


    def _log_config(self):
        logger.info(f"{'#'*20} Configuration {'#'*20}")
        for key, val in self.config.items():
            if isinstance(val, dict):
                logger.info(f"{key}:")
                for sub_key, sub_val in val.items():
                    logger.info(f"  {sub_key}: {sub_val}")
            else:
                logger.info(f"{key}: {val}")

        logger.info(f"{'#'*48}")


    def _get_data(self):
        """
        Load the latent coefficients
        """
        logger.info(f"{'#'*20}\t{'Loading data...':<20}\t{'#'*20}")
    
        self._compute_latent_coefficients()

        # load latent_config 
        with open(self.paths_bib.latent_config_path, 'rb') as f:
            self.l_config = pickle.load(f)

        self._latent_split()

        # load dof scaler if exists
        scaler_path = os.path.join(self.paths_bib.model_dir, 'dof_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.dof_mean, self.dof_std = pickle.load(f)
            logger.info(f"DOF scaler loaded from {scaler_path}")
        else:
            logger.info(f"No DOF scaler found at {scaler_path}. Will compute before training.")


        

    def _compress_data(self, data_path, group_name, latent_config=None):
        """Dispatch latent-coefficient computation for one dataset to the
        configured latent backend ('dls' or 'pod'), writing the coefficients to
        the latent file under ``group_name``. Returns the latent config object.

        ``latent_config`` reuses an existing basis (modes) so that every dataset
        shares the same latent space; pass ``None`` for the source dataset that
        defines the basis.
        """
        lp = self.config['latent_params']
        max_snaps = self._compress_limit(data_path)
        if max_snaps is not None and (lp['type'] != 'dls' or self.dim == 3):
            raise NotImplementedError("latent_params.compress_used_only is implemented for 2D DLS only")
        if lp['type'] == 'dls':
            extra = {} if max_snaps is None else {'max_snaps': max_snaps}
            return self.dls.gfem_compress_flexible(
                data_source=data_path,
                field_name='UV',
                group_name=group_name,
                patch_size=lp['patch_size'],
                num_modes=lp['num_modes'],
                latent_target=self.paths_bib.latent_path,
                batch_size=lp['batch_size'],
                dls_config=latent_config,
                **extra,
            )
        elif lp['type'] == 'pod':
            if lp.get('localized', False):
                raise ValueError("latent_params.type 'pod' does not support localized=true")
            return pod.pod_compress(
                data_source=data_path,
                field_name='UV',
                group_name=group_name,
                num_modes=lp['num_modes'],
                latent_target=self.paths_bib.latent_path,
                batch_size=lp['batch_size'],
                pod_config=latent_config,
            )
        else:
            raise ValueError(f"latent_params.type '{lp['type']}' not recognized. Use 'dls' or 'pod'.")

    def _compute_latent_coefficients(self):

        logger.info("Computing latent coefficients...")
        # compute the latent coefficients from source data
        logger.info(f"Source path: {self.paths_bib.source_path}")

        source_name = self.config['latent_params']['source_name']
        # Compute the source dataset first; it defines the latent basis that all
        # other datasets are projected onto.
        source_exists = False
        if os.path.exists(self.paths_bib.latent_path):
            with h5py.File(self.paths_bib.latent_path, 'r') as f:
                source_exists = source_name in f

        if source_exists:
            logger.info(f"Latent coefficients for source {source_name} already exist in {self.paths_bib.latent_path}, skipping computation.")
        else:
            logger.info(f"Computing latent coefficients for source {source_name}...")
            latent_config = self._compress_data(
                data_path=self.paths_bib.source_path,
                group_name=source_name,
                latent_config=None,
            )
            with open(self.paths_bib.latent_config_path, 'wb') as f:
                pickle.dump(latent_config, f)
            logger.info("Latent coefficient config saved")

        # load latent_config for use in computing latent coefficients for train and eval data
        with open(self.paths_bib.latent_config_path, 'rb') as f:
            latent_config = pickle.load(f)

        for data_source in self.data_sources:
            logger.info(f"Processing data source {data_source} for latent coefficient computation...")
            if self.config[data_source] is not None:
                for id, source in enumerate(self.config[data_source]):
                    source_config = source
                    path = source.get('path')
                    path = self.paths_bib.data_dir + path + '.h5'
                    data_name = source.get('name')

                    if path == self.paths_bib.source_path:
                        logger.info(f"Source data {self.config['latent_params']['source_name']} found in {data_source}, skipping latent coefficient computation for this data.")
                        continue
                    else:
                        # Check if data_name group already exists in latent file
                        with h5py.File(self.paths_bib.latent_path, 'r') as f:
                            if data_name in f:
                                logger.info(f"Group {data_name} already exists in latent file, skipping computation.")
                                continue

                        logger.info(f"Computing latent coefficients for {data_source} {data_name}...")
                        self._compress_data(
                            data_path=path,
                            group_name=data_name,
                            latent_config=latent_config,
                        )
        

    def _latent_split(self):
        # Load existing splits if available
        if os.path.exists(self.paths_bib.model_dir + 'split_ids.pkl'):
            with open(self.paths_bib.model_dir + 'split_ids.pkl', 'rb') as f:
                snaps = pickle.load(f)
                logger.info(f"Train, test, and validation indices loaded from {self.paths_bib.model_dir + 'split_ids.pkl'}")
        else:
            snaps = {}

        # Initialize data source dicts if needed
        for data_source in self.data_sources:
            if data_source not in snaps:
                snaps[data_source] = {}

        # Compute splits for missing [data_source][data_name] combinations
        with h5py.File(self.paths_bib.latent_path, 'r') as f:
            for data_source in self.data_sources:
                if self.config[data_source] is not None:
                    for id, source in enumerate(self.config[data_source]):
                        path = source.get('path')
                        path = self.paths_bib.data_dir + path + '.h5'
                        data_name = source.get('name')
                        
                        # Check if splits already exist for this combination
                        if data_name in snaps[data_source] and 'train_indices' in snaps[data_source][data_name]:
                            logger.info(f"Splits already exist for {data_source} '{data_name}', skipping")
                            version = snaps[data_source][data_name].get('split_version', 1)
                            # only the training split's test windows drive early stopping
                            if data_source == 'train_data' and version < self.SPLIT_VERSION:
                                logger.warning(
                                    f"{data_source} '{data_name}' uses split version {version}: its test "
                                    "windows overlap the training data, so early stopping watched "
                                    "in-sample loss (validation is unaffected). See _split_indices.")
                            continue
                        
                        # Compute splits for this combination
                        logger.info(f"Computing splits for {data_source} '{data_name}...")
                        if data_name not in snaps[data_source]:
                            snaps[data_source][data_name] = {}
                        
                        snaps[data_source][data_name]['total'] = self._limit_snapshots(
                            f[data_name]['dof_u'].shape[0], source, data_name)
                        logger.info(f"Total snapshots for {data_source} '{data_name}': {snaps[data_source][data_name]['total']}")

                        if data_source == 'train_data':
                            indices = self._split_indices(snaps[data_source][data_name]['total'], 
                                                        train_split=source['train_split'], 
                                                        test_split=source['test_split'])
                        elif data_source == 'eval_data':
                            indices = self._split_indices(snaps[data_source][data_name]['total'], 
                                                        train_split=1-source['pred_split'])

                        snaps[data_source][data_name]['train_indices'] = indices['train_indices']
                        snaps[data_source][data_name]['test_indices'] = indices['test_indices']
                        snaps[data_source][data_name]['val_indices'] = indices['val_indices']
                        snaps[data_source][data_name]['split_version'] = indices['split_version']
        
        # Save updated splits
        with open(self.paths_bib.model_dir + 'split_ids.pkl', 'wb') as f:
            pickle.dump(snaps, f)
        logger.info(f"Train, test, and validation indices saved to {self.paths_bib.model_dir + 'split_ids.pkl'}")
            
        with h5py.File(self.paths_bib.latent_path, 'r') as f:
            latent_keys = list(f.keys())
            if self.config['latent_params']['type'] == 'dls':
                if self.config['latent_params'].get('localized', False):
                    input_dim = self.dim * self.l_config.dof_elem
                else:
                    input_dim = self.dim * self.l_config.num_gfem_nodes * self.l_config.dof_node
            elif self.config['latent_params']['type'] == 'pod':
                # one temporal coefficient block (num_modes) per velocity component
                input_dim = self.dim * self.l_config.num_modes
        logger.info(f"Input dimension for model: {input_dim}")
        self.config['model_params']['input_dim'] = input_dim
        self.indices = snaps
            

    def _compress_limit(self, data_path):
        """Leading snapshots of `data_path` the configured splits use, or None for all.

        Only with latent_params.compress_used_only. The latent is otherwise
        built from the whole file even when max_snapshots / snapshot_frac use
        a fraction of it -- reading all 376 GB of the 60k-snapshot Re 30k
        record to use 4,500 snapshots. Every train/eval entry that reads the
        file counts, so the latent covers the largest of them.
        """
        if not self.config['latent_params'].get('compress_used_only', False):
            return None
        with h5py.File(data_path, 'r') as f:
            total = f['UV'].shape[0]
        limits = [self._limit_snapshots(total, src, src.get('name'), log=False)
                  for group in self.data_sources for src in (self.config.get(group) or [])
                  if self.paths_bib.data_dir + src.get('path') + '.h5' == data_path]
        if not limits:
            return None
        limit = max(limits)
        return None if limit >= total else limit

    def _limit_snapshots(self, total, source, name, log=True):
        """Cap how much of a record is used, per data source.

        ``max_snapshots`` (a count) and ``snapshot_frac`` (a fraction) both take
        the LEADING snapshots, so the series stays contiguous and the temporal
        train/test/prediction splits below keep their meaning. For a record far
        larger than the dynamics need, this bounds training volume without
        re-generating the latent space. The limit applies to the splits, not to
        latent compression, which still processes the file it is handed.
        """
        limit = total
        frac = source.get('snapshot_frac')
        if frac is not None:
            limit = min(limit, max(1, int(round(total * float(frac)))))
        cap = source.get('max_snapshots')
        if cap is not None:
            limit = min(limit, int(cap))
        if limit < total and log:
            logger.info("Limiting '%s' to %d of %d snapshots (%.1f%%)",
                        name, limit, total, limit / total * 100)
        if log and cap is not None and int(cap) > total:
            logger.warning("'%s': max_snapshots=%d but the latent holds only %d snapshots "
                           "(if the source has more, recompress with overwrite 'l')", name, int(cap), total)
        return limit

    # recorded with each run's split and metrics; see _split_indices
    SPLIT_VERSION = 2

    def _split_indices(self, total_snaps, train_split=0.8, test_split=0.1, sample_train=0, sample_test=0):
        """Contiguous train / test / validation blocks of a record.

        The record [0, total) splits into training [0, a), test [a, b) and
        validation [b, total), with b = int(total * train_split) and
        a = b - int(b * test_split). Train and test hold the start index of every
        (time_lag + train_ahead) window that fits inside their own block, so no
        test window touches a training snapshot or the reverse; validation holds
        every snapshot id of its block (eval rolls out from them).

        Version 1 (before 2026-10-13) let the training windows run through the
        test block and started the test windows at 0, inside the training data,
        so early stopping and the LR schedule watched in-sample loss. Existing
        runs keep the split cached in their split_ids.pkl. Pure: _latent_split
        does the caching (this used to write that file itself, which handed a
        second data source the first one's split).
        """
        mp = self.config['model_params']
        window = dict(time_lag=mp['time_lag'], train_ahead=mp['train_ahead'], seed=42)
        b = int(total_snaps * train_split)
        a = b - int(b * test_split)
        train = np.sort(datas.sample_series_indices(a, sample_train, **window))
        test = a + np.sort(datas.sample_series_indices(b - a, sample_test, **window))
        val = np.arange(b, total_snaps)
        if test_split > 0 and len(test) == 0:
            raise ValueError(f"test block of {b - a} snapshots holds no {mp['time_lag'] + mp['train_ahead']}-step "
                             "window; raise test_split")

        logger.info(f"{'Set':<12}|{'Block':<14}|{'Windows / ids':<14}|{'Sampled':<10}")
        logger.info("-" * 52)
        logger.info(f"{'Train':<12}|{f'[0, {a})':<14}|{len(train):<14}|{sample_train:<10}")
        logger.info(f"{'Test':<12}|{f'[{a}, {b})':<14}|{len(test):<14}|{sample_test:<10}")
        logger.info(f"{'Validation':<12}|{f'[{b}, {total_snaps})':<14}|{len(val):<14}|{'-':<10}")
        return {'train_indices': train, 'test_indices': test, 'val_indices': val,
                'split_version': self.SPLIT_VERSION}
    
    def _get_model(self):
        """
        Load the model
        """
        # Load the model
        logger.info(f"{'#'*20}\t{'Loading model...':<20}\t{'#'*20}")
        if self.config['model_params']['model_type'] == 'tr_enc':
            self.model = models.TransformerEncoderModel(
                        time_lag=self.config['model_params']['time_lag'],
                        input_dim=self.config['model_params']['input_dim'],
                        d_model=self.config['model_params']['d_model'],
                        ff_dim=self.config['model_params'].get('ff_dim', None),
                        nhead=self.config['model_params']['nhead'],
                        num_layers=self.config['model_params']['num_layers'],
                        embed=self.config['model_params'].get('embed', 'lin'),
                        activation=self.config['model_params'].get('activation', 'relu'),
                        pre_norm=self.config['model_params'].get('prenorm', False)
                        )
        elif self.config['model_params']['model_type'] == 'lstm':
            self.model = models.LSTMModel(
                        time_lag=self.config['model_params']['time_lag'],
                        input_dim=self.config['model_params']['input_dim'],
                        hidden_dim=self.config['model_params']['d_model'],
                        num_layers=self.config['model_params']['num_layers'],
                        batch_size= self.config['model_params']['batch_size'],
                        )
            
        elif self.config['model_params']['model_type'] == 'tr_encdec':
            mp = self.config['model_params']
            self.model = models.GlobalLocalTransformer(
                        time_lag=mp['time_lag'],
                        input_dim=mp['input_dim'],
                        d_model=mp['d_model'],
                        ff_dim=mp.get('ff_dim', 4 * mp['d_model']),
                        nhead=mp['nhead'],
                        num_layers=mp['num_layers'],
                        enc_num_layers=mp.get('enc_num_layers', 2),
                        enc_nhead=mp.get('enc_nhead', mp['nhead']),
                        enc_ff_dim=mp.get('enc_ff_dim', mp.get('ff_dim', 4 * mp['d_model'])),
                        activation=mp.get('activation', 'relu'),
                        pre_norm=mp.get('prenorm', False),
                        spatial_dim=self.dim,
                        num_freqs=mp.get('coord_freqs', 6),
                        context_window=mp.get('context_window', 1),
                        num_context_tokens=mp.get('num_context_tokens', 0),
                        causal=mp.get('causal', True),
                        context_dim=(None if self._pod_context() is None
                                     else self._pod_context().shape[1]),
                        residual=mp.get('residual', False),
                        )
        elif self.config['model_params']['model_type'] == 'tr_spatial':
            mp = self.config['model_params']
            self.model = models.SpatialTransformer(
                        time_lag=mp['time_lag'],
                        input_dim=mp['input_dim'],
                        d_model=mp['d_model'],
                        ff_dim=mp.get('ff_dim', 4 * mp['d_model']),
                        nhead=mp['nhead'],
                        num_layers=mp['num_layers'],
                        activation=mp.get('activation', 'relu'),
                        pre_norm=mp.get('prenorm', False),
                        spatial_dim=self.dim,
                        num_freqs=mp.get('coord_freqs', 6),
                        residual=mp.get('residual', False),
                        )
        elif self.config['model_params']['model_type'] == 'f_extrap':
            self.model = None
        else:
            raise ValueError(f"Model {self.config['model_params']['model_type']} not recognized. Please use 'tr_enc', 'tr_encdec', 'tr_spatial', or 'lstm'.")
        
        # Load the model weights if they exist and overwrite is not set to 'l' or 'm'
        if os.path.exists(self.paths_bib.model_path) and not self.config['overwrite'] in ['l', 'm']:
            weights = torch.load(self.paths_bib.model_path, weights_only=True, map_location=self.device)
            if not self.config['distributed']:
                weights = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in weights.items()}
            self.model.load_state_dict(weights)
        if self.model is not None:
            self.num_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model initialized with {self.num_params} parameters")

            # print model summary
            if self.config['latent_params'].get('localized', False):
                input_size = (self.config['model_params']['batch_size'], self.config['model_params']['time_lag'], self.l_config.dof_elem * self.dim)
            else:
                input_size = (self.config['model_params']['batch_size'], self.config['model_params']['time_lag'], self.config['model_params']['input_dim'])

            if self.config['model_params']['model_type'] not in ('tr_encdec', 'tr_spatial'):
                try:
                    from torchinfo import summary
                except ImportError:
                    logger.info("torchinfo not installed; skipping the model summary")
                else:
                    logger.info("Model summary:\n%s", summary(self.model, input_size=input_size, verbose=0))

        if self.config['distributed']:
            from torch.nn.parallel import DistributedDataParallel as DDP
            local_rank = int(os.environ["LOCAL_RANK"]) # automatically set by torchrun
            device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank)
            logger.info("Model wrapped in DistributedDataParallel for distributed training")
        else:
            self.model = self.model.to(self.device)

    def _compile_model(self):
        """
        Compile the model
        """
        # Define the loss function and optimizer
        logger.info(f"{'#'*20}\t{'Compiling model...':<20}\t{'#'*20}")

        if self.model is not None: 
            
            mp_ = self.config['model_params']
            self.criterion = losses.build_loss(mp_.get('loss', 'mse'))
            self.rollout_cfg = self._rollout_config()
            self.loss_plan = losses.plan(self.criterion)
            if self.loss_plan.needs_trajectory:
                ta = mp_.get('train_ahead', 1)
                if ta < self.loss_plan.min_train_ahead:
                    raise ValueError(
                        f"the loss needs train_ahead >= {self.loss_plan.min_train_ahead}, "
                        f"config has {ta}")
                if self.rollout_cfg['mode'] != 'bptt':
                    raise ValueError(
                        "trajectory loss terms need rollout.mode = 'bptt'; under "
                        "'detached' the rollout graph is cut and they would "
                        "contribute no gradient")
                logger.info("Trajectory loss terms: %d (scored once per rollout)",
                            len(self.loss_plan.traj_terms))
            self.last_grad_norm = float('nan')
            self.last_skipped = 0
            self.last_steps = 0
            self.last_terms = None
            self._term_totals = None
            self._term_count = 0
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['model_params']['lr'])
            
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config['model_params'].get('lr_factor', 0.5),
                patience=self.config['model_params'].get('lr_patience', 5),
                min_lr=self.config['model_params'].get('lr_min', 1e-6)
            )
            # torch.amp.GradScaler exists from torch 2.3; fall back for older versions
            # growth_interval bounds how fast the loss scale climbs. The default
            # (2000) doubles it every ~5.6 epochs here, so after ~22 epochs the
            # scaled gradients reach the fp16 ceiling and overflow in bursts.
            scaler_kwargs = {
                'init_scale': mp_.get('amp_init_scale', 65536.0),
                'growth_interval': mp_.get('amp_growth_interval', 2000),
            }
            if hasattr(torch.amp, 'GradScaler'):
                self.scaler = torch.amp.GradScaler(**scaler_kwargs)
            else:
                self.scaler = torch.cuda.amp.GradScaler(**scaler_kwargs)

            logger.info(f"Loss function: {self.criterion}")
            logger.info(f"Rollout: {self.rollout_cfg}")
            lam = mp_.get('reassemble_lambda')
            logger.info("Reassembly projection: " + ("average of shared nodal copies" if lam is None
                                                      else f"regularized Galerkin, lambda={lam:g}"))
            logger.info(f"Optimizer: {self.optimizer}")
            logger.info(f"Scheduler: ReduceLROnPlateau (factor={self.config['model_params'].get('lr_factor', 0.5)}, patience={self.config['model_params'].get('lr_patience', 5)})")
            logger.info(f"Using mixed precision training with GradScaler: {self.scaler}")

            # Helper function to remap 'embed' keys to 'input_projection'
            def remap_embed_keys(state_dict):
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('embed'):
                        new_k = k.replace('embed', 'input_projection', 1)
                        new_state_dict[new_k] = v
                    else:
                        new_state_dict[k] = v
                return new_state_dict

            # if checkpoint file exists, model doesn't exist, and overwrite is not set to 'l' or 'm', load the checkpoint
            check_flag = os.path.exists(self.paths_bib.checkpoint_path)
            model_flag = os.path.exists(self.paths_bib.model_path)
            if check_flag and not model_flag and not self.config['overwrite'] in ['l', 'm']: 
                logger.info(f"Loading checkpoint from {self.paths_bib.checkpoint_path}")
                checkpoint = torch.load(self.paths_bib.checkpoint_path, weights_only=True, map_location=self.device)
                checkpoint['model_state_dict'] = remap_embed_keys(checkpoint['model_state_dict'])
                # strip 'module.' from state dict keys if present (from DDP)
                if not self.config['distributed']:
                    checkpoint['model_state_dict'] = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in checkpoint['model_state_dict'].items()}
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'lr_scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                if 'scaler_state_dict' in checkpoint:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                self.epoch = checkpoint['epoch']
                self.losses = checkpoint['losses']
                self.test_losses = checkpoint['test_losses']
                self.early_stop_counter = checkpoint['early_stop_counter']
                logger.info(f"Checkpoint loaded")
                self.checkpointed = True

            # if model exists and overwrite is not set to 'l' or 'm', load the model and skip training
            elif model_flag and not self.config['overwrite'] in ['l', 'm']:
                logger.info(f"Model already exists at {self.paths_bib.model_path}. Skipping training.")
                logger.info(f"Loading model weights from {self.paths_bib.model_path}")
                state_dict = torch.load(self.paths_bib.model_path, weights_only=True, map_location=self.device)
                state_dict = remap_embed_keys(state_dict)
                if not self.config['distributed']:
                    state_dict = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in state_dict.items()}
                self.model.load_state_dict(state_dict)
                self.checkpointed = False
            else:
                logger.info(f"Model does not exist at {self.paths_bib.model_path}. Training from scratch.")
                self.checkpointed = False



    def train(self) -> None:
        """Train the model."""
        # Load the latent coefficients
        logger.info(f"{'#'*20}\t{'Training model...':<20}\t{'#'*20}")
        self._get_train_data()
        self._apply_loss_metric()
        self._model_fit()

    def _apply_loss_metric(self) -> None:
        """Optionally re-express the loss in field-energy coordinates.

        `model_params.loss_metric`:
            'latent' (default)  score raw latent DOF error, as before
            'gram'              score DLS field-reconstruction energy

        Needs dof_std, so it runs after _get_train_data rather than at compile
        time. Applies to whichever source the loss is built for; with several
        train sources the first one's scaler is used, since the criterion is
        shared across them.
        """
        if self.model is None:
            return
        mode = self.config['model_params'].get('loss_metric', 'latent')
        if mode == 'latent':
            return
        if mode != 'gram':
            raise ValueError(
                f"model_params.loss_metric '{mode}' not recognized. Use 'latent' or 'gram'.")
        name = next(iter(self.dof_std))
        metric = decode.build_gram_metric(self, name)
        if metric is None:
            logger.warning("loss_metric='gram' requested but unavailable; "
                           "falling back to the latent metric.")
            return
        self.criterion = losses.MetricLoss(self.criterion, metric).to(self.device)
        self.loss_plan = losses.plan(self.criterion)
        logger.info(f"Loss re-expressed in field-energy coordinates: {self.criterion}")


    def _load_dof_rows(self, dof_u, dof_v, dof_w, indices, lltogl_mat=None) -> torch.Tensor:
        """Load and concatenate DOF rows for the given time indices.

        Returns shape (n, n_elems, dim*dof_elem) when lltogl_mat is provided,
        or (n, dim*num_dofs) otherwise.
        """
        rows = []
        for idx in indices:
            comps = [np.array(dof_u[idx, :]), np.array(dof_v[idx, :])]
            if dof_w is not None:
                comps.append(np.array(dof_w[idx, :]))
            if lltogl_mat is not None:
                comps = [c[lltogl_mat] for c in comps]
                rows.append(np.concatenate(comps, axis=1))
            else:
                rows.append(np.concatenate(comps, axis=0))
        return torch.from_numpy(np.stack(rows)).float()

    def _get_dof_sequence(self, dof_u, dof_v, dof_w, idx: int, length: int,
                          name: str, lltogl_mat=None) -> torch.Tensor:
        """Read a contiguous block of length DOF rows starting at idx, then normalize."""
        u_rows = np.array(dof_u[idx:idx + length, :])
        v_rows = np.array(dof_v[idx:idx + length, :])
        comps = [u_rows, v_rows]
        if dof_w is not None:
            comps.append(np.array(dof_w[idx:idx + length, :]))

        if lltogl_mat is not None:
            comps = [c[:, lltogl_mat] for c in comps]
            dofs_cat = np.concatenate(comps, axis=2)
        else:
            dofs_cat = np.concatenate(comps, axis=1)

        dof = torch.from_numpy(dofs_cat).float()
        dof = (dof - self.dof_mean[name]) / self.dof_std[name]
        return dof

    def _dof_stats(self, dofs):
        """Mean/std for normalizing latent DOFs, per model_params.norm.

        'scalar' (default) reproduces the historical reduction over axes (0, 1).
        For localized configs that already yields one value per element-local
        DOF slot, so the two modes agree there; it is only the non-localized
        case, where `dofs` is 2-D, that collapses to a single global scalar.

        'per_dof' always reduces over every axis but the last, giving one
        mean/std per feature. On the cavity latent the per-DOF spread is ~3e3,
        so a single scale leaves some coefficients at ~170 sigma while the
        median sits near 0.2.
        """
        mode = self.config['model_params'].get('norm', 'scalar')
        if mode == 'per_dof':
            dims = tuple(range(dofs.dim() - 1))
        elif mode == 'scalar':
            dims = (0, 1)
        else:
            raise ValueError(
                f"model_params.norm '{mode}' not recognized. Use 'scalar' or 'per_dof'.")
        mean = torch.mean(dofs, dim=dims)
        std = torch.std(dofs, dim=dims)
        if std.ndim > 0:
            # a feature with no variance would otherwise be amplified without
            # bound; floor it relative to the typical scale
            floor = float(std.mean()) * 1e-6
            std = std.clamp_min(floor)
        return mean, std

    def _get_train_data(self):
        """Get training and test data as torch tensors, minimizing memory usage."""
        if self.config['model_params']['model_type'] in ('tr_encdec', 'tr_spatial'):
            return self._get_train_data_grouped()
        if self.model is not None:
            logger.info('Getting training and test data')
            tl = self.config['model_params']['time_lag']
            ta = self.config['model_params']['train_ahead']
            dof_dim = self.config['model_params']['input_dim']
            localized = self.config['latent_params'].get('localized', False)

            lltogl_mat = None
            if localized:
                skipx = self.config['latent_params'].get('skipx', 1)
                skipy = self.config['latent_params'].get('skipy', 1)
                skipz = self.config['latent_params'].get('skipz', 1) if self.dim == 3 else 1
                lltogl_mat = self._compute_lltogl_mat(skipx=skipx, skipy=skipy, skipz=skipz)
                num_unique_elems = lltogl_mat.shape[0]

            self.dof_mean = {}
            self.dof_std = {}
            self.train_loader = {}
            self.test_loader = {}
            self.sampler = {}

            with h5py.File(self.paths_bib.latent_path, 'r') as f:
                for source in self.config['train_data']:
                    name = source.get('name')
                    train_indices = self.indices['train_data'][name]['train_indices']
                    test_indices = self.indices['train_data'][name]['test_indices']

                    if self.config['latent_params']['type'] in ('dls', 'pod'):
                        dof_u = f[name]['dof_u']
                        dof_v = f[name]['dof_v']
                        dof_w = f[name]['dof_w'] if self.dim == 3 else None

                        dofs = self._load_dof_rows(dof_u, dof_v, dof_w, train_indices, lltogl_mat)

                    self.dof_mean[name], self.dof_std[name] = self._dof_stats(dofs)
                    logger.info(f"Normalization '{self.config['model_params'].get('norm','scalar')}': "
                                f"mean/std shapes {tuple(self.dof_mean[name].shape)}, "
                                f"{tuple(self.dof_std[name].shape)}")

                    latent_type = self.config['latent_params']['type']
                    if localized:
                        num_samples = num_unique_elems
                        X_train = torch.zeros(len(train_indices) * num_samples, tl, dof_dim)
                        Y_train = torch.zeros(len(train_indices) * num_samples, ta, dof_dim)
                        X_test = torch.zeros(len(test_indices) * num_samples, tl, dof_dim)
                        Y_test = torch.zeros(len(test_indices) * num_samples, ta, dof_dim)

                        for t, idx in enumerate(train_indices):
                            dof_seq = self._get_dof_sequence(dof_u, dof_v, dof_w, idx, tl + ta, name, lltogl_mat)
                            for iind in range(num_samples):
                                X_train[t * num_samples + iind] = dof_seq[:tl, iind, :]
                                Y_train[t * num_samples + iind] = dof_seq[tl:tl + ta, iind, :]
                        for t, idx in enumerate(test_indices):
                            dof_seq = self._get_dof_sequence(dof_u, dof_v, dof_w, idx, tl + ta, name, lltogl_mat)
                            for iind in range(num_samples):
                                X_test[t * num_samples + iind] = dof_seq[:tl, iind, :]
                                Y_test[t * num_samples + iind] = dof_seq[tl:tl + ta, iind, :]
                    else:
                        X_train = torch.zeros(len(train_indices), tl, dof_dim)
                        Y_train = torch.zeros(len(train_indices), ta, dof_dim)
                        X_test = torch.zeros(len(test_indices), tl, dof_dim)
                        Y_test = torch.zeros(len(test_indices), ta, dof_dim)

                        for t, idx in enumerate(train_indices):
                            dof_seq = self._get_dof_sequence(dof_u, dof_v, dof_w, idx, tl + ta, name)
                            X_train[t] = dof_seq[:tl, :]
                            Y_train[t] = dof_seq[tl:tl + ta, :]
                        for t, idx in enumerate(test_indices):
                            dof_seq = self._get_dof_sequence(dof_u, dof_v, dof_w, idx, tl + ta, name)
                            X_test[t] = dof_seq[:tl, :]
                            Y_test[t] = dof_seq[tl:tl + ta, :]

                    logger.info(f"Data loaded for {name}: X_train {X_train.shape}, X_test {X_test.shape}")

                    if self.config['distributed']:
                        self.train_loader[name], self.sampler[name] = datas.make_dataloader(
                            X_train, Y_train,
                            batch_size=self.config['model_params']['batch_size'],
                            shuffle=True, distributed=True,
                        )
                    else:
                        self.train_loader[name] = datas.make_dataloader(
                            X_train, Y_train,
                            batch_size=self.config['model_params']['batch_size'],
                            shuffle=True,
                        )

                    self.test_loader[name] = datas.make_dataloader(
                        X_test, Y_test,
                        batch_size=self.config['model_params']['batch_size'],
                        shuffle=False,
                    )
                    logger.info(f"Train loader: {len(self.train_loader[name])} batches | Test loader: {len(self.test_loader[name])} batches")

                    with open(os.path.join(self.paths_bib.model_dir, 'dof_scaler.pkl'), 'wb') as pkl_f:
                        pickle.dump((self.dof_mean, self.dof_std), pkl_f)


    def _pod_context(self):
        """Global POD coefficients used as the encoder context, or None.

        ``latent_params.pod_context`` names the companion file from
        scripts/pod_decompose.py; ``pod_context_coeffs`` picks 'coeffs' (true)
        or 'coeffs_pred' (forecast). The same K numbers are shown to every
        element, so the encoder reads a genuinely global state at a fraction of
        the width of the reassembled DOF tokens it would otherwise build -- and
        one an isolated element cannot infer for itself.
        """
        if getattr(self, '_pod_ctx_cache', None) is not None:
            return self._pod_ctx_cache
        stem = self.config['latent_params'].get('pod_context')
        if not stem:
            return None
        path = self.paths_bib.data_dir + stem + '.h5'
        if not os.path.exists(path):
            logger.warning("pod_context file %s not found; using DOF tokens", path)
            return None
        key = self.config['latent_params'].get('pod_context_coeffs', 'coeffs')
        with h5py.File(path, 'r') as f:
            if key not in f:
                logger.warning("pod_context_coeffs '%s' missing; using 'coeffs'", key)
                key = 'coeffs'
            arr = f[key][:].astype(np.float32)
        mu, sd = arr.mean(0), arr.std(0)
        sd = np.clip(sd, sd.mean() * 1e-6, None)
        self._pod_ctx_cache = (arr - mu) / sd
        logger.info("Encoder context: POD '%s' from %s, %d modes",
                    key, os.path.basename(path), arr.shape[1])
        return self._pod_ctx_cache

    def _rollout_config(self) -> dict:
        """Normalize model_params.rollout. Absent means today's detached behavior.

        mode        'detached' (per-step backward, gradients never cross steps)
                    or 'bptt' (carry the graph across up to `chunk` steps)
        chunk       truncated-BPTT length; 0 means the whole train_ahead rollout
        checkpoint  gradient-checkpoint each model call: ~+33% forward cost for
                    O(1) activation memory in the rollout depth
        ramp_epochs grow backprop depth 1 -> train_ahead over this many epochs
        noise_std   stddev of Gaussian noise added to the fed-back state during
                    training only; 0 (default) is the historical behavior. In
                    normalized DOF units, so ~0.01-0.05 is a small perturbation.
        reassemble  tr_spatial only: apply the inference-time reassembly
                    (_reassemble_tokens) to every predicted frame in the training
                    and validation rollouts, so the loss scores the assembled
                    output and the model is fed the consistent states it sees
                    at inference. False (default) trains on raw element outputs.
        """
        mp = self.config['model_params']
        ro = dict(mp.get('rollout', {}))
        ro.setdefault('mode', 'detached')
        ro.setdefault('chunk', 0)
        ro.setdefault('checkpoint', False)
        ro.setdefault('ramp_epochs', 0)
        ro.setdefault('noise_std', 0.0)
        ro.setdefault('reassemble', False)
        if ro['mode'] not in ('detached', 'bptt'):
            raise ValueError(
                f"rollout.mode '{ro['mode']}' not recognized. Use 'detached' or 'bptt'.")
        if ro['reassemble']:
            if mp['model_type'] != 'tr_spatial':
                raise ValueError("rollout.reassemble is only supported for model_type 'tr_spatial'")
            if not mp.get('reassemble', True):
                raise ValueError("rollout.reassemble trains on assembled states, but "
                                 "model_params.reassemble=false would not assemble them at inference")
        return ro

    def _rollout_adapter(self, epoch: int, train: bool) -> _RolloutAdapter:
        """Build the per-model-type glue for one epoch of _rollout_epoch."""
        model_type = self.config['model_params']['model_type']

        if model_type == 'tr_encdec':
            geom = self._elem_geometry()
            coords = geom['coords']
            # scheduled sampling perturbs the encoder context only while training
            # scheduled sampling refreshes the context from the model's own
            # reassembled DOFs; with a POD context there is nothing to refresh
            ss_p = (self._ss_prob(epoch)
                    if (train and self._pod_context() is None) else 0.0)

            def unpack(batch):
                inputs, targets, tokens, _ = batch
                return (inputs.to(self.device), targets.to(self.device),
                        {'tokens': tokens.to(self.device)})

            def call(inputs, state):
                return self.model(inputs, state['tokens'], coords)

            def after_step(outputs, state):
                if ss_p > 0 and torch.rand(()).item() < ss_p:
                    new_frame = self._reassemble_tokens(outputs.detach(), state['key'], geom)
                    state['tokens'] = self._shift_tokens(state['tokens'], new_frame)

            return _RolloutAdapter(2, unpack, call, after_step)

        if model_type == 'tr_spatial':
            geom = self._elem_geometry()
            coords = geom['coords']
            reassemble = self.rollout_cfg['reassemble']

            def unpack(batch):
                inputs, targets, _, _ = batch
                return inputs.to(self.device), targets.to(self.device), {}

            def call(inputs, state):
                out = self.model(inputs, coords)
                return self._reassemble_tokens(out, state['key'], geom) if reassemble else out

            return _RolloutAdapter(2, unpack, call)

        def unpack(batch):
            inputs, targets = batch
            return inputs.to(self.device), targets.to(self.device), {}

        return _RolloutAdapter(1, unpack, lambda inputs, state: self.model(inputs))

    def _bptt_depth(self, epoch: int, steps: int) -> int:
        """Backprop depth for this epoch, ramping 1 -> steps over ramp_epochs.

        A fresh model rolled out many steps diverges immediately, which makes
        every gradient non-finite and stalls training silently, so the depth is
        grown in over the first epochs when ramp_epochs is set.
        """
        ramp = self.rollout_cfg['ramp_epochs']
        if ramp <= 0:
            return steps
        frac = min((epoch + 1) / ramp, 1.0)
        return max(1, int(round(1 + (steps - 1) * frac)))

    def _rollout_batch(self, batch, adapter, key, train, max_norm, epoch=0):
        """One optimizer step's worth of multi-step rollout.

        With rollout.mode == 'detached' each step is backpropagated on its own
        and the fed-back prediction is detached, so gradients never cross steps.
        With 'bptt' the graph is carried across up to `chunk` steps before a
        single backward, which is what lets the model learn to compensate for
        compounding rollout error.

        Returns (mean per-step loss, pre-clip grad norm or None).
        """
        inputs, targets, state = adapter.unpack(batch)
        state['key'] = key
        t_ax = adapter.time_axis
        steps = targets.shape[t_ax]
        window = inputs.shape[t_ax]

        ro = self.rollout_cfg
        bptt = train and ro['mode'] == 'bptt'
        chunk = min(ro['chunk'] or steps, self._bptt_depth(epoch, steps)) if bptt else 1
        use_ckpt = bptt and ro['checkpoint']

        if train:
            self.optimizer.zero_grad()

        total = 0.0
        chunk_loss = None
        collect = train and self.loss_plan.needs_trajectory
        traj_p, traj_t = [], []
        for n in range(steps):
            target = targets.select(t_ax, n)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                if use_ckpt:
                    outputs = torch.utils.checkpoint.checkpoint(
                        adapter.call, inputs, state, use_reentrant=False)
                else:
                    outputs = adapter.call(inputs, state)
                loss = self.criterion(outputs, target)
            # accumulate as a float: holding the tensor would pin its autograd graph
            total += float(loss.detach())
            if collect:
                traj_p.append(outputs)
                traj_t.append(target)
            self._accumulate_terms()

            did_backward = False
            if train:
                if bptt:
                    chunk_loss = loss if chunk_loss is None else chunk_loss + loss
                    # with trajectory terms the graph must survive to the end of
                    # the rollout, so the single backward is deferred past the loop
                    if not collect and ((n + 1) % chunk == 0 or (n + 1) == steps):
                        self.scaler.scale(chunk_loss).backward()
                        chunk_loss = None
                        did_backward = True
                else:
                    self.scaler.scale(loss).backward()
                    did_backward = True

            # carry the graph forward only while it is still alive; backward()
            # frees it, so the window must be detached at every chunk boundary
            carry = outputs if (not train or (bptt and not did_backward)) else outputs.detach()
            # optional noise on the fed-back state only -- never on the output
            # the loss scores. The spatial rollout is already free-running, so
            # this is not teacher forcing; it teaches the map to contract its
            # own perturbations, which is what extends the stable horizon. The
            # noise carries no gradient, so a live BPTT graph through `carry`
            # is unaffected.
            if train and ro['noise_std'] > 0.0:
                carry = carry + torch.randn_like(carry) * ro['noise_std']
            inputs = torch.cat(
                (inputs.narrow(t_ax, 1, window - 1), carry.unsqueeze(t_ax)), dim=t_ax)
            if bptt and did_backward:
                inputs = inputs.detach()
            if adapter.after_step is not None:
                adapter.after_step(outputs, state)

        if collect and traj_p:
            # (B, ta, F) or (B, E, ta, F) -> always (B, S, ta, F)
            tp = torch.stack(traj_p, dim=t_ax)
            tt = torch.stack(traj_t, dim=t_ax)
            if t_ax == 1:
                tp, tt = tp.unsqueeze(1), tt.unsqueeze(1)
            with torch.autocast(device_type="cuda", enabled=False):
                extra = losses.deferred_loss(self.criterion, tp.float(), tt.float())
            if extra is not None:
                total += float(extra.detach()) * steps
                chunk_loss = extra if chunk_loss is None else chunk_loss + extra
        if train and chunk_loss is not None:
            self.scaler.scale(chunk_loss).backward()

        grad_norm = None
        if train:
            self.scaler.unscale_(self.optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm))
            # a non-finite norm means fp16 overflow; scaler.step then skips the
            # update and halves the scale, so no corrupted step is ever applied
            self.scaler.step(self.optimizer)
            self.scaler.update()
        return total / steps, grad_norm

    def _rollout_epoch(self, epoch: int, train: bool, max_norm: float = None) -> float:
        """One epoch of multi-step rollouts over every loader. Returns the mean loss.

        Shared by all model types and by both training and validation; the
        per-model differences live in the adapter.
        """
        loaders = self.train_loader if train else self.test_loader
        self.model.train(train)
        adapter = self._rollout_adapter(epoch, train)

        epoch_loss = 0.0
        grad_norms = []
        n_steps = 0
        n_skipped = 0
        self._term_totals = None
        self._term_count = 0
        with torch.set_grad_enabled(train):
            for key in loaders:
                if train and self.config['distributed']:
                    self.sampler[key].set_epoch(epoch)
                n_batches = len(loaders[key])
                for batch in loaders[key]:
                    loss, grad_norm = self._rollout_batch(
                        batch, adapter, key, train, max_norm, epoch)
                    epoch_loss += loss / n_batches
                    if grad_norm is not None:
                        n_steps += 1
                        # inf/nan means fp16 overflow and a skipped optimizer step;
                        # averaging it in would poison the whole epoch's reported norm
                        if np.isfinite(grad_norm):
                            grad_norms.append(grad_norm)
                        else:
                            n_skipped += 1

        self.last_terms = self._mean_terms()
        if train:
            self.last_grad_norm = float(np.mean(grad_norms)) if grad_norms else float('nan')
            self.last_skipped = n_skipped
            self.last_steps = n_steps
            if n_steps and n_skipped > n_steps // 2:
                logger.warning(
                    "Epoch %d: GradScaler skipped %d/%d optimizer steps (fp16 overflow). "
                    "Training is barely progressing; lower the learning rate or the "
                    "rollout depth.", epoch + 1, n_skipped, n_steps)
        return epoch_loss

    def _accumulate_terms(self):
        """Sum each combo-loss term's contribution on device (no host sync)."""
        terms = getattr(self.criterion, 'last_terms', None)
        if terms is None:
            return
        if self._term_totals is None:
            self._term_totals = [t.clone() for t in terms]
        else:
            for total, term in zip(self._term_totals, terms):
                total += term
        self._term_count += 1

    def _mean_terms(self):
        """Per-term mean contributions for the epoch, as {name: float}."""
        if not self._term_totals or not self._term_count:
            return None
        names = getattr(self.criterion, 'names', None) or [
            f"term{i}" for i in range(len(self._term_totals))]
        return {n: float(t) / self._term_count for n, t in zip(names, self._term_totals)}

    def _train_epoch(self, epoch: int, max_norm: float) -> float:
        """Single forward+backward pass over all training loaders. Returns mean epoch loss."""
        return self._rollout_epoch(epoch, train=True, max_norm=max_norm)

    def _validate_epoch(self) -> float:
        """Single forward pass over all test loaders. Returns mean test loss."""
        return self._rollout_epoch(0, train=False)

    ## ======================= Global-local encoder-decoder (tr_encdec) ==========================

    def _elem_geometry(self, use_skip=True):
        """Precompute and cache element geometry for the element-token models (encdec, spatial).

        use_skip=True follows the training element subsampling (skipx/y/z);
        use_skip=False uses all elements (prediction/rollout, matching baseline behavior).
        """
        if use_skip:
            skipx = self.config['latent_params'].get('skipx', 1)
            skipy = self.config['latent_params'].get('skipy', 1)
            skipz = self.config['latent_params'].get('skipz', 1) if self.dim == 3 else 1
        else:
            skipx = skipy = skipz = 1
        key = (skipx, skipy, skipz)
        if not hasattr(self, '_geom_cache'):
            self._geom_cache = {}
        if key in self._geom_cache:
            return self._geom_cache[key]

        lltogl_mat = self._compute_lltogl_mat(skipx=skipx, skipy=skipy, skipz=skipz)
        coords = self._compute_elem_centroids(skipx=skipx, skipy=skipy, skipz=skipz)
        num_dofs = self.l_config.num_gfem_nodes * self.l_config.dof_node
        flat_index = lltogl_mat.reshape(-1)
        # nodes not covered by any (skipped) element keep count 1 to avoid divide-by-zero
        counts = np.bincount(flat_index, minlength=num_dofs).clip(min=1)
        geom = {
            'lltogl_np': lltogl_mat,
            'lltogl': torch.from_numpy(lltogl_mat).long().to(self.device),
            'coords': torch.from_numpy(coords).float().to(self.device),
            'scatter_index': torch.from_numpy(flat_index).long().to(self.device),
            'scatter_index_cpu': torch.from_numpy(flat_index).long(),
            'counts': torch.from_numpy(counts).float().to(self.device),
            'counts_cpu': torch.from_numpy(counts).float(),
            'num_dofs': num_dofs,
        }
        self._geom_cache[key] = geom
        logger.info(f"element geometry (skip={key}): {lltogl_mat.shape[0]} elements, {num_dofs} nodal DOFs")
        return geom

    def _compute_elem_centroids(self, skipx=1, skipy=1, skipz=1):
        """Element centroid coordinates, min-max normalized to [0,1].

        Loop order must mirror _compute_lltogl_mat so coords align with lltogl rows.
        GFEM node (i, j[, k]) sits at grid index (i*nskip, j*nskip[, k*nskip]).
        """
        nx = self.l_config.nx_g
        ny = self.l_config.ny_g
        nskip = self.l_config.nskip
        coords = []
        if self.dim == 3:
            nz = self.l_config.nz_g
            for kx in range(0, nx - 1, skipx):
                for ky in range(0, ny - 1, skipy):
                    for kz in range(0, nz - 1, skipz):
                        lo = (kx * nskip, ky * nskip, kz * nskip)
                        hi = ((kx + 1) * nskip, (ky + 1) * nskip, (kz + 1) * nskip)
                        coords.append([
                            0.5 * (self.x_grid[lo] + self.x_grid[hi]),
                            0.5 * (self.y_grid[lo] + self.y_grid[hi]),
                            0.5 * (self.z_grid[lo] + self.z_grid[hi]),
                        ])
        else:
            for kx in range(0, nx - 1, skipx):
                for ky in range(0, ny - 1, skipy):
                    lo = (kx * nskip, ky * nskip)
                    hi = ((kx + 1) * nskip, (ky + 1) * nskip)
                    coords.append([
                        0.5 * (self.x_grid[lo] + self.x_grid[hi]),
                        0.5 * (self.y_grid[lo] + self.y_grid[hi]),
                    ])
        coords = np.asarray(coords, dtype=np.float64)
        cmin, cmax = coords.min(axis=0), coords.max(axis=0)
        return (coords - cmin) / np.where(cmax > cmin, cmax - cmin, 1.0)

    # dense assembled inverse up to this many nodal DOFs per component
    MAX_DENSE_ASSEMBLY = 10000

    def _assembly_ops(self, geom):
        """Regularized Galerkin assembly operators for `geom`, built once.

        Per velocity component, M = B^T diag(Wt) B is the element mass matrix (Wt
        splits shared boundary points so element energies sum to the field
        energy) and Mr = M + lam * tr(M)/n * I. Assembly solves

            (sum_e G_e^T Mr G_e) d = sum_e G_e^T Mr d_hat_e,

        so Mr and the inverse of the assembled matrix are all a step needs. Kept
        in float64 (numpy, final assembly) and float32 (torch, rollout steps).
        """
        if 'assembly' in geom:
            return geom['assembly']
        lam = float(self.config['model_params']['reassemble_lambda'])
        cfg = self.l_config
        if self.dim == 3:
            Wt = self.dls._build_wt_vec(cfg.nskip)
            mats = [cfg.modemat_local_u, cfg.modemat_local_v, cfg.modemat_local_w]
        else:
            Wt = self.dls._build_wt_vec(*self.dls.config_nskip(cfg))
            mats = [cfg.modemat_local_u, cfg.modemat_local_v]
        L = geom['lltogl_np']
        E, n = L.shape
        N = geom['num_dofs']
        if N > self.MAX_DENSE_ASSEMBLY:
            raise NotImplementedError(
                f"reassemble_lambda needs a dense {N}x{N} inverse; too large, use the default average")
        ops = []
        for B in mats:
            B = np.asarray(B, dtype=np.float64)
            M = (B * Wt[:, None]).T @ B
            Mr = M + lam * np.trace(M) / n * np.eye(n)
            A = np.zeros((N, N))
            for e in range(E):
                A[np.ix_(L[e], L[e])] += Mr
            # DOFs no element covers (skipped elements) are never gathered; pin them
            free = np.diag(A) == 0
            A[free, free] = 1.0
            Ainv = np.linalg.pinv(A, hermitian=True) if lam == 0 else np.linalg.inv(A)
            ops.append({'Mr': Mr, 'Ainv': Ainv,
                        'Mr_t': torch.from_numpy(Mr).float().to(self.device),
                        'Ainv_t': torch.from_numpy(Ainv).float().to(self.device)})
        logger.info(f"Regularized Galerkin assembly: lambda={lam:g}, {E} elements, {N} nodal DOFs "
                    f"per component, cond(A)={np.linalg.cond(A):.2e}")
        geom['assembly'] = ops
        return ops

    def _reassemble_tokens(self, pred_norm, name, geom):
        """Project normalized element predictions to a consistent global field and back.

        (G, E, F) -> assemble the shared nodal DOFs -> regather -> (G, E, F).
        `model_params.reassemble_lambda` picks the projection. None (default)
        averages each node's copies: least squares in raw coefficients. A value
        lam solves the regularized Galerkin system (_assembly_ops): least squares
        in field energy, with a ridge against the codec's near-null space.
        lam -> inf recovers the average; lam = 0 is the pure Galerkin (L2)
        projection, which injects near-null coefficients and destabilizes
        rollouts on these ill-conditioned bases. fp32, differentiable.
        """
        G = pred_norm.shape[0]
        mean = self.dof_mean[name].to(pred_norm.device)
        std = self.dof_std[name].to(pred_norm.device)
        lam = self.config['model_params'].get('reassemble_lambda')
        ops = None if lam is None else self._assembly_ops(geom)
        ndof = self.l_config.dof_elem
        comps = []
        # the Galerkin solve must not drop to fp16 inside the training autocast
        with torch.autocast(device_type=pred_norm.device.type, enabled=False):
            pred = pred_norm.float() * std + mean
            for c in range(self.dim):
                local = pred[:, :, c * ndof:(c + 1) * ndof]
                flat = torch.zeros(G, geom['num_dofs'], device=pred.device)
                if ops is None:
                    flat = flat.index_add(1, geom['scatter_index'], local.reshape(G, -1))
                    flat = flat / geom['counts']
                else:
                    flat = flat.index_add(1, geom['scatter_index'], (local @ ops[c]['Mr_t']).reshape(G, -1))
                    flat = flat @ ops[c]['Ainv_t']
                comps.append(flat[:, geom['lltogl']])
            out = torch.cat(comps, dim=2)
            return (out - mean) / std

    def _ss_prob(self, epoch: int) -> float:
        """Scheduled-sampling probability of refreshing context from model predictions."""
        mp = self.config['model_params']
        start = mp.get('ss_start_epoch', None)
        if start is None or epoch < start:
            return 0.0
        ramp = max(mp.get('ss_ramp_epochs', 1), 1)
        return mp.get('ss_prob_max', 0.5) * min((epoch - start) / ramp, 1.0)

    def _shift_tokens(self, tokens, new_frame):
        """Append a reassembled frame to the context token window (G, E, cw*F)."""
        cw = self.config['model_params'].get('context_window', 1)
        if cw == 1:
            return new_frame
        G, E, F = new_frame.shape
        tok_view = tokens.reshape(G, E, cw, F)
        return torch.cat((tok_view[:, :, 1:], new_frame.unsqueeze(2)), dim=2).reshape(G, E, cw * F)

    def _get_train_data_grouped(self):
        """Build snapshot-grouped train/test loaders for the encdec and spatial models."""
        if self.model is None:
            return
        model_type = self.config['model_params']['model_type']
        logger.info(f'Getting training and test data (snapshot-grouped, {model_type})')
        if not self.config['latent_params'].get('localized', False):
            raise ValueError(f"model_type '{model_type}' requires latent_params.localized = true")
        mp = self.config['model_params']
        tl = mp['time_lag']
        ta = mp['train_ahead']
        cw = mp.get('context_window', 1)
        ct = mp.get('context_time', 'window_end')

        geom = self._elem_geometry()
        lltogl_cpu = torch.from_numpy(geom['lltogl_np']).long()

        self.dof_mean = {}
        self.dof_std = {}
        self.train_loader = {}
        self.test_loader = {}
        self.sampler = {}

        with h5py.File(self.paths_bib.latent_path, 'r') as f:
            for source in self.config['train_data']:
                name = source.get('name')
                train_indices = self.indices['train_data'][name]['train_indices']
                test_indices = self.indices['train_data'][name]['test_indices']

                dof_u = f[name]['dof_u']
                dof_v = f[name]['dof_v']
                dof_w = f[name]['dof_w'] if self.dim == 3 else None

                dofs = self._load_dof_rows(dof_u, dof_v, dof_w, train_indices, geom['lltogl_np'])
                self.dof_mean[name], self.dof_std[name] = self._dof_stats(dofs)
                logger.info(f"Normalization '{self.config['model_params'].get('norm','scalar')}': "
                            f"mean/std shapes {tuple(self.dof_mean[name].shape)}, "
                            f"{tuple(self.dof_std[name].shape)}")
                del dofs

                dof_comps = [torch.from_numpy(dof_u[:]).float(), torch.from_numpy(dof_v[:]).float()]
                if self.dim == 3:
                    dof_comps.append(torch.from_numpy(dof_w[:]).float())

                pod_ctx = self._pod_context()
                train_set = datas.LocalGlobalDataset(
                    dof_comps, lltogl_cpu, train_indices, tl, ta,
                    self.dof_mean[name], self.dof_std[name], cw, ct,
                    pod_context=pod_ctx)
                test_set = datas.LocalGlobalDataset(
                    dof_comps, lltogl_cpu, test_indices, tl, ta,
                    self.dof_mean[name], self.dof_std[name], cw, ct,
                    pod_context=pod_ctx)

                if self.config['distributed']:
                    self.train_loader[name], self.sampler[name] = datas.make_group_dataloader(
                        train_set, batch_size=mp['batch_size'], shuffle=True, distributed=True)
                else:
                    self.train_loader[name] = datas.make_group_dataloader(
                        train_set, batch_size=mp['batch_size'], shuffle=True)
                self.test_loader[name] = datas.make_group_dataloader(
                    test_set, batch_size=mp['batch_size'], shuffle=False)
                logger.info(f"Data loaded for {name}: {len(train_set)} train / {len(test_set)} test snapshot groups "
                            f"of {geom['lltogl_np'].shape[0]} elements")
                logger.info(f"Train loader: {len(self.train_loader[name])} batches | Test loader: {len(self.test_loader[name])} batches")

        with open(os.path.join(self.paths_bib.model_dir, 'dof_scaler.pkl'), 'wb') as fs:
            pickle.dump((self.dof_mean, self.dof_std), fs)

    def _epoch_header(self, terms):
        """Header rows for the compact training table."""
        cols = [f"{'Epoch':>10}", f"{'Train':>9}", f"{'Test':>9}", 'B',
                f"{'Patience':>7}", 'C', f"{'LR':>7}", f"{'GrdNrm':>7}", 'Skp']
        if terms:
            cols += [f"{losses.abbrev(n):>8}" for n in terms]
        cols.append(f"{'Time':>6}")
        head = '| ' + ' | '.join(cols) + ' |'
        return ['-' * len(head), head, '-' * len(head)]

    def _epoch_row(self, epoch, train_loss, test_loss, best_flag, patience,
                   ckpt_flag, lr, terms, minutes):
        """One row of the compact training table (see _epoch_header)."""
        n_ep = self.config['model_params']['num_epochs']
        cols = [f"{epoch+1:>4}/{n_ep:<5}", f"{train_loss:>9.3e}", f"{test_loss:>9.3e}",
                best_flag,
                f"{patience:>3}/{self.config['model_params']['patience']:<3}", ckpt_flag,
                f"{lr:>7.1e}", f"{self.last_grad_norm:>7.1e}",
                f"{self.last_skipped:>3}" if self.last_skipped else '  -']
        if terms:
            cols += [f"{v:>8.2e}" for v in terms.values()]
        cols.append(f"{minutes:>5.1f}m")
        return '| ' + ' | '.join(cols) + ' |'

    def _model_fit(self):
        if self.model is not None:
            if self.checkpointed:
                best_epoch = self.epoch
                losses = self.losses
                test_losses = self.test_losses
                best_model = copy.deepcopy(self.model.state_dict())
                early_stop_counter = self.early_stop_counter
                best_test_loss = min(test_losses)
            else:
                losses = []
                test_losses = []
                best_model = None
                early_stop_counter = 0
                best_test_loss = float('inf')
                best_epoch = 0

            start_time = time.time()
            # ReduceLROnPlateau.get_last_lr only exists from torch ~2.2; read the optimizer directly
            new_lr = self.optimizer.param_groups[0]['lr']

            mp = self.config['model_params']
            max_norm = mp.get('grad_clip', GRAD_CLIP_MAX_NORM)
            min_delta = mp.get('es_min_delta', ES_MIN_DELTA)
            lr_floor_patience = mp.get('lr_floor_patience', LR_FLOOR_PATIENCE)
            lr_min = mp.get('lr_min', 1e-6)
            patience_ref = best_test_loss
            logger.info(f"Gradient clip max norm: {max_norm}")
            logger.info(f"Early stopping: patience={mp['patience']}, min_delta={min_delta:.2%}"
                        f", lr_floor_patience={lr_floor_patience}")

            first_epoch = len(losses)
            for epoch in range(first_epoch, self.config['model_params']['num_epochs']):
                epoch_loss = self._train_epoch(epoch, max_norm)
                losses.append(epoch_loss)
                train_terms = self.last_terms

                test_loss = self._validate_epoch()
                test_losses.append(test_loss)

                self.scheduler.step(test_loss)
                new_lr = self.optimizer.param_groups[0]['lr']

                if epoch > 1:
                    if not (np.isfinite(test_losses[-1]) and np.isfinite(losses[-1])):
                        logger.info(f'Non-finite loss at epoch {epoch+1} '
                                    f'(train {losses[-1]}, test {test_losses[-1]}). Stopping training.')
                        self.model.load_state_dict(best_model)
                        break

                    # Always keep the genuinely best weights...
                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        best_model = copy.deepcopy(self.model.state_dict())
                        best_epoch = epoch + 1

                    # ...but only a materially better loss resets patience. On a
                    # noisy plateau the run-to-run spread can exceed the total
                    # remaining improvement, so bare `<` keeps finding new minima
                    # and early stopping never fires.
                    if test_loss < patience_ref * (1.0 - min_delta):
                        patience_ref = test_loss
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                        # once the scheduler has bottomed out there is nothing
                        # left to wait for, so allow a shorter fuse there
                        at_lr_floor = new_lr <= lr_min * 1.001
                        eff_patience = (lr_floor_patience
                                        if (at_lr_floor and lr_floor_patience is not None)
                                        else self.config['model_params']['patience'])
                        if early_stop_counter >= eff_patience:
                            reason = 'at LR floor' if (at_lr_floor and lr_floor_patience is not None) else ''
                            logger.info(f'Early stopping at epoch {epoch+1} {reason}'.rstrip())
                            self.model.load_state_dict(best_model)
                            logger.info(f'Best model loaded from epoch {best_epoch}, with test loss: {best_test_loss:.6e}')
                            break

                    if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'losses': losses,
                            'test_losses': test_losses,
                            'early_stop_counter': early_stop_counter,
                            'best_model': best_model
                        }, self.paths_bib.checkpoint_path)

                best_flag = 'X' if (epoch + 1) == best_epoch else ' '
                checkpoint_flag = 'X' if (epoch + 1) % CHECKPOINT_INTERVAL == 0 else ' '
                # pre-clip gradient norm: if this sits far above max_norm the clip is
                # setting the step size, and loss changes cannot be compared meaningfully
                if (epoch - first_epoch) % EPOCH_HEADER_INTERVAL == 0:
                    for line in self._epoch_header(train_terms):
                        logger.info(line)
                logger.info(self._epoch_row(
                    epoch, losses[-1], test_losses[-1], best_flag, early_stop_counter,
                    checkpoint_flag, new_lr, train_terms, (time.time() - start_time) / 60))

            end_time = time.time()
            logger.info('\n\nTime taken for training: %s minutes', (end_time - start_time) / 60)
            logger.info('Time taken per epoch: %s minutes', (end_time - start_time) / 60 / len(losses))

            torch.save(self.model.state_dict(), self.paths_bib.model_path)
            logger.info(f"Final model saved to {self.paths_bib.model_path}")
            with open(self.paths_bib.model_dir + 'losses.pkl', 'wb') as f:
                pickle.dump({'train_losses': losses, 'test_losses': test_losses}, f)
            logger.info(f"Training and test losses saved to {self.paths_bib.model_dir + 'losses.pkl'}")
            logger.info('\nTraining complete')

        else:
            logger.info("No training required for f_extrap model")
            self.pred()

    def _compute_lltogl_mat(self, skipx=1, skipy=1, skipz=1):
        nx = self.l_config.nx_g
        ny = self.l_config.ny_g
        nz = self.l_config.nz_g if self.dim == 3 else 1
        dof_node = self.l_config.dof_node
        # one element per loop iteration below; (nx//skipx)*(ny//skipy) overcounts when skip==1,
        # leaving all-zero rows that alias every slot to global DOF 0
        num_unique_elems = len(range(0, nx - 1, skipx)) * len(range(0, ny - 1, skipy))
        if self.dim == 3:
            num_unique_elems *= len(range(0, nz - 1, skipz))
        dof_elem = self.l_config.dof_elem
        lltogl_mat = np.zeros((num_unique_elems, dof_elem), dtype=int)
        elem_idx = 0
        if self.dim == 3:
            for kx in range(0,nx-1,skipx):
                for ky in range(0,ny-1,skipy):
                    for kz in range(0,nz-1,skipz):
                        lltogl_mat[elem_idx] = self.dls.build_lltogl(kx, ky, kz, ny, nz, dof_node, self.dls.node_map())
                        elem_idx += 1
        else:
            for kx in range(0,nx-1, skipx):
                for ky in range(0,ny-1, skipy):
                    lltogl_mat[elem_idx] = self.dls.build_lltogl(kx, ky, ny, dof_node, self.dls.node_map())
                    elem_idx += 1
        # logger.info(f"Precomputed lltogl_mat shape: {lltogl_mat.shape}")
        return lltogl_mat

    def _predict_dofs_forward(self, name, dof_u, dof_v, dof_w=None, total_steps=None,
                              ids=None):
        """
        Predict dof_u/dof_v(/dof_w) forward in time from initial conditions.
        Inputs dof_u/dof_v(/dof_w) are expected to contain at least `time_lag` snapshots.
        Returns full predicted trajectories with shape (total_steps, num_dofs).
        """
        if self.config['model_params']['model_type'] == 'tr_encdec':
            return self._predict_dofs_forward_encdec(name, dof_u, dof_v, dof_w,
                                                     total_steps, ids)
        if self.config['model_params']['model_type'] == 'tr_spatial':
            return self._predict_dofs_forward_spatial(name, dof_u, dof_v, dof_w, total_steps)
        time_lag = self.config['model_params']['time_lag']
        if total_steps is None:
            total_steps = dof_u.shape[0]
        if total_steps < time_lag:
            raise ValueError(f"total_steps ({total_steps}) must be >= time_lag ({time_lag})")
        if dof_u.shape[0] < time_lag or dof_v.shape[0] < time_lag:
            raise ValueError("dof_u and dof_v must contain at least time_lag snapshots")
        if self.dim == 3 and (dof_w is None or dof_w.shape[0] < time_lag):
            raise ValueError("dof_w must contain at least time_lag snapshots for 3D predictions")

        init_u = dof_u[:time_lag, :]
        init_v = dof_v[:time_lag, :]
        init_w = dof_w[:time_lag, :] if self.dim == 3 else None
        if self.config['latent_params'].get('localized', False):
            lltogl_mat = self._compute_lltogl_mat()
            num_elems = lltogl_mat.shape[0]
        else:
            lltogl_mat = None
            num_elems = 1

        if self.dim == 3:
            u_sel = init_u[:, lltogl_mat] # shape: (time_lag, num_elems, dof_elem)
            v_sel = init_v[:, lltogl_mat]
            w_sel = init_w[:, lltogl_mat]
            dof_input = np.transpose(np.concatenate([u_sel, v_sel, w_sel], axis=2), (1, 0, 2)) # shape: (num_elems, time_lag, dim * dof_elem)
        else:
            u_sel = init_u[:, lltogl_mat]
            v_sel = init_v[:, lltogl_mat]
            dof_input = np.transpose(np.concatenate([u_sel, v_sel], axis=2), (1, 0, 2))

        dof_input = (dof_input - self.dof_mean[name].numpy()) / self.dof_std[name].numpy()
        dof_input = torch.from_numpy(dof_input).float().to(self.device)

        self.model.eval()
        predictions = np.zeros((num_elems, total_steps, self.config['model_params']['input_dim']), dtype=np.float64)
        predictions[:, :time_lag, :] = dof_input.cpu().numpy()

        with torch.no_grad():
            for t in range(total_steps - time_lag):
                output = self.model(dof_input)
                predictions[:, t + time_lag, :] = output.cpu().numpy()
                dof_input = torch.cat((dof_input[:, 1:, :], output.unsqueeze(1)), dim=1)

        predictions = predictions * self.dof_std[name].numpy() + self.dof_mean[name].numpy()

        num_dofs = self.l_config.num_gfem_nodes * self.l_config.dof_node

        if self.config['latent_params'].get('localized', False):
            ndof = self.l_config.dof_elem
            flat_index = torch.from_numpy(lltogl_mat.reshape(-1)).long()
            counts = torch.from_numpy(
                np.bincount(flat_index.numpy(), minlength=num_dofs).astype(np.float64)
            )
            counts = counts.clamp_min(1.0)

            predictions_t = torch.from_numpy(predictions).double()
            dof_u_pred = torch.zeros((total_steps, num_dofs), dtype=torch.float64)
            dof_v_pred = torch.zeros((total_steps, num_dofs), dtype=torch.float64)
            dof_w_pred = torch.zeros((total_steps, num_dofs), dtype=torch.float64) if self.dim == 3 else None

            for c in range(self.dim):
                vals = predictions_t[:, :, c * ndof:(c + 1) * ndof]
                vals = vals.permute(1, 0, 2).reshape(total_steps, -1)
                acc = torch.zeros((total_steps, num_dofs), dtype=torch.float64)
                acc.index_add_(1, flat_index, vals)
                out = acc / counts
                if c == 0:
                    dof_u_pred = out
                elif c == 1:
                    dof_v_pred = out
                else:
                    dof_w_pred = out

            dof_u_pred = dof_u_pred.numpy()
            dof_v_pred = dof_v_pred.numpy()
            dof_w_pred = dof_w_pred.numpy() if self.dim == 3 else None
        else:
            dof_u_pred = predictions[:, :, :num_dofs].squeeze()
            dof_v_pred = predictions[:, :, num_dofs:2 * num_dofs].squeeze()
            dof_w_pred = predictions[:, :, 2 * num_dofs:].squeeze() if self.dim == 3 else None

        return dof_u_pred, dof_v_pred, dof_w_pred

    def _predict_dofs_forward_encdec(self, name, dof_u, dof_v, dof_w=None,
                                     total_steps=None, ids=None):
        """Two-timescale autoregressive rollout for the encdec model.

        Local dynamics advance every step; the global context is re-encoded from the
        scatter-mean reassembled predicted field every `global_K` steps.
        Returns (total_steps, num_dofs) trajectories like _predict_dofs_forward.
        """
        mp = self.config['model_params']
        time_lag = mp['time_lag']
        cw = mp.get('context_window', 1)
        ct = mp.get('context_time', 'window_end')
        global_K = mp.get('global_K', 10)
        if total_steps is None:
            total_steps = dof_u.shape[0]
        if total_steps < time_lag:
            raise ValueError(f"total_steps ({total_steps}) must be >= time_lag ({time_lag})")
        if dof_u.shape[0] < time_lag or dof_v.shape[0] < time_lag:
            raise ValueError("dof_u and dof_v must contain at least time_lag snapshots")
        if self.dim == 3 and (dof_w is None or dof_w.shape[0] < time_lag):
            raise ValueError("dof_w must contain at least time_lag snapshots for 3D predictions")

        geom = self._elem_geometry(use_skip=False)
        lltogl_mat = geom['lltogl_np']
        num_elems = lltogl_mat.shape[0]
        coords = geom['coords']
        mean_np = self.dof_mean[name].numpy()
        std_np = self.dof_std[name].numpy()

        comps = [dof_u[:time_lag, :][:, lltogl_mat], dof_v[:time_lag, :][:, lltogl_mat]]
        if self.dim == 3:
            comps.append(dof_w[:time_lag, :][:, lltogl_mat])
        init = (np.concatenate(comps, axis=2) - mean_np) / std_np  # (tl, E, F)
        dof_input = torch.from_numpy(init).float().permute(1, 0, 2).unsqueeze(0).to(self.device)  # (1, E, tl, F)

        pod_ctx = self._pod_context()
        if pod_ctx is not None and ids is None:
            raise ValueError("a POD encoder context needs the absolute snapshot ids "
                             "of the prediction window")

        def pod_tokens(step):
            """Context tokens at rollout position `step` (0 = first seeded frame)."""
            end = step + (time_lag if ct == 'window_end' else cw)
            rows = np.asarray(ids)[max(0, end - cw):end]
            ctx = torch.from_numpy(pod_ctx[rows].reshape(-1)).float().to(self.device)
            return ctx.view(1, 1, -1).expand(1, num_elems, ctx.numel())

        def window_tokens(window):
            frames = window[:, :, -cw:, :] if ct == 'window_end' else window[:, :, :cw, :]
            return frames.reshape(1, num_elems, cw * frames.shape[-1])

        self.model.eval()
        model = self.model.module if hasattr(self.model, 'module') else self.model
        F_dim = self.dim * self.l_config.dof_elem
        predictions = np.zeros((num_elems, total_steps, F_dim))
        predictions[:, :time_lag, :] = dof_input[0].cpu().numpy()

        with torch.no_grad():
            context = model.encode(
                pod_tokens(0) if pod_ctx is not None else window_tokens(dof_input), coords)
            for t in range(total_steps - time_lag):
                output = model.decode(dof_input, context)  # (1, E, F)
                predictions[:, t + time_lag, :] = output[0].cpu().numpy()
                dof_input = torch.cat((dof_input[:, :, 1:, :], output.unsqueeze(2)), dim=2)
                if (t + 1) % global_K == 0 and (t + 1) < total_steps - time_lag:
                    if pod_ctx is not None:
                        # the global branch supplies its own forecast; nothing is
                        # reassembled from the local predictions
                        context = model.encode(pod_tokens(t + 1), coords)
                    else:
                        frames = dof_input[:, :, -cw:, :] if ct == 'window_end' else dof_input[:, :, :cw, :]
                        re = torch.stack(
                            [self._reassemble_tokens(frames[:, :, i, :], name, geom) for i in range(cw)],
                            dim=2)
                        context = model.encode(re.reshape(1, num_elems, cw * F_dim), coords)

        predictions = predictions * std_np + mean_np
        return self._assemble_dof_preds(predictions, geom)

    def _assemble_dof_preds(self, predictions, geom):
        """Assemble element-local predictions (E, T, dim*dof_elem) onto the shared nodal DOFs.

        Interior nodes belong to multiple elements; their duplicate slots are
        combined with the same projection as _reassemble_tokens (averaged by
        default, regularized Galerkin with model_params.reassemble_lambda).
        Returns (dof_u, dof_v, dof_w) each of shape (T, num_dofs); dof_w is None in 2D.
        """
        total_steps = predictions.shape[1]
        num_dofs = geom['num_dofs']
        ndof = self.l_config.dof_elem
        counts = geom['counts_cpu'].numpy()
        ops = (None if self.config['model_params'].get('reassemble_lambda') is None
               else self._assembly_ops(geom))
        out = []
        for c in range(self.dim):
            acc = torch.zeros(total_steps, num_dofs, dtype=torch.float64)
            local = predictions[:, :, c * ndof:(c + 1) * ndof]
            if ops is not None:
                local = local @ ops[c]['Mr']
            vals = torch.from_numpy(np.ascontiguousarray(local.transpose(1, 0, 2)).reshape(total_steps, -1))
            acc.index_add_(1, geom['scatter_index_cpu'], vals)
            out.append(acc.numpy() / counts if ops is None else acc.numpy() @ ops[c]['Ainv'])
        dof_w_pred = out[2] if self.dim == 3 else None
        return out[0], out[1], dof_w_pred

    def _predict_dofs_forward_spatial(self, name, dof_u, dof_v, dof_w=None, total_steps=None):
        """Autoregressive rollout for the spatial-attention model.

        Every step attends across all element tokens and emits the next snapshot. With
        `reassemble` enabled (default), each predicted frame is projected onto the shared
        nodal DOFs and regathered before being fed back, so elements stay consistent at
        the nodes they share instead of drifting apart over the rollout.
        """
        mp = self.config['model_params']
        time_lag = mp['time_lag']
        if total_steps is None:
            total_steps = dof_u.shape[0]
        if total_steps < time_lag:
            raise ValueError(f"total_steps ({total_steps}) must be >= time_lag ({time_lag})")
        if dof_u.shape[0] < time_lag or dof_v.shape[0] < time_lag:
            raise ValueError("dof_u and dof_v must contain at least time_lag snapshots")
        if self.dim == 3 and (dof_w is None or dof_w.shape[0] < time_lag):
            raise ValueError("dof_w must contain at least time_lag snapshots for 3D predictions")

        geom = self._elem_geometry(use_skip=False)
        lltogl_mat = geom['lltogl_np']
        num_elems = lltogl_mat.shape[0]
        coords = geom['coords']
        mean_np = self.dof_mean[name].numpy()
        std_np = self.dof_std[name].numpy()
        reassemble = mp.get('reassemble', True)

        comps = [dof_u[:time_lag, :][:, lltogl_mat], dof_v[:time_lag, :][:, lltogl_mat]]
        if self.dim == 3:
            comps.append(dof_w[:time_lag, :][:, lltogl_mat])
        init = (np.concatenate(comps, axis=2) - mean_np) / std_np  # (tl, E, F)
        dof_input = torch.from_numpy(init).float().permute(1, 0, 2).unsqueeze(0).to(self.device)  # (1, E, tl, F)

        self.model.eval()
        model = self.model.module if hasattr(self.model, 'module') else self.model
        F_dim = self.dim * self.l_config.dof_elem
        predictions = np.zeros((num_elems, total_steps, F_dim))
        predictions[:, :time_lag, :] = dof_input[0].cpu().numpy()

        with torch.no_grad():
            for t in range(total_steps - time_lag):
                output = model(dof_input, coords)  # (1, E, F)
                if reassemble:
                    output = self._reassemble_tokens(output, name, geom)
                predictions[:, t + time_lag, :] = output[0].cpu().numpy()
                dof_input = torch.cat((dof_input[:, :, 1:, :], output.unsqueeze(2)), dim=2)

        predictions = predictions * std_np + mean_np
        return self._assemble_dof_preds(predictions, geom)


    def _pred_sources(self):
        """(source, split_key) pairs that pred() and eval() cover.

        Top-level `eval_train_split: false` skips the in-sample rollout over the
        training block. On the 512x512 cavity it writes ~4 GB of reconstructed
        fields per run (~7 GB at Re 30k), most of a run's disk use, for in-sample
        numbers only; the held-out validation block is always covered.
        """
        out = []
        if self.config.get('train_data') is not None and self.config.get('eval_train_split', True):
            out.extend((source, 'train_indices') for source in self.config['train_data'])
        if self.config.get('eval_data') is not None:
            out.extend((source, 'val_indices') for source in self.config['eval_data'])
        return out

    def pred(self) -> None:
        """Make predictions over the validation data for all sets in config['eval_data']."""
        logger.info(f"{'#'*20}\t{'Predicting...':<20}\t{'#'*20}")
        pred_sources = self._pred_sources()
            

        for id, (source, split_key) in enumerate(pred_sources):
            name = source.get('name')
            source_group = 'train_data' if split_key == 'train_indices' else 'eval_data'
            val_id = self.indices[source_group][name][split_key]
            time_lag = self.config['model_params']['time_lag']
            init_id = val_id[:time_lag]
            path = source.get('path')
            path = self.paths_bib.data_dir + path + '.h5'
            pred_path = self.paths_bib.pred_dir + name + f'_{source_group}_pred.h5'


            logger.info(f"Predicting for {name} from {path}...")
            with h5py.File(self.paths_bib.latent_path, 'r') as f:
                dof_u = f[name]['dof_u'][init_id, :]
                dof_v = f[name]['dof_v'][init_id, :]
                dof_w = f[name]['dof_w'][init_id, :] if self.dim == 3 else None
            logger.info(f"Predicting for {name} {source_group}")

            # long forward prediction of dofs
            dof_u_pred, dof_v_pred, dof_w_pred = self._predict_dofs_forward(
                ids=val_id,
                name=name,
                dof_u=dof_u,
                dof_v=dof_v,
                dof_w=dof_w,
                total_steps=len(val_id)
            )

            # Save predictions to HDF5
            with h5py.File(pred_path, 'w') as f:
                f.create_dataset('dof_u', data=dof_u_pred)
                f.create_dataset('dof_v', data=dof_v_pred)
                if self.dim == 3:
                    f.create_dataset('dof_w', data=dof_w_pred)
            logger.info(f"Predictions saved to {pred_path}")
            
            # Many short horizon predictions for error growth analysis
            horizon = self.config['model_params'].get('horizon', 30)
            max_horizons = MAX_ERROR_HORIZONS
            num_horizons = len(val_id) - time_lag - horizon + 1
            num_horizons = min(num_horizons, max_horizons)

            dof_true_horizons = np.zeros((num_horizons, horizon, dof_u.shape[1]*self.dim))  # for error growth metrics
            dof_pred_horizon = np.zeros((num_horizons, horizon, dof_u.shape[1]*self.dim))  # for error growth metrics

            with h5py.File(pred_path, 'r') as f,  h5py.File(self.paths_bib.latent_path, 'r') as f_latent:
                for i in range(num_horizons):
                    start_id = val_id[i:i+time_lag]
                    dof_u_init = f_latent[name]['dof_u'][start_id, :]
                    dof_v_init = f_latent[name]['dof_v'][start_id, :]
                    dof_w_init = f_latent[name]['dof_w'][start_id, :] if self.dim == 3 else None

                    dof_u_pred_horizon, dof_v_pred_horizon, dof_w_pred_horizon = self._predict_dofs_forward(
                        name=name,
                        ids=val_id[i:i + time_lag + horizon],
                        dof_u=dof_u_init,
                        dof_v=dof_v_init,
                        dof_w=dof_w_init,
                        total_steps=time_lag+horizon
                    )
                    
                    # compute error growth metrics and save to pred_path
                    true_id = val_id[i + time_lag:i+time_lag+horizon]
                    dof_true_u = f_latent[name]['dof_u'][true_id, :]
                    dof_true_v = f_latent[name]['dof_v'][true_id, :]
                    dof_true_w = f_latent[name]['dof_w'][true_id, :] if self.dim == 3 else None

                    dof_true = np.concatenate([dof_true_u, dof_true_v, dof_true_w], axis=1) if self.dim == 3 else np.concatenate([dof_true_u, dof_true_v], axis=1)
                    dof_true_horizons[i] = dof_true

                    temp_dof_pred_horizon = np.concatenate([dof_u_pred_horizon, dof_v_pred_horizon, dof_w_pred_horizon], axis=1) if self.dim == 3 else np.concatenate([dof_u_pred_horizon, dof_v_pred_horizon], axis=1)
                    dof_pred_horizon[i] = temp_dof_pred_horizon[time_lag:time_lag+horizon]
            

            # shape: (num_horizons, horizon)
            horizon_errors = l2_err_norm(dof_true_horizons, dof_pred_horizon, axis=2)
            with h5py.File(pred_path, 'a') as f:
                f.create_dataset(f'horizon_errors_{name}_{source_group}', data=horizon_errors)




        logger.info("Prediction complete")
        logger.info(f"\nReconstructing all predictions")
        self._pred_rec()


    def _pred_rec(self):
        """
        Reconstruct the full field predictions from the predicted dofs and save to HDF5.
        """
        logger.info(f"{'#'*20}\t{'Reconstructing predictions...':<20}\t{'#'*20}")
        # get all files in pred directory and loop through them
        pred_sources = self._pred_sources()
        for id, (source, split_key) in enumerate(pred_sources):
            name = source.get('name')
            source_group = 'train_data' if split_key == 'train_indices' else 'eval_data'
            pred_path = self.paths_bib.pred_dir + name + f'_{source_group}_pred.h5'
            ids = self.indices[source_group][name][split_key]
            logger.info(f"Reconstructing for {name} from {pred_path}...")
            with h5py.File(pred_path, 'r') as f:
                dof_u_pred = f['dof_u'][:]
                dof_v_pred = f['dof_v'][:]
                dof_w_pred = f['dof_w'][:] if self.dim == 3 else None

            if self.config['latent_params']['type'] == 'pod':
                pod.pod_recon(
                    rec_target=pred_path.replace('.h5', '_rec.h5'),
                    config=self.l_config,
                    dof_u=dof_u_pred,
                    dof_v=dof_v_pred,
                    dof_w=dof_w_pred,
                    batch_size=self.config['latent_params'].get('batch_size', 100),
                )
            elif self.dim == 3:
                self.dls.gfem_recon_flexible(
                        rec_target=pred_path.replace('.h5', '_rec.h5'),
                        config=self.l_config,
                        dof_u=dof_u_pred,
                        dof_v=dof_v_pred,
                        dof_w=dof_w_pred,
                        batch_size=self.config['latent_params'].get('batch_size', 100)
                    )
            else:
                self.dls.gfem_recon_flexible(
                        rec_target=pred_path.replace('.h5', '_rec.h5'),
                        config=self.l_config,
                        dof_u=dof_u_pred,
                        dof_v=dof_v_pred,
                        batch_size=self.config['latent_params'].get('batch_size', 100)
                    )
            self._add_global_pod(pred_path.replace('.h5', '_rec.h5'),
                                 name, source_group, ids)

    def _add_global_pod(self, rec_path, name, source_group, ids):
        """Add the global POD field back onto a reconstructed DLS residual.

        When the DLS source is a POD residual (scripts/pod_decompose.py) the
        reconstruction holds only the fine-scale part, while evaluation scores
        against the original field, so the global part must be restored:

            u = mean + sum_i a_i(t) phi_i(x) + r(x,t)

        ``latent_params.global_pod`` names the companion file with the modes and
        coefficients. The mean stays subtracted, matching the convention that
        Q_rec is a fluctuation field everywhere else.

        The coefficients used are the *true* ones, so this scores the local
        branch against a perfect global branch -- an upper bound on the hybrid,
        which is what a first feasibility test wants.
        """
        stem = self.config['latent_params'].get('global_pod')
        if not stem:
            return
        pod_path = self.paths_bib.data_dir + stem + '.h5'
        if not os.path.exists(pod_path):
            logger.warning("global_pod file %s not found; leaving the residual "
                           "reconstruction unmodified.", pod_path)
            return

        # 'coeffs' are the true coefficients (an upper bound on the hybrid, since
        # the global branch is then free); 'coeffs_pred' are a forecast from
        # scripts/pod_forecast.py, which is the honest end-to-end configuration
        key = self.config['latent_params'].get('global_pod_coeffs', 'coeffs')
        with h5py.File(pod_path, 'r') as f:
            if key not in f:
                logger.warning("global_pod_coeffs '%s' not in %s; falling back to 'coeffs'",
                               key, os.path.basename(pod_path))
                key = 'coeffs'
            modes = f['modes'][:]                    # (nx, ny[, nz], nv, K)
            coeffs = f[key][:]                       # (num_snaps, K)
        logger.info("Global POD branch: using '%s'", key)

        with h5py.File(rec_path, 'r+') as f:
            n_rec = f['Q_rec'].shape[0]
            idx = np.asarray(ids)[:n_rec]
            # rec row i is snapshot ids[i] (the seeded window included), so the
            # same rows index the coefficient record
            field = np.tensordot(coeffs[idx], modes, axes=([1], [modes.ndim - 1]))
            crop = tuple(slice(0, n) for n in f['Q_rec'].shape[1:])
            f['Q_rec'][:] = f['Q_rec'][:] + field[(slice(None),) + crop]
        logger.info("Restored POD-%d global field onto %s",
                    coeffs.shape[1], os.path.basename(rec_path))

    def _codec_reference(self, name, source_group, ids):
        """Reconstruct the TRUE latent coefficients over `ids`, cached to disk.

        Comparing a predicted field's spectrum against raw ground truth is
        dominated by the latent codec, not the model: the DLS round trip alone
        injects broadband high-k energy at the element boundaries (measured at
        ~4e5x the true high-k energy on the LDC config, with zero model error).
        Scoring against this round trip instead isolates the model's own
        contribution. It depends only on the latent space, so it is stored once
        per latent and dataset (lib/codec_ref) and this run's path is a small
        link into it. Returns that path, or None if it could not be built.
        """
        ref_path = os.path.join(
            self.paths_bib.pred_dir, f"{name}_{source_group}_codec_ref_rec.h5")
        try:
            return codec_ref.link(ref_path, codec_ref.shared_path(self.paths_bib.latent_dir, name),
                                  ids, lambda lo, hi: self._reconstruct_true(name, lo, hi))
        except Exception:
            logger.exception("Could not build codec reference for %s %s", name, source_group)
            return None

    def _reconstruct_true(self, name, lo, hi):
        """Field reconstruction of the true latent coefficients of snapshots lo..hi-1."""
        with h5py.File(self.paths_bib.latent_path, 'r') as f:
            dof_u = f[name]['dof_u'][lo:hi]
            dof_v = f[name]['dof_v'][lo:hi]
            dof_w = f[name]['dof_w'][lo:hi] if self.dim == 3 else None
        batch = self.config['latent_params'].get('batch_size', 100)
        if self.config['latent_params']['type'] == 'pod':
            return pod.pod_recon(rec_target=None, config=self.l_config, dof_u=dof_u,
                                 dof_v=dof_v, dof_w=dof_w, batch_size=batch)
        if self.dim == 3:
            return self.dls.gfem_recon_flexible(rec_target=None, config=self.l_config, dof_u=dof_u,
                                                dof_v=dof_v, dof_w=dof_w, batch_size=batch)
        return self.dls.gfem_recon_flexible(rec_target=None, config=self.l_config,
                                            dof_u=dof_u, dof_v=dof_v, batch_size=batch)

    def _save_metrics(self, name, source_group, scalars, arrays=None):
        """Persist one evaluation's scalars and time series to metrics_dir.

        eval() previously emitted numbers only through logger.info, so runs
        could not be compared after the fact. The loss and rollout config are
        stamped in so a metrics file identifies the run that produced it.
        """
        import json
        os.makedirs(self.paths_bib.metrics_dir, exist_ok=True)
        stem = os.path.join(self.paths_bib.metrics_dir, f"{name}_{source_group}")
        mp = self.config['model_params']
        record = {
            'name': self.config.get('name'),
            'latent_id': self.paths_bib.latent_id,
            'model_type': mp.get('model_type'),
            'train_ahead': mp.get('train_ahead'),
            'loss': mp.get('loss', 'mse'),
            'rollout': mp.get('rollout', {'mode': 'detached'}),
            'grad_clip': mp.get('grad_clip', GRAD_CLIP_MAX_NORM),
            'seed': mp.get('seed'),
            'split_version': self.indices.get('train_data', {}).get(name, {}).get('split_version', 1),
            'metrics': {k: (None if v is None else float(v)) for k, v in scalars.items()},
        }
        with open(stem + '.json', 'w') as f:
            json.dump(record, f, indent=2)
        if arrays:
            with h5py.File(stem + '.h5', 'w') as f:
                for key, val in arrays.items():
                    f.create_dataset(key, data=np.asarray(val))
        logger.info(f"{'#'*20} Metrics for {name} {source_group} {'#'*20}")
        for key, val in scalars.items():
            logger.info(f"  {key:<28} {val:.6g}" if val is not None else f"  {key:<28} n/a")
        logger.info(f"Metrics saved to {stem}.json")

    def eval(self) -> None:
        """Evaluate predictions against ground truth for all sets in config['eval_data'], save metrics and plots."""
        
        logger.info(f"{'#'*20}\t{'Evaluating...':<20}\t{'#'*20}")
        # get all files in pred directory and loop through them
        self.latent_id = self.paths_bib.latent_id
        pl.plot_loss(self)

        pred_sources = self._pred_sources()

        for id, (source, split_key) in enumerate(pred_sources):
            name = source.get('name')
            source_group = 'train_data' if split_key == 'train_indices' else 'eval_data'
            ids = self.indices[source_group][name][split_key]
            self.frequency = source.get('frequency', 100)
            z_slice = source.get('z_slice', self.l_config.nz_t//2) if self.dim == 3 else None
            y_slice = source.get('y_slice', self.l_config.ny_t//2) if self.dim == 3 else None
            x_slice = source.get('x_slice', 50) if self.dim == 3 else None
            
            self.paths_bib.pred_fig_dir = os.path.join(self.paths_bib.fig_dir, 'pred' + name + source_group + '/')
            
            os.makedirs(self.paths_bib.pred_fig_dir, exist_ok=True)
            rec_path = self.paths_bib.pred_dir + name + f'_{source_group}_pred_rec.h5'
            # with a POD-residual source the model trains on the residual
            # but is scored against the original field
            gt_stem = self.config['latent_params'].get('full_source', source.get('path'))
            gt_path = self.paths_bib.data_dir + gt_stem + '.h5'
            logger.info(f"Evaluating for {name} between {rec_path} and {gt_path}...")
            
            # Compute metrics
            stat_errors = self.compute_TKE(rec_path, gt_path, name, source_group, ids)
            stat_errors.update(self.compute_RMS(rec_path, gt_path, name, source_group, ids))

            pl.plot_TKE(self, rec_path, gt_path, name, source_group,  ids)
            if self.dim == 3:
                pl.plot_RMS(self, rec_path, gt_path, name, source_group, z_slice=z_slice)
                pl.plot_RMS(self, rec_path, gt_path, name, source_group, y_slice=y_slice)
                pl.plot_RMS(self, rec_path, gt_path, name, source_group, x_slice=x_slice)

                pl.plot_slice_compare(self, rec_path, gt_path, name, ids, 0,  z_slice=z_slice)
                pl.plot_slice_compare(self, rec_path, gt_path, name, ids, 29,  z_slice=z_slice)
                pl.plot_slice_compare(self, rec_path, gt_path, name, ids, 0,  z_slice=z_slice)
                pl.plot_slice_compare(self, rec_path, gt_path, name, ids, 29,  z_slice=z_slice)

                pl.plot_slice_compare(self, rec_path, gt_path, name, ids, 0,  y_slice=y_slice)
                pl.plot_slice_compare(self, rec_path, gt_path, name, ids, 29,  y_slice=y_slice)
                
                pl.plot_slice_compare(self, rec_path, gt_path, name, ids, 0,  x_slice=x_slice)
                pl.plot_slice_compare(self, rec_path, gt_path, name, ids, 29,  x_slice=x_slice)
                
                pl.q_criterion(self, rec_path, gt_path, name, ids, 0)
                pl.q_criterion(self, rec_path, gt_path, name, ids, 29)
                pl.anim_q_criterion(self, rec_path, gt_path, name, ids)
            else:
                pl.plot_RMS(self, rec_path, gt_path, name, source_group)
                pl.plot_slice_compare(self, rec_path, gt_path, name, ids, 0)
                pl.plot_slice_compare(self, rec_path, gt_path, name, ids, 9)
                pl.plot_slice_compare(self, rec_path, gt_path, name, ids, 19)
                pl.plot_slice_compare(self, rec_path, gt_path, name, ids, 29)
                pl.plot_slice_compare(self, rec_path, gt_path, name, ids, 39)
                pl.plot_slice_compare(self, rec_path, gt_path, name, ids, 49)
                pl.plot_slice_compare(self, rec_path, gt_path, name, ids, 99)

            pl.plot_horizon_errors(self, rec_path, gt_path, name, source_group)

            # rollout statistics: whether the prediction still behaves like the
            # right flow after its trajectory has decorrelated
            # TKE/RMS errors vs truth: previously only logged, now in the metrics file
            scalars = dict(stat_errors)
            arrays = {}
            try:
                s, a = pl.plot_energy_retention(self, rec_path, gt_path, name, source_group, ids)
                scalars.update(s); arrays.update(a)
                s, a = pl.plot_field_error_vs_time(self, rec_path, gt_path, name, source_group, ids)
                scalars.update(s); arrays.update(a)
                ref_path = self._codec_reference(name, source_group, ids)
                scalars.update(pl.plot_energy_spectrum(self, rec_path, gt_path, name, source_group, ids,
                                                       ref_path=ref_path))
                scalars.update(pl.plot_velocity_pdf(self, rec_path, gt_path, name, source_group, ids))
                scalars.update(pl.plot_autocorrelation(self, rec_path, gt_path, name, source_group, ids))
            except Exception:
                logger.exception("Rollout statistics failed for %s %s", name, source_group)

            # many short forecasts launched across the split: the across-forecast
            # error and the persistence baseline a short-horizon task is judged on
            if self.model is not None:
                try:
                    s, a = multistart.evaluate(self, name, ids)
                    scalars.update(s); arrays.update(a)
                    if a:
                        pl.plot_multistart_errors(self, a['ms_err'], a['ms_persist'])
                except Exception:
                    logger.exception("Multi-start evaluation failed for %s %s", name, source_group)
            self._save_metrics(name, source_group, scalars, arrays=arrays or None)






            
                
            

            
    def _cached_gt(self, gt_path, key, compute, valid=None):
        """Read `key` from the ground-truth file, computing and storing it on first use.

        Opened read-only unless the entry is missing (or fails `valid`, which then
        replaces it): a write open takes an exclusive lock on the shared data
        file, which made a second eval running alongside crash. A write that
        finds the file busy waits and retries.
        """
        ok = (lambda ds: True) if valid is None else valid
        for attempt in range(60):             # another eval may hold the write lock briefly
            try:
                f = h5py.File(gt_path, 'r')
                break
            except OSError:
                if attempt == 59:
                    raise
                time.sleep(10)
        with f:
            if key in f and ok(f[key]):
                logger.info(f"{key} found in {gt_path}")
                return f[key][:]
            value = compute(f)
        for attempt in range(60):
            try:
                with h5py.File(gt_path, 'r+') as f:
                    if key in f and not ok(f[key]):
                        del f[key]
                    if key not in f:
                        f.create_dataset(key, data=value)
                break
            except OSError:
                if attempt == 59:
                    raise
                time.sleep(10)
        logger.info(f"{key} computed and saved to {gt_path}")
        return value

    def _true_tke(self, f, weights, n, batch=250):
        """Per-snapshot weighted fluctuation energy of snapshots [0, n) of an open truth file.

        Cropped to the latent's grid and read in batches, so memory is bounded
        whatever the record length.
        """
        nx, ny = self.l_config.nx_t, self.l_config.ny_t
        nz = self.l_config.nz_t if self.dim == 3 else 1
        if self.dim == 3:
            mean_gt = f['mean'][:nx, :ny, :nz, :]
        else:
            crop = self._xy_crop(nx, ny)
            mean_gt = self._to_xy(f['mean'][crop], 0)
        TKE = np.empty(n)
        for lo in range(0, n, batch):
            hi = min(lo + batch, n)
            if self.dim == 3:
                Q_gt = f['UV'][lo:hi, :nx, :ny, :nz, :] - mean_gt[np.newaxis]
            else:
                Q_gt = self._to_xy(f['UV'][(slice(lo, hi),) + crop], 1) - mean_gt[np.newaxis]
            Q_gt_weighted = (Q_gt * weights[np.newaxis, ..., np.newaxis]).reshape(hi - lo, -1, order='F')
            TKE[lo:hi] = 0.5 * np.sum(Q_gt_weighted**2, axis=-1)
        return TKE

    def compute_TKE(self, rec_path, gt_path, name, source, ids):
        """
        Compute TKE and save to h5s. Returns {'tke_error_pct': ...}.
        """
        import lib.weights as wg
        time_lag = self.config['model_params']['time_lag']
        val_id = ids
        nx = self.l_config.nx_t
        ny = self.l_config.ny_t
        nz = self.l_config.nz_t if self.dim == 3 else 1

        def compute_weights(f):
            if self.dim == 3:
                return wg.generate_weights_grid_3d(self.x_grid, self.y_grid, self.z_grid)
            if self.source_is_yx:
                # stored in the source's own layout, like every other array there
                return wg.generate_weights_grid_2d(self.x_grid.T, self.y_grid.T)
            return wg.generate_weights_grid_2d(self.x_grid, self.y_grid)
        weights = self._cached_gt(gt_path, 'weights', compute_weights)

        weights = self._to_xy(weights, 0)

        if self.dim == 3:
            weights = weights[:nx, :ny, :nz]
        else:
            weights = weights[:nx, :ny]


        with h5py.File(rec_path, 'r+') as f:
            if 'TKE_rec' in f:
                logger.info(f"TKE for reconstructed already computed and found in {rec_path}. Skipping computation...")
                TKE_rec = f['TKE_rec'][:]
            else:
                logger.info(f"TKE for reconstructed not found in {rec_path}. Computing TKE...")
                Q_rec = f['Q_rec'][:] # shape: (snaps, ..., dim)
                Q_rec_weighted = Q_rec * weights[np.newaxis, ..., np.newaxis] # shape: (snaps, ..., dim)
                Q_rec_weighted = Q_rec_weighted.reshape(Q_rec_weighted.shape[0], -1) 
                
                TKE_rec = 0.5 * np.sum(Q_rec_weighted**2, axis=-1) # shape: (snaps, ...)
                f.create_dataset('TKE_rec', data=TKE_rec)

            
        

        # per-snapshot over the snapshots the latent holds, indexed by id below so
        # any run's window reads it; keyed by latent because the crop is. Read in
        # batches: a record can be far larger than what is used (the Re 30k file
        # holds 60k snapshots, 116 GB at this crop, of which 4,500 are used).
        with h5py.File(self.paths_bib.latent_path, 'r') as f:
            n_latent = f[name]['dof_u'].shape[0]
        need = int(np.max(val_id)) + 1
        TKE_gt = self._cached_gt(gt_path, 'TKE_gt_' + self.latent_id + '_' + name + source,
                                 lambda f: self._true_tke(f, weights, min(n_latent, f['UV'].shape[0])),
                                 valid=lambda ds: ds.shape[0] >= need)

        TKE_error = pl.l2_err_norm(TKE_gt[val_id[time_lag:]], TKE_rec[time_lag:])
        logger.info(f"TKE error for {name} {source}: {100 * TKE_error:.4f}%")
        return {'tke_error_pct': 100 * float(TKE_error)}
                
            
    def compute_RMS(self, rec_path, gt_path, name, source, ids):
        """
        Compute RMS error and save to h5s
        """
        time_lag = self.config['model_params']['time_lag']
        val_id = ids
        nx = self.l_config.nx_t
        ny = self.l_config.ny_t
        nz = self.l_config.nz_t if self.dim == 3 else 1

        print(f"nx: {nx}, ny: {ny}, nz: {nz}")

        with h5py.File(rec_path, 'r+') as f:
            if 'RMS_rec' in f:
                logger.info(f"RMS for reconstructed already computed and found in {rec_path}. Skipping computation...")
                RMS_rec = f['RMS_rec'][:]
            else:
                logger.info(f"RMS for reconstructed not found in {rec_path}. Computing RMS...")
                Q_rec = f['Q_rec'][time_lag:] # shape: (snaps, ..., dim)
                RMS_rec = np.sqrt(np.mean(Q_rec**2, axis=0)) # shape: (snaps, ...)
                f.create_dataset('RMS_rec', data=RMS_rec)


        # The true RMS must average the same snapshots as RMS_rec, val_id[time_lag:],
        # so the cache is keyed by that window. Keying it by latent and split (as
        # before) froze whichever run's window was evaluated first: on the LDC
        # p99m10 latent every run was scored against val_id[64:]. The uncropped
        # field does not depend on the latent, so latents share the entry.
        window = np.asarray(val_id[time_lag:])
        digest = hashlib.md5(window.astype(np.int64).tobytes()).hexdigest()[:8]
        key = f"RMS_gt_{name}_{int(window[0])}-{int(window[-1])}_{digest}"

        def compute_rms_gt(f):
            # kept in the source's own layout, like every other array there
            mean_gt = f['mean'][:]
            Q_gt = f['UV'][window] - mean_gt[np.newaxis]  # shape: (snaps, ..., dim)
            return np.sqrt(np.mean(Q_gt**2, axis=0))  # shape: (..., dim)
        RMS_gt = self._cached_gt(gt_path, key, compute_rms_gt)

        RMS_gt = self._to_xy(RMS_gt, 0)
        RMS_gt = RMS_gt[:nx, :ny, :nz, :] if self.dim == 3 else RMS_gt[:nx, :ny, :]
        print(f"RMS_gt shape: {RMS_gt.shape}, RMS_rec shape: {RMS_rec.shape}")
        RMS_error = pl.l2_err_norm(RMS_gt, RMS_rec)
        logger.info(f"RMS error for {name} {source}: {100 * RMS_error:.4f}%")
        RMS_u_error = pl.l2_err_norm(RMS_gt[..., 0], RMS_rec[..., 0])
        RMS_v_error = pl.l2_err_norm(RMS_gt[..., 1], RMS_rec[..., 1])
        logger.info(f"RMS_u error for {name} {source}: {100 * RMS_u_error:.4f}%")
        logger.info(f"RMS_v error for {name} {source}: {100 * RMS_v_error:.4f}%")
        out = {'rms_error_pct': 100 * float(RMS_error), 'rms_u_error_pct': 100 * float(RMS_u_error),
               'rms_v_error_pct': 100 * float(RMS_v_error)}
        if self.dim == 3:
            RMS_w_error = pl.l2_err_norm(RMS_gt[..., 2], RMS_rec[..., 2])
            logger.info(f"RMS_w error for {name} {source}: {100 * RMS_w_error:.4f}%")
            out['rms_w_error_pct'] = 100 * float(RMS_w_error)
        return out
                