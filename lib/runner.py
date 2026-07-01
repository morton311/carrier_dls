import logging
import os
import sys
import h5py
import pickle
import copy

import numpy as np
import torch
from torchinfo import summary
from torch import nn
from tqdm import tqdm
import time

import lib.init as init
import lib.pod as pod
import lib.models as models
import lib.datas as datas
import lib.plotting as pl
from lib.metrics import l2_err_norm

logger = logging.getLogger(__name__)

GRAD_CLIP_MAX_NORM = 0.2
CHECKPOINT_INTERVAL = 5
MAX_ERROR_HORIZONS = 30


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
        self._get_model()
        self._compile_model()

        
        

    def _init_paths_and_logging(self, config):
        is_init_path, paths = init.init_path(config)
        self.data_sources = ['train_data', 'eval_data']
        self.config['group_names'] = paths.data_dict
        if config['mode'] != 'compare' and config['log'] == 'file':
            log_path = paths.log_path
            logging.info("To follow the log in real time, run 'tail -f %s'", log_path, exc_info=True)
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

            logger.info("Logging to %s", log_path, exc_info=True)
        logger.info('Using device: %s', self.device, exc_info=True)
        return paths
    
    def _get_grid(self):
        with h5py.File(self.paths_bib.source_path, 'r') as f:
            self.dim = 3 if 'z_grid' in f else 2
            self.x_grid = f['x_grid'][:]
            self.y_grid = f['y_grid'][:]
            self.z_grid = f['z_grid'][:] if self.dim == 3 else None

            if 'x' in f:
                self.x = f['x'][:]
                self.y = f['y'][:]
                self.z = f['z'][:] if 'z' in f else None
            else:
                self.x = self.x_grid.shape[0]
                self.y = self.y_grid.shape[0]
                self.z = self.z_grid.shape[0] if self.dim == 3 else None
        
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
        if lp['type'] == 'dls':
            return self.dls.gfem_compress_flexible(
                data_source=data_path,
                field_name='UV',
                group_name=group_name,
                patch_size=lp['patch_size'],
                num_modes=lp['num_modes'],
                latent_target=self.paths_bib.latent_path,
                batch_size=lp['batch_size'],
                dls_config=latent_config,
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
                            continue
                        
                        # Compute splits for this combination
                        logger.info(f"Computing splits for {data_source} '{data_name}...")
                        if data_name not in snaps[data_source]:
                            snaps[data_source][data_name] = {}
                        
                        snaps[data_source][data_name]['total'] = f[data_name]['dof_u'].shape[0]
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
            

    def _split_indices(self, total_snaps, train_split=0.8, test_split=0.1, sample_train=0, sample_test=0):
        # find indices for train, test, and validation sets
        train_len = int(total_snaps * train_split)
        test_len = int(train_len * test_split)

        train_indices = np.arange(0, train_len - test_len)
        test_indices = np.arange(train_len - test_len, train_len)
        val_indices = np.arange(train_len, total_snaps)

        logger.info(f"{'Set':<12}|{'Total':<10}|{'First Idx':<12}|{'Last Idx':<12}|{'Sampled':<10}")
        logger.info("-" * 56)
        logger.info(f"{'Train':<12}|{train_len:<10}|{train_indices[0]:<12}|{train_indices[-1]:<12}|{sample_train:<10}")
        logger.info(f"{'Test':<12}|{len(test_indices):<10}|{test_indices[0]:<12}|{test_indices[-1]:<12}|{sample_test:<10}")
        logger.info(f"{'Validation':<12}|{len(val_indices):<10}|{val_indices[0]:<12}|{val_indices[-1]:<12}|{'-':<10}")

        if not os.path.exists(self.paths_bib.model_dir + 'split_ids.pkl'):
            indices = {}


            indices['train_indices'] = np.sort(datas.sample_series_indices(
                                        train_len, 
                                        sample_train, 
                                        time_lag=self.config['model_params']['time_lag'], 
                                        train_ahead=self.config['model_params']['train_ahead'], 
                                        seed=42))
            indices['test_indices'] = np.sort(datas.sample_series_indices(
                                        len(test_indices), 
                                        sample_test, 
                                        time_lag=self.config['model_params']['time_lag'], 
                                        train_ahead=self.config['model_params']['train_ahead'], 
                                        seed=42))
            indices['val_indices'] = val_indices

            # save the train, test, and validation indices
            with open(self.paths_bib.model_dir + 'split_ids.pkl', 'wb') as f:
                pickle.dump(indices, f)
            logger.info(f"Train, test, and validation indices saved to {self.paths_bib.model_dir + 'split_ids.pkl'}")

        else:
            with open(self.paths_bib.model_dir + 'split_ids.pkl', 'rb') as f:
                indices = pickle.load(f)
            logger.info(f"Train, test, and validation indices loaded from {self.paths_bib.model_dir + 'split_ids.pkl'}")
        
        return indices
    
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
                        )
        elif self.config['model_params']['model_type'] == 'f_extrap':
            self.model = None
        else:
            raise ValueError(f"Model {self.config['model_params']['model_type']} not recognized. Please use 'tr_enc', 'tr_encdec', or 'lstm'.")
        
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

            if not self.config['model_params']['model_type'] == 'tr_encdec':
                summary(self.model, input_size=input_size)

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
            
            self.criterion = nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['model_params']['lr'])
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config['model_params'].get('lr_factor', 0.5),
                patience=self.config['model_params'].get('lr_patience', 5),
                min_lr=1e-7
            )
            # torch.amp.GradScaler exists from torch 2.3; fall back for older versions
            if hasattr(torch.amp, 'GradScaler'):
                self.scaler = torch.amp.GradScaler()
            else:
                self.scaler = torch.cuda.amp.GradScaler()

            logger.info(f"Loss function: {self.criterion}")
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
        self._model_fit()


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

    def _get_train_data(self):
        """Get training and test data as torch tensors, minimizing memory usage."""
        if self.config['model_params']['model_type'] == 'tr_encdec':
            return self._get_train_data_encdec()
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

                    self.dof_mean[name] = torch.mean(dofs, dim=(0, 1))
                    self.dof_std[name] = torch.std(dofs, dim=(0, 1))
                    logger.info(f"Mean/std shapes: {self.dof_mean[name].shape}, {self.dof_std[name].shape}")

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


    def _train_epoch(self, epoch: int, max_norm: float) -> float:
        """Single forward+backward pass over all training loaders. Returns mean epoch loss."""
        if self.config['model_params']['model_type'] == 'tr_encdec':
            return self._train_epoch_encdec(epoch, max_norm)
        self.model.train()
        epoch_loss = 0.0
        for key in self.train_loader:
            if self.config['distributed']:
                self.sampler[key].set_epoch(epoch)
            for inputs, targets in self.train_loader[key]:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                total_loss = 0.0
                for n in range(targets.shape[1]):
                    target = targets[:, n, :]
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, target)
                    total_loss += loss
                    self.scaler.scale(loss).backward()
                    inputs = torch.cat((inputs[:, 1:, :], outputs.detach().unsqueeze(1)), dim=1)
                epoch_loss += total_loss.item() / (targets.shape[1] * len(self.train_loader[key]))
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        return epoch_loss

    def _validate_epoch(self) -> float:
        """Single forward pass over all test loaders. Returns mean test loss."""
        if self.config['model_params']['model_type'] == 'tr_encdec':
            return self._validate_epoch_encdec()
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for key in self.test_loader:
                for inputs, targets in self.test_loader[key]:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    for n in range(targets.shape[1]):
                        target = targets[:, n, :]
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, target)
                        test_loss += loss.item() / (targets.shape[1] * len(self.test_loader[key]))
                        inputs = torch.cat((inputs[:, 1:, :], outputs.unsqueeze(1)), dim=1)
        return test_loss

    ## ======================= Global-local encoder-decoder (tr_encdec) ==========================

    def _encdec_geometry(self, use_skip=True):
        """Precompute and cache element geometry for the encdec model.

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
        if not hasattr(self, '_encdec_geom_cache'):
            self._encdec_geom_cache = {}
        if key in self._encdec_geom_cache:
            return self._encdec_geom_cache[key]

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
        self._encdec_geom_cache[key] = geom
        logger.info(f"encdec geometry (skip={key}): {lltogl_mat.shape[0]} elements, {num_dofs} nodal DOFs")
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

    def _reassemble_tokens(self, pred_norm, name, geom):
        """Project normalized element predictions to a consistent global field and back.

        (G, E, F) -> scatter-mean shared nodal DOFs -> regather -> (G, E, F).
        Runs in fp32; keep outside autocast.
        """
        G = pred_norm.shape[0]
        mean = self.dof_mean[name].to(pred_norm.device)
        std = self.dof_std[name].to(pred_norm.device)
        pred = pred_norm.float() * std + mean
        ndof = self.l_config.dof_elem
        comps = []
        for c in range(self.dim):
            flat = torch.zeros(G, geom['num_dofs'], device=pred.device)
            flat.index_add_(1, geom['scatter_index'],
                            pred[:, :, c * ndof:(c + 1) * ndof].reshape(G, -1))
            flat = flat / geom['counts']
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

    def _get_train_data_encdec(self):
        """Build snapshot-grouped train/test loaders for the encdec model."""
        if self.model is None:
            return
        logger.info('Getting training and test data (snapshot-grouped, tr_encdec)')
        if not self.config['latent_params'].get('localized', False):
            raise ValueError("model_type 'tr_encdec' requires latent_params.localized = true")
        mp = self.config['model_params']
        tl = mp['time_lag']
        ta = mp['train_ahead']
        cw = mp.get('context_window', 1)
        ct = mp.get('context_time', 'window_end')

        geom = self._encdec_geometry()
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
                self.dof_mean[name] = torch.mean(dofs, dim=(0, 1))
                self.dof_std[name] = torch.std(dofs, dim=(0, 1))
                logger.info(f"Mean/std shapes: {self.dof_mean[name].shape}, {self.dof_std[name].shape}")
                del dofs

                dof_comps = [torch.from_numpy(dof_u[:]).float(), torch.from_numpy(dof_v[:]).float()]
                if self.dim == 3:
                    dof_comps.append(torch.from_numpy(dof_w[:]).float())

                train_set = datas.LocalGlobalDataset(
                    dof_comps, lltogl_cpu, train_indices, tl, ta,
                    self.dof_mean[name], self.dof_std[name], cw, ct)
                test_set = datas.LocalGlobalDataset(
                    dof_comps, lltogl_cpu, test_indices, tl, ta,
                    self.dof_mean[name], self.dof_std[name], cw, ct)

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

    def _train_epoch_encdec(self, epoch: int, max_norm: float) -> float:
        """Encdec training epoch: encode shared context per snapshot group, multi-step local loss."""
        self.model.train()
        geom = self._encdec_geometry()
        coords = geom['coords']
        ss_p = self._ss_prob(epoch)
        epoch_loss = 0.0
        for key in self.train_loader:
            if self.config['distributed']:
                self.sampler[key].set_epoch(epoch)
            for inputs, targets, tokens, _ in self.train_loader[key]:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                tokens = tokens.to(self.device)
                self.optimizer.zero_grad()
                total_loss = 0.0
                for n in range(targets.shape[2]):
                    target = targets[:, :, n, :]
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.model(inputs, tokens, coords)
                        loss = self.criterion(outputs, target)
                    total_loss += loss
                    self.scaler.scale(loss).backward()
                    inputs = torch.cat((inputs[:, :, 1:, :], outputs.detach().unsqueeze(2)), dim=2)
                    if ss_p > 0 and torch.rand(()).item() < ss_p:
                        new_frame = self._reassemble_tokens(outputs.detach(), key, geom)
                        tokens = self._shift_tokens(tokens, new_frame)
                epoch_loss += total_loss.item() / (targets.shape[2] * len(self.train_loader[key]))
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        return epoch_loss

    def _validate_epoch_encdec(self) -> float:
        """Encdec validation epoch (teacher-forced context)."""
        self.model.eval()
        geom = self._encdec_geometry()
        coords = geom['coords']
        test_loss = 0.0
        with torch.no_grad():
            for key in self.test_loader:
                for inputs, targets, tokens, _ in self.test_loader[key]:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    tokens = tokens.to(self.device)
                    for n in range(targets.shape[2]):
                        target = targets[:, :, n, :]
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = self.model(inputs, tokens, coords)
                            loss = self.criterion(outputs, target)
                        test_loss += loss.item() / (targets.shape[2] * len(self.test_loader[key]))
                        inputs = torch.cat((inputs[:, :, 1:, :], outputs.unsqueeze(2)), dim=2)
        return test_loss

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

            for epoch in range(len(losses), self.config['model_params']['num_epochs']):
                epoch_loss = self._train_epoch(epoch, GRAD_CLIP_MAX_NORM)
                losses.append(epoch_loss)

                test_loss = self._validate_epoch()
                test_losses.append(test_loss)

                self.scheduler.step(test_loss)
                new_lr = self.optimizer.param_groups[0]['lr']

                if epoch > 1:
                    if np.isnan(test_losses[-1]) or np.isnan(losses[-1]):
                        logger.info(f'NaN loss at epoch {epoch+1}. Stopping training.')
                        self.model.load_state_dict(best_model)
                        break
                    elif test_loss < best_test_loss:
                        best_test_loss = test_loss
                        best_model = copy.deepcopy(self.model.state_dict())
                        best_epoch = epoch + 1
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= self.config['model_params']['patience']:
                            logger.info(f'Early stopping at epoch {epoch+1}')
                            self.model.load_state_dict(best_model)
                            logger.info(f'Best model loaded from epoch {best_epoch}, with test loss: {best_test_loss:.4f}')
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
                logger.info(f"| Epoch: {epoch+1:<4}/{self.config['model_params']['num_epochs']:<4} | Train Loss: {losses[-1]:7.4f} | Test Loss: {test_losses[-1]:7.4f} | Best: {best_flag:<1} | Patience: {early_stop_counter:<3}/{self.config['model_params']['patience']} | Checkpoint: {checkpoint_flag:<1} | LR: {new_lr:.2e} | Time: {(time.time() - start_time)/60:10.2f} min |")

            end_time = time.time()
            logger.info('\n\nTime taken for training: %s minutes', (end_time - start_time) / 60, exc_info=True)
            logger.info('Time taken per epoch: %s minutes', (end_time - start_time) / 60 / len(losses), exc_info=True)

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

    def _predict_dofs_forward(self, name, dof_u, dof_v, dof_w=None, total_steps=None):
        """
        Predict dof_u/dof_v(/dof_w) forward in time from initial conditions.
        Inputs dof_u/dof_v(/dof_w) are expected to contain at least `time_lag` snapshots.
        Returns full predicted trajectories with shape (total_steps, num_dofs).
        """
        if self.config['model_params']['model_type'] == 'tr_encdec':
            return self._predict_dofs_forward_encdec(name, dof_u, dof_v, dof_w, total_steps)
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

    def _predict_dofs_forward_encdec(self, name, dof_u, dof_v, dof_w=None, total_steps=None):
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

        geom = self._encdec_geometry(use_skip=False)
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

        def window_tokens(window):
            frames = window[:, :, -cw:, :] if ct == 'window_end' else window[:, :, :cw, :]
            return frames.reshape(1, num_elems, cw * frames.shape[-1])

        self.model.eval()
        model = self.model.module if hasattr(self.model, 'module') else self.model
        F_dim = self.dim * self.l_config.dof_elem
        predictions = np.zeros((num_elems, total_steps, F_dim))
        predictions[:, :time_lag, :] = dof_input[0].cpu().numpy()

        with torch.no_grad():
            context = model.encode(window_tokens(dof_input), coords)
            for t in range(total_steps - time_lag):
                output = model.decode(dof_input, context)  # (1, E, F)
                predictions[:, t + time_lag, :] = output[0].cpu().numpy()
                dof_input = torch.cat((dof_input[:, :, 1:, :], output.unsqueeze(2)), dim=2)
                if (t + 1) % global_K == 0 and (t + 1) < total_steps - time_lag:
                    frames = dof_input[:, :, -cw:, :] if ct == 'window_end' else dof_input[:, :, :cw, :]
                    re = torch.stack(
                        [self._reassemble_tokens(frames[:, :, i, :], name, geom) for i in range(cw)],
                        dim=2)
                    context = model.encode(re.reshape(1, num_elems, cw * F_dim), coords)

        predictions = predictions * std_np + mean_np

        # scatter-mean assembly of shared nodal DOFs (interior nodes belong to multiple elements)
        num_dofs = geom['num_dofs']
        ndof = self.l_config.dof_elem
        counts = geom['counts_cpu'].numpy()
        out = []
        for c in range(self.dim):
            acc = torch.zeros(total_steps, num_dofs, dtype=torch.float64)
            vals = torch.from_numpy(
                predictions[:, :, c * ndof:(c + 1) * ndof].transpose(1, 0, 2).reshape(total_steps, -1))
            acc.index_add_(1, geom['scatter_index_cpu'], vals)
            out.append((acc.numpy() / counts))
        dof_u_pred, dof_v_pred = out[0], out[1]
        dof_w_pred = out[2] if self.dim == 3 else None
        return dof_u_pred, dof_v_pred, dof_w_pred


    def pred(self) -> None:
        """Make predictions over the validation data for all sets in config['eval_data']."""
        logger.info(f"{'#'*20}\t{'Predicting...':<20}\t{'#'*20}")
        pred_sources = []
        if self.config.get('train_data') is not None:
            pred_sources.extend((source, 'train_indices') for source in self.config['train_data'])
        if self.config.get('eval_data') is not None:
            pred_sources.extend((source, 'val_indices') for source in self.config['eval_data'])
            

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
            horizon = self.config['model_params'].get('horizon', 10)
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
        pred_sources = []
        if self.config.get('train_data') is not None:
            pred_sources.extend((source, 'train_indices') for source in self.config['train_data'])
        if self.config.get('eval_data') is not None:
            pred_sources.extend((source, 'val_indices') for source in self.config['eval_data'])
        for id, (source, split_key) in enumerate(pred_sources):
            name = source.get('name')
            source_group = 'train_data' if split_key == 'train_indices' else 'eval_data'
            pred_path = self.paths_bib.pred_dir + name + f'_{source_group}_pred.h5'
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

    def eval(self) -> None:
        """Evaluate predictions against ground truth for all sets in config['eval_data'], save metrics and plots."""
        
        logger.info(f"{'#'*20}\t{'Evaluating...':<20}\t{'#'*20}")
        # get all files in pred directory and loop through them
        self.latent_id = self.paths_bib.latent_id
        pl.plot_loss(self)

        pred_sources = []
        if self.config.get('train_data') is not None:
            pred_sources.extend((source, 'train_indices') for source in self.config['train_data'])
        if self.config.get('eval_data') is not None:
            pred_sources.extend((source, 'val_indices') for source in self.config['eval_data'])

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
            gt_path = self.paths_bib.data_dir + source.get('path') + '.h5'
            logger.info(f"Evaluating for {name} between {rec_path} and {gt_path}...")
            
            # Compute metrics
            self.compute_TKE(rec_path, gt_path, name, source_group, ids)
            self.compute_RMS(rec_path, gt_path, name, source_group, ids)

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

            pl.plot_horizon_errors(self, rec_path, gt_path, name, source_group)






            
                
            

            
    def compute_TKE(self, rec_path, gt_path, name, source, ids):
        """
        Compute TKE and save to h5s
        """
        import lib.weights as wg
        time_lag = self.config['model_params']['time_lag']
        val_id = ids
        nx = self.l_config.nx_t
        ny = self.l_config.ny_t
        nz = self.l_config.nz_t if self.dim == 3 else 1
        # Check if weights are already computed and saved, if not compute and save them
        with h5py.File(gt_path, 'r+') as f:
            if 'weights' in f:
                logger.info(f"Weights already computed and found in {gt_path}. Loading weights...")
                weights = f['weights'][:]
            else:
                logger.info(f"Weights not found in {gt_path}. Computing weights...")
                if self.dim == 3:
                    weights = wg.generate_weights_grid_3d(self.x_grid, self.y_grid, self.z_grid)
                else:
                    weights = wg.generate_weights_grid_2d(self.x_grid, self.y_grid)
                f.create_dataset('weights', data=weights)
                logger.info(f"Weights computed and saved to {gt_path}.")
    
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

            
        

        with h5py.File(gt_path, 'r+') as f:

            if 'TKE_gt_'+self.latent_id + '_' + name + source in f:
                logger.info(f"TKE for GT already computed and found in {gt_path}. Skipping computation...")
                TKE_gt = f['TKE_gt_'+self.latent_id + '_' + name + source][:]
            else:
                logger.info(f"TKE for GT not found in {gt_path}. Computing TKE...")
                if self.dim == 3:
                    mean_gt = f['mean'][:nx, :ny, :nz, :] # shape: (nx, ny, nz, dim)
                    Q_gt = f['UV'][:, :nx, :ny, :nz, :] - mean_gt[np.newaxis]
                    Q_gt_weighted = Q_gt * weights[np.newaxis, ..., np.newaxis] # shape: (snaps, ..., dim)
                    Q_gt_weighted = Q_gt_weighted.reshape(Q_gt_weighted.shape[0], -1) 
                    
                else:
                    mean_gt = f['mean'][:nx, :ny, :] # shape: (nx, ny, dim)
                    Q_gt = f['UV'][:, :nx, :ny, :] - mean_gt[np.newaxis]
                    Q_gt_weighted = Q_gt * weights[np.newaxis, ..., np.newaxis] # shape: (snaps, ..., dim)
                    Q_gt_weighted = Q_gt_weighted.reshape(Q_gt_weighted.shape[0], -1)

                TKE_gt = 0.5 * np.sum(Q_gt_weighted**2, axis=-1) # shape: (snaps, ...)
                f.create_dataset('TKE_gt_'+self.latent_id + '_' + name + source, data=TKE_gt)

        TKE_error = pl.l2_err_norm(TKE_gt[val_id[time_lag:]], TKE_rec[time_lag:])
        logger.info(f"TKE error for {name} {source}: {100 * TKE_error:.4f}%")
                
            
    def compute_RMS(self, rec_path, gt_path, name, source, ids):
        """
        Compute RMS error and save to h5s
        """
        time_lag = self.config['model_params']['time_lag']
        val_id = ids
        nx = self.l_config.nx_t
        ny = self.l_config.ny_t
        nz = self.l_config.nz_t if self.dim == 3 else 1
        with h5py.File(rec_path, 'r+') as f:
            if 'RMS_rec' in f:
                logger.info(f"RMS for reconstructed already computed and found in {rec_path}. Skipping computation...")
                RMS_rec = f['RMS_rec'][:]
            else:
                logger.info(f"RMS for reconstructed not found in {rec_path}. Computing RMS...")
                Q_rec = f['Q_rec'][time_lag:] # shape: (snaps, ..., dim)
                Q_rec = Q_rec.reshape(Q_rec.shape[0], -1)
                RMS_rec = np.sqrt(np.mean(Q_rec**2, axis=0)) # shape: (snaps, ...)
                RMS_rec = RMS_rec.reshape(nx, ny, nz, 3) if self.dim == 3 else RMS_rec.reshape(nx, ny, 2)
                f.create_dataset('RMS_rec', data=RMS_rec)


        nx = self.l_config.nx
        ny = self.l_config.ny
        nz = self.l_config.nz if self.dim == 3 else 1

        with h5py.File(gt_path, 'r+') as f:
            if 'RMS_gt_'+self.latent_id + '_' + name+ source in f:
                logger.info(f"RMS for GT already computed and found in {gt_path}. Skipping computation...")
                RMS_gt = f['RMS_gt_'+self.latent_id + '_' + name+ source][:]
            else:
                logger.info(f"RMS for GT not found in {gt_path}. Computing RMS...")
                if self.dim == 3:
                    mean_gt = f['mean'][:]
                    Q_gt = f['UV'][val_id[time_lag:]] - mean_gt[np.newaxis]  # shape: (snaps, ..., dim)
                    Q_gt = Q_gt.reshape(Q_gt.shape[0], -1)
                    RMS_gt = np.sqrt(np.mean(Q_gt**2, axis=0)) 
                    RMS_gt = RMS_gt.reshape(nx, ny, nz, 3) 
                else:
                    mean_gt = f['mean'][:]
                    Q_gt = f['UV'][val_id[time_lag:]] - mean_gt[np.newaxis]  # shape: (snaps, ..., dim)
                    Q_gt = Q_gt.reshape(Q_gt.shape[0], -1)
                    RMS_gt = np.sqrt(np.mean(Q_gt**2, axis=0)) # shape: (snaps, ...)
                    RMS_gt = RMS_gt.reshape(nx, ny, 2)


                f.create_dataset('RMS_gt_'+self.latent_id + '_' + name+ source, data=RMS_gt)
                logger.info(f"RMS computed and saved to {gt_path}.")

        nx = self.l_config.nx_t
        ny = self.l_config.ny_t
        nz = self.l_config.nz_t if self.dim == 3 else 1
        RMS_gt = RMS_gt[:nx, :ny, :nz, :] if self.dim == 3 else RMS_gt[:nx, :ny, :]
        RMS_error = pl.l2_err_norm(RMS_gt, RMS_rec)
        logger.info(f"RMS error for {name} {source}: {100 * RMS_error:.4f}%")
        RMS_u_error = pl.l2_err_norm(RMS_gt[..., 0], RMS_rec[..., 0])
        RMS_v_error = pl.l2_err_norm(RMS_gt[..., 1], RMS_rec[..., 1])
        logger.info(f"RMS_u error for {name} {source}: {100 * RMS_u_error:.4f}%")
        logger.info(f"RMS_v error for {name} {source}: {100 * RMS_v_error:.4f}%")
        if self.dim == 3:
            RMS_w_error = pl.l2_err_norm(RMS_gt[..., 2], RMS_rec[..., 2])
            logger.info(f"RMS_w error for {name} {source}: {100 * RMS_w_error:.4f}%")
                