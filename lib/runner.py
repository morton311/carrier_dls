import os
import sys
# from directory_tree import DisplayTree
import h5py
import pickle
import copy

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import time
import builtins
from functools import partial
print = partial(print, flush=True)
builtins.print = print

import lib.init as init
import lib.pod as pod
import lib.models as models
import lib.datas as datas
import lib.plotting as pl


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

        print(self.dim)
        if self.dim == 3:
            print("Using 3D DLS for latent coefficient computation")
            import lib.dls as dls
        else:
            print("Using 2D DLS for latent coefficient computation")
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
        if config['mode'] != 'compare':
            if config['log'] == 'file':
                print(f"To follow the log in real time, run 'tail -f {paths.log_path}'")
                sys.stdout = open(paths.log_path, 'w')
                sys.stderr = open(paths.log_path, 'a')
        

        print(f'Using device: {self.device}')
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
        print(f"{'#'*20} Configuration {'#'*20}")
        for key, val in self.config.items():
            if isinstance(val, dict):
                print(f"{key}:")
                for sub_key, sub_val in val.items():
                    print(f"  {sub_key}: {sub_val}")
            else:
                print(f"{key}: {val}")

        print(f"{'#'*48}")


    def _get_data(self):
        """
        Load the latent coefficients
        """
        print(f"{'#'*20}\t{'Loading data...':<20}\t{'#'*20}")
    
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
            print(f"DOF scaler loaded from {scaler_path}")
        else:
            print(f"No DOF scaler found at {scaler_path}. Will compute before training.")


        

    def _compute_latent_coefficients(self):
        
        print("Computing latent coefficients...")
        # compute the latent coefficients from source data
        print(f"Source path: {self.paths_bib.source_path}")
        
        if self.config['latent_params']['type'] == 'dls':
            # Check if latent_path exists and contains the source_name group
            if os.path.exists(self.paths_bib.latent_path):
                with h5py.File(self.paths_bib.latent_path, 'r+') as f:
                    if self.config['latent_params']['source_name'] in f:
                        print(f"Latent coefficients for source {self.config['latent_params']['source_name']} already exist in {self.paths_bib.latent_path}, skipping computation.")
                    else:
                        print(f"Latent coefficients for source {self.config['latent_params']['source_name']} not found in {self.paths_bib.latent_path}, computing...")
                        latent_config = self.dls.gfem_compress_flexible(
                            data_source = self.paths_bib.source_path,
                            field_name = 'UV',
                            group_name = self.config['latent_params']['source_name'],
                            patch_size = self.config['latent_params']['patch_size'],
                            num_modes = self.config['latent_params']['num_modes'],
                            latent_target = self.paths_bib.latent_path,
                            batch_size = self.config['latent_params']['batch_size'],
                        )
                        with open(self.paths_bib.latent_config_path, 'wb') as f:
                            pickle.dump(latent_config, f)
                        print("Latent coefficient config saved")
            else:
                print(f"Latent file {self.paths_bib.latent_path} not found, computing latent coefficients for source {self.config['latent_params']['source_name']}...")
                latent_config = self.dls.gfem_compress_flexible(
                    data_source = self.paths_bib.source_path,
                    field_name = 'UV',
                    group_name = self.config['latent_params']['source_name'],
                    patch_size = self.config['latent_params']['patch_size'],
                    num_modes = self.config['latent_params']['num_modes'],
                    latent_target = self.paths_bib.latent_path,
                    batch_size = self.config['latent_params']['batch_size'],
                )
            
        
                with open(self.paths_bib.latent_config_path, 'wb') as f:
                    pickle.dump(latent_config, f)
                print("Latent coefficient config saved")

        # load latent_config for use in computing latent coefficients for train and eval data
        with open(self.paths_bib.latent_config_path, 'rb') as f:
            latent_config = pickle.load(f)

        for data_source in self.data_sources:
            print(f"Processing data source {data_source} for latent coefficient computation...")
            if self.config[data_source] is not None:
                for id, source in enumerate(self.config[data_source]):
                    source_config = source
                    path = source.get('path')
                    path = self.paths_bib.data_dir + path + '.h5'
                    data_name = source.get('name')
                    
                    if path == self.paths_bib.source_path:
                        print(f"Source data {self.config['latent_params']['source_name']} found in {data_source}, skipping latent coefficient computation for this data.")
                        continue
                    else:
                        # Check if data_name group already exists in latent file
                        with h5py.File(self.paths_bib.latent_path, 'r') as f:
                            if data_name in f:
                                print(f"Group {data_name} already exists in latent file, skipping computation.")
                                continue
                        
                        print(f"Computing latent coefficients for {data_source} {data_name}...")
                        if self.config['latent_params']['type'] == 'dls':
                            self.dls.gfem_compress_flexible(
                                data_source = path,
                                field_name = 'UV',
                                group_name = data_name,
                                patch_size = self.config['latent_params']['patch_size'],
                                num_modes = self.config['latent_params']['num_modes'],
                                latent_target = self.paths_bib.latent_path,
                                batch_size = self.config['latent_params']['batch_size'],
                                dls_config = latent_config
                            )
        

    def _latent_split(self):

        # Load existing splits if available
        if os.path.exists(self.paths_bib.model_dir + 'split_ids.pkl'):
            with open(self.paths_bib.model_dir + 'split_ids.pkl', 'rb') as f:
                snaps = pickle.load(f)
                print(f"Train, test, and validation indices loaded from {self.paths_bib.model_dir + 'split_ids.pkl'}")
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
                            print(f"Splits already exist for {data_source} '{data_name}', skipping")
                            continue
                        
                        # Compute splits for this combination
                        print(f"Computing splits for {data_source} '{data_name}...")
                        if data_name not in snaps[data_source]:
                            snaps[data_source][data_name] = {}
                        
                        snaps[data_source][data_name]['total'] = f[data_name]['dof_u'].shape[0]
                        print(f"Total snapshots for {data_source} '{data_name}': {snaps[data_source][data_name]['total']}")

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
        print(f"Train, test, and validation indices saved to {self.paths_bib.model_dir + 'split_ids.pkl'}")
            
        with h5py.File(self.paths_bib.latent_path, 'r') as f:
            latent_keys = list(f.keys())
            if self.config['latent_params']['type'] == 'dls':
                if self.config['latent_params'].get('localized', False):
                    input_dim = self.dim * self.l_config.dof_elem
                else:
                    input_dim = self.dim * self.l_config.num_gfem_elems * self.l_config.dof_node
        print(f"Input dimension for model: {input_dim}")
        self.config['model_params']['input_dim'] = input_dim
        self.indices = snaps
            

    def _split_indices(self, total_snaps, train_split=0.8, test_split=0.1, sample_train=0, sample_test=0):
        # find indices for train, test, and validation sets
        train_len = int(total_snaps * train_split)
        test_len = int(train_len * test_split)

        train_indices = np.arange(0, train_len - test_len)
        test_indices = np.arange(train_len - test_len, train_len)
        val_indices = np.arange(train_len, total_snaps)

        print(f"{'Set':<12}|{'Total':<10}|{'First Idx':<12}|{'Last Idx':<12}|{'Sampled':<10}")
        print("-" * 56)
        print(f"{'Train':<12}|{train_len:<10}|{train_indices[0]:<12}|{train_indices[-1]:<12}|{sample_train:<10}")
        print(f"{'Test':<12}|{len(test_indices):<10}|{test_indices[0]:<12}|{test_indices[-1]:<12}|{sample_test:<10}")
        print(f"{'Validation':<12}|{len(val_indices):<10}|{val_indices[0]:<12}|{val_indices[-1]:<12}|{'-':<10}")

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
            print(f"Train, test, and validation indices saved to {self.paths_bib.model_dir + 'split_ids.pkl'}")

        else:
            with open(self.paths_bib.model_dir + 'split_ids.pkl', 'rb') as f:
                indices = pickle.load(f)
            print(f"Train, test, and validation indices loaded from {self.paths_bib.model_dir + 'split_ids.pkl'}")
        
        return indices
    
    def _get_model(self):
        """
        Load the model
        """
        # Load the model
        print(f"{'#'*20}\t{'Loading model...':<20}\t{'#'*20}")
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
            
        elif self.config['model_params']['model_type'] == 'f_extrap':
            self.model = None
        else:
            raise ValueError(f"Model {self.config['model_params']['model_type']} not recognized. Please use 'tr_enc' or 'lstm'.")
        
        # Load the model weights if they exist and overwrite is not set to 'l' or 'm'
        if os.path.exists(self.paths_bib.model_path) and not self.config['overwrite'] in ['l', 'm']:
            weights = torch.load(self.paths_bib.model_path, weights_only=True, map_location=self.device)
            if not self.config['distributed']:
                weights = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in weights.items()}
            self.model.load_state_dict(weights)
        if self.model is not None:
            self.num_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model initialized with {self.num_params} parameters")

        if self.config['distributed']:
            from torch.nn.parallel import DistributedDataParallel as DDP
            local_rank = int(os.environ["LOCAL_RANK"]) # automatically set by torchrun
            device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank)
            print("Model wrapped in DistributedDataParallel for distributed training")
        else:
            self.model = self.model.to(self.device)

    def _compile_model(self):
        """
        Compile the model
        """
        # Define the loss function and optimizer
        print(f"{'#'*20}\t{'Compiling model...':<20}\t{'#'*20}")

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
            self.scaler = torch.amp.GradScaler()

            print(f"Loss function: {self.criterion}")
            print(f"Optimizer: {self.optimizer}")
            print(f"Scheduler: ReduceLROnPlateau (factor={self.config['model_params'].get('lr_factor', 0.5)}, patience={self.config['model_params'].get('lr_patience', 5)})")
            print(f"Using mixed precision training with GradScaler: {self.scaler}")

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
                print(f"Loading checkpoint from {self.paths_bib.checkpoint_path}")
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
                print(f"Checkpoint loaded")
                self.checkpointed = True

            # if model exists and overwrite is not set to 'l' or 'm', load the model and skip training
            elif model_flag and not self.config['overwrite'] in ['l', 'm']:
                print(f"Model already exists at {self.paths_bib.model_path}. Skipping training.")
                print(f"Loading model weights from {self.paths_bib.model_path}")
                state_dict = torch.load(self.paths_bib.model_path, weights_only=True, map_location=self.device)
                state_dict = remap_embed_keys(state_dict)
                if not self.config['distributed']:
                    state_dict = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in state_dict.items()}
                self.model.load_state_dict(state_dict)
                self.checkpointed = False
            else:
                print(f"Model does not exist at {self.paths_bib.model_path}. Training from scratch.")
                self.checkpointed = False



    def train(self):
        """
        Train the model
        """
        # Load the latent coefficients
        print(f"{'#'*20}\t{'Training model...':<20}\t{'#'*20}")
        self._get_train_data()
        self._model_fit()


    def _get_train_data(self):
        """
        Get training and test data as torch tensors, minimizing memory usage.
        """
        if self.model is not None:
            print('Getting training and test data')
            tl = self.config['model_params']['time_lag']
            ta = self.config['model_params']['train_ahead']
            dof_dim = self.config['model_params']['input_dim']
            localized = self.config['latent_params'].get('localized', False)

            if localized:
                dof_elem = self.l_config.dof_elem
                skipx = self.config['latent_params'].get('skipx', 1)
                skipy = self.config['latent_params'].get('skipy', 1)
                skipz = self.config['latent_params'].get('skipz', 1) if self.dim == 3 else 1

                # print l_config attributes
                print(f"l_config attributes:")
                for attr in dir(self.l_config):
                    if not attr.startswith('modemat') and not attr.startswith('_'):
                        print(f"  {attr}: {getattr(self.l_config, attr)}")

                IJK = self.dls.node_map()

            self.dof_mean = {}
            self.dof_std = {}
            self.train_loader = {}
            self.test_loader = {}
            self.sampler = {}

            with h5py.File(self.paths_bib.latent_path, 'r') as f:
                for id, source in enumerate(self.config['train_data']):
                    name = source.get('name')
                    print(f"indices keys {self.indices['train_data'].keys()}")
                    train_indices = self.indices['train_data'][name]['train_indices']
                    test_indices = self.indices['train_data'][name]['test_indices']

                    if self.config['latent_params']['type'] == 'dls':
                        dof_u = f[name]['dof_u']
                        dof_v = f[name]['dof_v']
                        dof_w = f[name]['dof_w'] if self.dim == 3 else None

                        if localized:
                            lltogl_mat = self._compute_lltogl_mat(skipx=skipx, 
                                                                  skipy=skipy, 
                                                                  skipz=skipz)
                            num_unique_elems = lltogl_mat.shape[0]

                            # Compute mean and std: loop over time snapshots (not elements)
                            dofs = torch.zeros(len(train_indices), num_unique_elems, self.dim * dof_elem)
                            if self.dim == 3:
                                for t, idx in enumerate(train_indices):
                                    u_row = np.array(dof_u[idx, :])
                                    v_row = np.array(dof_v[idx, :])
                                    w_row = np.array(dof_w[idx, :])

                                    u_sel = u_row[lltogl_mat]
                                    v_sel = v_row[lltogl_mat]
                                    w_sel = w_row[lltogl_mat]

                                    dofs_cat = np.concatenate([u_sel, v_sel, w_sel], axis=1)
                                    dofs[t] = torch.from_numpy(dofs_cat).float()
                            else:
                                for t, idx in enumerate(train_indices):
                                    u_row = np.array(dof_u[idx, :])
                                    v_row = np.array(dof_v[idx, :])

                                    u_sel = u_row[lltogl_mat]
                                    v_sel = v_row[lltogl_mat]

                                    dofs_cat = np.concatenate([u_sel, v_sel], axis=1)
                                    dofs[t] = torch.from_numpy(dofs_cat).float()
                        else:
                            num_dofs = dof_u.shape[1]
                            dofs = torch.zeros(len(train_indices), self.dim * num_dofs)
                            if self.dim == 3:
                                for t, idx in enumerate(train_indices):
                                    u_row = np.array(dof_u[idx, :])
                                    v_row = np.array(dof_v[idx, :])
                                    w_row = np.array(dof_w[idx, :])
                                    dofs_cat = np.concatenate([u_row, v_row, w_row], axis=0)
                                    dofs[t] = torch.from_numpy(dofs_cat).float()
                            else:
                                for t, idx in enumerate(train_indices):
                                    u_row = np.array(dof_u[idx, :])
                                    v_row = np.array(dof_v[idx, :])
                                    dofs_cat = np.concatenate([u_row, v_row], axis=0)
                                    dofs[t] = torch.from_numpy(dofs_cat).float()

                    self.dof_mean[name] = torch.mean(dofs, dim=(0, 1))  # scalar normalization
                    self.dof_std[name] = torch.std(dofs, dim=(0, 1))
                    print(f"Mean/std shapes: {self.dof_mean[name].shape}, {self.dof_std[name].shape}")

                    # Helper to get normalized dof sequence as torch tensor (vectorized batch read)
                    def get_dof_seq(idx, length, latent_type='dls'):
                        if latent_type == 'dls':
                            u_rows = np.array(dof_u[idx:idx+length, :])
                            v_rows = np.array(dof_v[idx:idx+length, :])
                            w_rows = np.array(dof_w[idx:idx+length, :]) if self.dim == 3 else None

                            if localized:
                                u_sel = u_rows[:, lltogl_mat]
                                v_sel = v_rows[:, lltogl_mat]
                                w_sel = w_rows[:, lltogl_mat] if self.dim == 3 else None

                                if self.dim == 3:
                                    dofs_cat = np.concatenate([u_sel, v_sel, w_sel], axis=2)
                                else:
                                    dofs_cat = np.concatenate([u_sel, v_sel], axis=2)
                            else:
                                if self.dim == 3:
                                    dofs_cat = np.concatenate([u_rows, v_rows, w_rows], axis=1)
                                else:
                                    dofs_cat = np.concatenate([u_rows, v_rows], axis=1)

                            dof = torch.from_numpy(dofs_cat).float()

                        dof = (dof - self.dof_mean[name]) / self.dof_std[name]
                        return dof

                    if localized:
                        num_samples = num_unique_elems
                        X_train = torch.zeros(len(train_indices) * num_samples, tl, dof_dim)
                        Y_train = torch.zeros(len(train_indices) * num_samples, ta, dof_dim)
                        X_test = torch.zeros(len(test_indices) * num_samples, tl, dof_dim)
                        Y_test = torch.zeros(len(test_indices) * num_samples, ta, dof_dim)

                        for t, idx in enumerate(train_indices):
                            dof_seq = get_dof_seq(idx, tl + ta, latent_type=self.config['latent_params']['type'])
                            for iind in range(num_samples):
                                X_train[t*num_samples + iind] = dof_seq[:tl, iind, :]
                                Y_train[t*num_samples + iind] = dof_seq[tl:tl+ta, iind, :]

                        for t, idx in enumerate(test_indices):
                            dof_seq = get_dof_seq(idx, tl + ta, latent_type=self.config['latent_params']['type'])
                            for iind in range(num_samples):
                                X_test[t*num_samples + iind] = dof_seq[:tl, iind, :]
                                Y_test[t*num_samples + iind] = dof_seq[tl:tl+ta, iind, :]
                    else:
                        X_train = torch.zeros(len(train_indices), tl, dof_dim)
                        Y_train = torch.zeros(len(train_indices), ta, dof_dim)
                        X_test = torch.zeros(len(test_indices), tl, dof_dim)
                        Y_test = torch.zeros(len(test_indices), ta, dof_dim)

                        for t, idx in enumerate(train_indices):
                            dof_seq = get_dof_seq(idx, tl + ta, latent_type=self.config['latent_params']['type'])
                            X_train[t] = dof_seq[:tl, :]
                            Y_train[t] = dof_seq[tl:tl+ta, :]

                        for t, idx in enumerate(test_indices):
                            dof_seq = get_dof_seq(idx, tl + ta, latent_type=self.config['latent_params']['type'])
                            X_test[t] = dof_seq[:tl, :]
                            Y_test[t] = dof_seq[tl:tl+ta, :]

                    print(f"Data loaded for {name}. Shapes before stacking: X_train {X_train.shape}, Y_train {Y_train.shape}, X_test {X_test.shape}, Y_test {Y_test.shape}")
                    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}, dtype: {X_train.dtype}")
                    print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}, dtype: {X_test.dtype}")

                    if self.config['distributed']:
                        self.train_loader[name], self.sampler[name] = datas.make_dataloader(X_train, Y_train, batch_size=self.config['model_params']['batch_size'], shuffle=True, distributed=True)
                    else:
                        self.train_loader[name] = datas.make_dataloader(X_train, Y_train, batch_size=self.config['model_params']['batch_size'], shuffle=True)

                    print(f"Train loader created with {len(self.train_loader[name])} batches")

                    self.test_loader[name] = datas.make_dataloader(X_test, Y_test, batch_size=self.config['model_params']['batch_size'], shuffle=False)
                    print(f"Test loader created with {len(self.test_loader[name])} batches")

                    with open(os.path.join(self.paths_bib.model_dir, 'dof_scaler.pkl'), 'wb') as f:
                        pickle.dump((self.dof_mean, self.dof_std), f)


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
            
            max_norm = 0.2
            new_lr = self.scheduler.get_last_lr() if self.scheduler is not None else self.config['model_params']['lr']
            if new_lr is list:
                new_lr = new_lr[0]


            for epoch in range(len(losses), self.config['model_params']['num_epochs']):
                self.model.train()
                epoch_loss = 0
                

                ## --------------------------------------- Train ---------------------------------------
                for key in self.train_loader: 
                    if self.config['distributed']:
                        self.sampler[key].set_epoch(epoch)  # set epoch for distributed sampler if using distributed training
                    for inputs, targets in self.train_loader[key]: 
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        self.optimizer.zero_grad()
                        total_loss = 0.0
                        

                        for n in range(targets.shape[1]):
                            print(f"Step {n+1}/{targets.shape[1]}, VRAM usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                            target = targets[:, n, :]  # shape: [B, input_dim]

                            # Forward pass
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                outputs = self.model(inputs)  # shape: [B, input_dim]
                                loss = self.criterion(outputs, target)

                            # Backward and optimization for current step only
                            total_loss += loss
                            self.scaler.scale(loss).backward()

                            # Prepare input for next step
                            inputs = torch.cat((inputs[:, 1:, :], outputs.detach().unsqueeze(1)), dim=1)

                        epoch_loss += total_loss.item() / (targets.shape[1] * len(self.train_loader[key]))
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                losses.append(epoch_loss)

                ## --------------------------------------- Test ---------------------------------------
                # Evaluate the model on the test set
                self.model.eval()
                test_loss = 0
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

                test_losses.append(test_loss)
                
                # Step scheduler based on test loss (plateau detection)
                self.scheduler.step(test_loss)
                new_lr = self.scheduler.get_last_lr()[0] if isinstance(new_lr, list) else new_lr
                
                
                ## ------------------------------- Early stop and Checkpoint -------------------------------
                # Early stopping and saving the best model
                if epoch > 1:
                    if np.isnan(test_losses[-1]) or np.isnan(losses[-1]):
                        print(f'NaN loss at epoch {epoch+1}. Stopping training.')
                        self.model.load_state_dict(best_model)
                        break
                    elif test_loss < best_test_loss:
                        best_test_loss = test_loss
                        best_model = copy.deepcopy(self.model.state_dict())
                        best_epoch = epoch + 1
                        # print(f'Best model saved at epoch {best_epoch} with test loss: {best_test_loss:.4f}')
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= self.config['model_params']['patience']:
                            print(f'Early stopping at epoch {epoch+1}')
                            self.model.load_state_dict(best_model)
                            print(f'Best model loaded from epoch {best_epoch}, with test loss: {best_test_loss:.4f}')
                            break

                    if (epoch + 1) % 5 == 0:
                        # Save model checkpoint every 5 epochs
                        # Save model losses, current weights, best weights, and optimizer state 

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
                        # print(f"Checkpoint saved at epoch {epoch+1} to {self.paths_bib.checkpoint_path}")
                    
                best_flag = 'X' if (epoch + 1) == best_epoch else ' '
                checkpoint_flag = 'X' if (epoch + 1) % 5 == 0 else ' '
                print(f"| Epoch: {epoch+1:<4}/{self.config['model_params']['num_epochs']:<4} | Train Loss: {losses[-1]:7.4f} | Test Loss: {test_losses[-1]:7.4f} | Best: {best_flag:<1} | Patience: {early_stop_counter:<3}/{self.config['model_params']['patience']} | Checkpoint: {checkpoint_flag:<1} | LR: {new_lr:.2e} | Time: {(time.time() - start_time)/60:10.2f} min |")

            end_time = time.time()
            print('\n\nTime taken for training: ', end_time - start_time)
            print('Time taken per epoch: ', (end_time - start_time) / (epoch + 1))

            # Save the final model after training
            torch.save(self.model.state_dict(), self.paths_bib.model_path)
            print(f"Final model saved to {self.paths_bib.model_path}")
            # Save the training and test losses
            with open(self.paths_bib.model_dir + 'losses.pkl', 'wb') as f:
                pickle.dump({'train_losses': losses, 'test_losses': test_losses}, f)

            print(f"Training and test losses saved to {self.paths_bib.model_dir + 'losses.pkl'}")

            print('\nTraining complete')

        else:
            print("No training required for f_extrap model")
            self.pred()

    def _compute_lltogl_mat(self, skipx=1, skipy=1, skipz=1):
        nx = self.l_config.nx_g
        ny = self.l_config.ny_g
        nz = self.l_config.nz_g if self.dim == 3 else 1
        dof_node = self.l_config.dof_node
        num_unique_elems = (nx//skipx) * (ny//skipy) * (nz//skipz) if self.dim == 3 else (nx//skipx) * (ny//skipy)
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
        # print(f"Precomputed lltogl_mat shape: {lltogl_mat.shape}")
        return lltogl_mat

    def _predict_dofs_forward(self, name, dof_u, dof_v, dof_w=None, total_steps=None):
        """
        Predict dof_u/dof_v(/dof_w) forward in time from initial conditions.
        Inputs dof_u/dof_v(/dof_w) are expected to contain at least `time_lag` snapshots.
        Returns full predicted trajectories with shape (total_steps, num_dofs).
        """
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

        lltogl_mat = self._compute_lltogl_mat()
        num_elems = lltogl_mat.shape[0]

        if self.dim == 3:
            u_sel = init_u[:, lltogl_mat]
            v_sel = init_v[:, lltogl_mat]
            w_sel = init_w[:, lltogl_mat]
            dof_input = np.transpose(np.concatenate([u_sel, v_sel, w_sel], axis=2), (1, 0, 2))
        else:
            u_sel = init_u[:, lltogl_mat]
            v_sel = init_v[:, lltogl_mat]
            dof_input = np.transpose(np.concatenate([u_sel, v_sel], axis=2), (1, 0, 2))

        dof_input = (dof_input - self.dof_mean[name].numpy()) / self.dof_std[name].numpy()
        dof_input = torch.from_numpy(dof_input).float().to(self.device)

        self.model.eval()
        predictions = np.zeros((num_elems, total_steps, self.dim * self.l_config.dof_elem))
        predictions[:, :time_lag, :] = dof_input.cpu().numpy()

        with torch.no_grad():
            for t in range(total_steps - time_lag):
                output = self.model(dof_input)
                predictions[:, t + time_lag, :] = output.cpu().numpy()
                dof_input = torch.cat((dof_input[:, 1:, :], output.unsqueeze(1)), dim=1)

        predictions = predictions * self.dof_std[name].numpy() + self.dof_mean[name].numpy()

        num_dofs = self.l_config.num_gfem_nodes * self.l_config.dof_node
        ndof = self.l_config.dof_elem

        dof_u_pred = np.zeros((total_steps, num_dofs))
        dof_v_pred = np.zeros((total_steps, num_dofs))
        dof_w_pred = np.zeros((total_steps, num_dofs)) if self.dim == 3 else None

        for i in range(num_elems):
            dof_u_pred[:, lltogl_mat[i]] = predictions[i, :, :ndof]
            dof_v_pred[:, lltogl_mat[i]] = predictions[i, :, ndof:2*ndof]
            if self.dim == 3:
                dof_w_pred[:, lltogl_mat[i]] = predictions[i, :, 2*ndof:]

        return dof_u_pred, dof_v_pred, dof_w_pred


    def pred(self):
        """
        Make predictions over the validation data with the model for all sets in config['eval_data']
        """
        print(f"{'#'*20}\t{'Predicting...':<20}\t{'#'*20}")
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


            print(f"Predicting for {name} from {path}...")
            with h5py.File(self.paths_bib.latent_path, 'r') as f:
                dof_u = f[name]['dof_u'][init_id, :]
                dof_v = f[name]['dof_v'][init_id, :]
                dof_w = f[name]['dof_w'][init_id, :] if self.dim == 3 else None
            print(f"Predicting for {name} {source_group}")

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
            print(f"Predictions saved to {pred_path}")
            
            # Many short horizon predictions for error growth analysis
            horizon = self.config['model_params'].get('horizon', 10)
            max_horizons = 30
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
            horizon_errors = self._l2_err_norm(dof_true_horizons, dof_pred_horizon, axis=2)
            with h5py.File(pred_path, 'a') as f:
                f.create_dataset(f'horizon_errors_{name}_{source_group}', data=horizon_errors)




        print("Prediction complete")
        print(f"\nReconstructing all predictions")
        self._pred_rec()


    def _pred_rec(self):
        """
        Reconstruct the full field predictions from the predicted dofs and save to HDF5.
        """
        print(f"{'#'*20}\t{'Reconstructing predictions...':<20}\t{'#'*20}")
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
            print(f"Reconstructing for {name} from {pred_path}...")
            with h5py.File(pred_path, 'r') as f:
                dof_u_pred = f['dof_u'][:]
                dof_v_pred = f['dof_v'][:]
                dof_w_pred = f['dof_w'][:] if self.dim == 3 else None

            if self.dim == 3:
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
    def _l2_err_norm(self, true, pred, axis=None):
        """
        Compute the L2 norm between two arrays.
        """
        return np.linalg.norm(true - pred, axis=axis) / np.linalg.norm(true, axis=axis)

    def eval(self):
        """
        Evaluate the model predictions against the ground truth for all sets in config['eval_data'], save metrics, then generate plots
        """
        
        print(f"{'#'*20}\t{'Evaluating...':<20}\t{'#'*20}")
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
            print(f"Evaluating for {name} between {rec_path} and {gt_path}...")
            
            # Compute metrics
            self.compute_TKE(rec_path, gt_path, name, source_group, ids)
            self.compute_RMS(rec_path, gt_path, name, source_group, ids)

            pl.plot_TKE(self, rec_path, gt_path, name, source_group,  ids)
            pl.plot_RMS(self, rec_path, gt_path, name, source_group, z_slice=z_slice)
            pl.plot_RMS(self, rec_path, gt_path, name, source_group, y_slice=y_slice)
            pl.plot_RMS(self, rec_path, gt_path, name, source_group, x_slice=x_slice)

            # pl.plot_slice_compare(self, rec_path, gt_path, name, ids, 1,  z_slice=z_slice)
            # pl.plot_slice_compare(self, rec_path, gt_path, name, ids, 30,  z_slice=z_slice)

            # pl.plot_slice_compare(self, rec_path, gt_path, name, ids, 1,  y_slice=y_slice)
            # pl.plot_slice_compare(self, rec_path, gt_path, name, ids, 30,  y_slice=y_slice)
            
            # pl.plot_slice_compare(self, rec_path, gt_path, name, ids, 1,  x_slice=x_slice)
            # pl.plot_slice_compare(self, rec_path, gt_path, name, ids, 30,  x_slice=x_slice)
            
            # pl.q_criterion(self, rec_path, gt_path, name, ids, 1)
            # pl.q_criterion(self, rec_path, gt_path, name, ids, 30)
            # pl.anim_q_criterion(self, rec_path, gt_path, name, ids)

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
                print(f"Weights already computed and found in {gt_path}. Loading weights...")
                weights = f['weights'][:]
            else:
                print(f"Weights not found in {gt_path}. Computing weights...")
                if self.dim == 3:
                    weights = wg.generate_weights_grid_3d(self.x_grid, self.y_grid, self.z_grid)
                else:
                    weights = wg.generate_weights_grid_2d(self.x_grid, self.y_grid)
                f.create_dataset('weights', data=weights)
                print(f"Weights computed and saved to {gt_path}.")
    
        if self.dim == 3:
            weights = weights[:nx, :ny, :nz]
        else:
            weights = weights[:nx, :ny]
            

        with h5py.File(rec_path, 'r+') as f:
            if 'TKE_rec' in f:
                print(f"TKE for reconstructed already computed and found in {rec_path}. Skipping computation...")
                TKE_rec = f['TKE_rec'][:]
            else:
                print(f"TKE for reconstructed not found in {rec_path}. Computing TKE...")
                Q_rec = f['Q_rec'][:] # shape: (snaps, ..., dim)
                Q_rec_weighted = Q_rec * weights[np.newaxis, ..., np.newaxis] # shape: (snaps, ..., dim)
                Q_rec_weighted = Q_rec_weighted.reshape(Q_rec_weighted.shape[0], -1) 
                
                TKE_rec = 0.5 * np.sum(Q_rec_weighted**2, axis=-1) # shape: (snaps, ...)
                f.create_dataset('TKE_rec', data=TKE_rec)

            
        

        with h5py.File(gt_path, 'r+') as f:

            if 'TKE_gt_'+self.latent_id + '_' + name + source in f:
                print(f"TKE for GT already computed and found in {gt_path}. Skipping computation...")
                TKE_gt = f['TKE_gt_'+self.latent_id + '_' + name + source][:]
            else:
                print(f"TKE for GT not found in {gt_path}. Computing TKE...")
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
        print(f"TKE error for {name} {source}: {100 * TKE_error:.4f}%")
                
            
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
                print(f"RMS for reconstructed already computed and found in {rec_path}. Skipping computation...")
                RMS_rec = f['RMS_rec'][:]
            else:
                print(f"RMS for reconstructed not found in {rec_path}. Computing RMS...")
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
                print(f"RMS for GT already computed and found in {gt_path}. Skipping computation...")
                RMS_gt = f['RMS_gt_'+self.latent_id + '_' + name+ source][:]
            else:
                print(f"RMS for GT not found in {gt_path}. Computing RMS...")
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
                print(f"RMS computed and saved to {gt_path}.")

        nx = self.l_config.nx_t
        ny = self.l_config.ny_t
        nz = self.l_config.nz_t if self.dim == 3 else 1
        RMS_gt = RMS_gt[:nx, :ny, :nz, :] if self.dim == 3 else RMS_gt[:nx, :ny, :]
        RMS_error = pl.l2_err_norm(RMS_gt, RMS_rec)
        print(f"RMS error for {name} {source}: {100 * RMS_error:.4f}%")
        RMS_u_error = pl.l2_err_norm(RMS_gt[..., 0], RMS_rec[..., 0])
        RMS_v_error = pl.l2_err_norm(RMS_gt[..., 1], RMS_rec[..., 1])
        print(f"RMS_u error for {name} {source}: {100 * RMS_u_error:.4f}%")
        print(f"RMS_v error for {name} {source}: {100 * RMS_v_error:.4f}%")
        if self.dim == 3:
            RMS_w_error = pl.l2_err_norm(RMS_gt[..., 2], RMS_rec[..., 2])
            print(f"RMS_w error for {name} {source}: {100 * RMS_w_error:.4f}%")
                