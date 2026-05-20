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
import lib.dls as dls
import lib.pod as pod
import lib.models as models
import lib.datas as datas


class runner(nn.Module):
    def __init__(self, config):
        super(runner, self).__init__()
        self.config = config
            
        self.device = config['device']
        self.paths_bib = self._init_paths_and_logging(config)

        self._log_config()

        self._get_grid()

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
                sys.stdout = open(paths.log_path, 'w')
                sys.stderr = open(paths.log_path, 'a')
        

        print(f'Using device: {self.device}')
        return paths
    
    def _get_grid(self):
        with h5py.File(self.paths_bib.source_path, 'r') as f:
            self.x_grid = f['x_grid'][:]
            self.y_grid = f['y_grid'][:]
            self.z_grid = f['z_grid'][:] if 'z_grid' in f else None

            self.x = self.x_grid.shape[0]
            self.y = self.y_grid.shape[0]
            self.z = self.z_grid.shape[0] if self.z_grid is not None else None
        
    def _log_config(self):
        print(f"{'#'*20} Configuration {'#'*20}")
        for key, val in self.config.items():
            if isinstance(val, dict):
                print(f"{key}:")
                for sub_key, sub_val in val.items():
                    print(f"  {sub_key}: {sub_val}")
            else:
                print(f"{key}: {val}")


    def _get_data(self):
        """
        Load the latent coefficients
        """
        print(f"{'#'*20}\t{'Loading data...':<20}\t{'#'*20}")
        
        if not os.path.exists(self.paths_bib.latent_path) or self.config['overwrite'] == 'l':
            self._compute_latent_coefficients()

        # load latent_config 
        with open(self.paths_bib.latent_config_path, 'rb') as f:
            self.l_config = pickle.load(f)

        self._latent_split()

        

    def _compute_latent_coefficients(self):
        print("Computing latent coefficients...")
        # compute the latent coefficients from source data
        print(f"Source path: {self.paths_bib.source_path}")

        if self.config['latent_params']['type'] == 'dls':
            latent_config = dls.gfem_3d_compress_flexible(
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
                        print(f"Computing latent coefficients for {data_source} {data_name}...")
                        if self.config['latent_params']['type'] == 'dls':
                            dls.gfem_3d_compress_flexible(
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

        if not os.path.exists(self.paths_bib.model_dir + 'split_ids.pkl'):

            with h5py.File(self.paths_bib.latent_path, 'r') as f:
                snaps = {}
                for data_source in self.data_sources:
                    snaps[data_source] = {}
                    if self.config[data_source] is not None:
                        for id, source in enumerate(self.config[data_source]):
                            path = source.get('path')
                            path = self.paths_bib.data_dir + path + '.h5'
                            data_name = source.get('name')
                            snaps[data_source][data_name] = {}
                            snaps[data_source][data_name]['total'] = f[data_name]['dof_u'].shape[0]
                            print(f"Total snapshots for {data_source} '{data_name}': {snaps[data_source][data_name]}")
                            print(f"Splitting data")

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
                
                with open(self.paths_bib.model_dir + 'split_ids.pkl', 'wb') as f:
                    pickle.dump(snaps, f)
                print(f"Train, test, and validation indices saved to {self.paths_bib.model_dir + 'split_ids.pkl'}")

        else:
            with open(self.paths_bib.model_dir + 'split_ids.pkl', 'rb') as f:
                snaps = pickle.load(f)
                print(f"Train, test, and validation indices loaded from {self.paths_bib.model_dir + 'split_ids.pkl'}")
            
        with h5py.File(self.paths_bib.latent_path, 'r') as f:
            latent_keys = list(f.keys())
            if self.config['latent_params']['type'] == 'dls':
                if self.config['latent_params'].get('localized', False):
                    input_dim = 3 * self.l_config['dof_elem'] 
                else:
                    input_dim = 3 * self.l_config['num_gfem_elems'] * self.l_config['dof_node']
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
            self.model.load_state_dict(torch.load(self.paths_bib.model_path, weights_only=True, map_location=self.device))
        if self.model is not None:
            print(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")

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
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config['model_params'].get('gamma', 0.9))

            print(f"Loss function: {self.criterion}")
            print(f"Optimizer: {self.optimizer}")

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
                checkpoint['model_state_dict'] = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in checkpoint['model_state_dict'].items()}
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'lr_scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                else: 
                    self.scheduler = None
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
            num_gfem_elems = self.l_config['num_gfem_elems']
            dof_node = self.l_config['dof_node']
            nx = self.l_config['nx_g']
            ny = self.l_config['ny_g']
            nz = self.l_config['nz_g']

            IJK = dls.node_map()

            self.dof_mean = {}
            self.dof_std = {}
            self.train_loader = {}
            self.test_loader = {}
            self.train_sampler = {}
            self.test_sampler = {}
            
            with h5py.File(self.paths_bib.latent_path, 'r') as f:
                for id, source in enumerate(self.config['train_data']):
                    name = source.get('name')
                    print(f"indices keys {self.indices['train_data'].keys()}")
                    train_indices = self.indices['train_data'][name]['train_indices']
                    test_indices = self.indices['train_data'][name]['test_indices']


                    if self.config['latent_params']['type'] == 'dls':
                        dof_u = f[name]['dof_u']
                        dof_v = f[name]['dof_v']
                        dof_w = f[name]['dof_w']
                        # Compute mean and std using only the training indices, in chunks to save memory
                        dofs = torch.zeros(len(train_indices), num_gfem_elems, dof_dim)
                        for kx in range(nx-1):
                             for ky in range(ny-1):
                                for kz in range(nz-1):
                                    iind = kx * (ny-1) * (nz-1) + ky * (nz-1) + kz
                                    lltogl = dls.build_lltogl(kx, ky, kz, ny, nz, dof_node, IJK)
                                    
                                    for t, idx in enumerate(train_indices):
                                        u = torch.from_numpy(dof_u[idx:idx+1, lltogl]).float()
                                        v = torch.from_numpy(dof_v[idx:idx+1, lltogl]).float()
                                        w = torch.from_numpy(dof_w[idx:idx+1, lltogl]).float()
                                        dofs[t, iind] = torch.cat((u, v, w), dim=1)
                            
                    self.dof_mean[name] = torch.mean(dofs, dim=(0, 1))
                    self.dof_std[name] = torch.std(dofs, dim=(0, 1))

                    

                    # Helper to get normalized dof sequence as torch tensor
                    def get_dof_seq(idx, length, latent_type='dls'):
                        if latent_type == 'dls':
                            u = torch.from_numpy(dof_u[idx:idx+length, lltogl]).float()
                            v = torch.from_numpy(dof_v[idx:idx+length, lltogl]).float()
                            w = torch.from_numpy(dof_w[idx:idx+length, lltogl]).float()
                            dof = torch.cat((u, v, w), dim=1)

                        dof = (dof - self.dof_mean[name]) / self.dof_std[name]
                        return dof

                    # Prepare lists for X/Y, then stack at the end
                    X_train = torch.zeros(len(train_indices) * num_gfem_elems, tl, dof_dim)
                    Y_train = torch.zeros(len(train_indices) * num_gfem_elems, ta, dof_dim)

                    X_test = torch.zeros(len(test_indices) * num_gfem_elems, tl, dof_dim)
                    Y_test = torch.zeros(len(test_indices) * num_gfem_elems, ta, dof_dim)

                    for kx in range(nx-1):
                        for ky in range(ny-1):
                            for kz in range(nz-1):
                                iind = kx * (ny-1) * (nz-1) + ky * (nz-1) + kz
                                lltogl = dls.build_lltogl(kx, ky, kz, ny, nz, dof_node, IJK)
                                for t, idx in enumerate(train_indices):
                                    dof_seq = get_dof_seq(idx, tl + ta, latent_type=self.config['latent_params']['type'])
                                    X_train[t*num_gfem_elems + iind] = dof_seq[:tl]
                                    Y_train[t*num_gfem_elems + iind] = dof_seq[tl:tl+ta]
                                
                                for t, idx in enumerate(test_indices):
                                    dof_seq = get_dof_seq(idx, tl + ta, latent_type=self.config['latent_params']['type'])
                                    X_test[t*num_gfem_elems + iind] = dof_seq[:tl]
                                    Y_test[t*num_gfem_elems + iind] = dof_seq[tl:tl+ta]
                                
                                print(f"Got data for element {iind} ({kx}, {ky}, {kz})")


                    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}, dtype: {X_train.dtype}")
                    print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}, dtype: {X_test.dtype}")

                    # convert to data loader
                    if self.config['distributed']:
                        self.train_loader[name], self.sampler[name] = datas.make_dataloader(X_train.to(self.device), Y_train.to(self.device), batch_size=self.config['model_params']['batch_size'], shuffle=True, distributed=True)
                    else:
                        self.train_loader[name] = datas.make_dataloader(X_train.to(self.device), Y_train.to(self.device), batch_size=self.config['model_params']['batch_size'], shuffle=True)

                    print(f"Train loader created with {len(self.train_loader[name])} batches")

                    self.test_loader[name] = datas.make_dataloader(X_test.to(self.device), Y_test.to(self.device), batch_size=self.config['model_params']['batch_size'], shuffle=False)
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

            for epoch in range(len(losses), self.config['model_params']['num_epochs']):
                self.model.train()
                epoch_loss = 0
                if self.config['distributed']:
                    self.sampler.set_epoch(epoch)  # set epoch for distributed sampler if using distributed training

                ## --------------------------------------- Train ---------------------------------------
                for key in self.train_loader: 
                    loader = self.train_loader[key]
                    for inputs, targets in loader: 
                        inputs, targets = inputs, targets
                        self.optimizer.zero_grad()
                        total_loss = 0.0

                        for n in range(targets.shape[1]):
                            # print(f"Step {n+1}/{targets.shape[1]}, VRAM usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                            target = targets[:, n, :]  # shape: [B, input_dim]

                            # Forward pass
                            outputs = self.model(inputs)  # shape: [B, input_dim]
                            loss = self.criterion(outputs, target)

                            # Backward and optimization for current step only
                            total_loss += loss
                            loss.backward()

                            # Prepare input for next step
                            inputs = torch.cat((inputs[:, 1:, :], outputs.detach().unsqueeze(1)), dim=1)

                        epoch_loss += total_loss.item() / (targets.shape[1] * len(loader))

                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                        self.optimizer.step()
                        self.scheduler.step()

                losses.append(epoch_loss)

                ## --------------------------------------- Test ---------------------------------------
                # Evaluate the model on the test set
                self.model.eval()
                test_loss = 0
                with torch.no_grad():
                    for key in self.test_loader:
                        loader = self.test_loader[key]
                        for inputs, targets in loader:
                            inputs, targets = inputs, targets
                            for n in range(targets.shape[1]):
                                target = targets[:, n, :]
                                outputs = self.model(inputs)
                                loss = self.criterion(outputs, target)
                                test_loss += loss.item() / (targets.shape[1] * len(loader))
                                inputs = torch.cat((inputs[:, 1:, :], outputs.unsqueeze(1)), dim=1)

                test_losses.append(test_loss)
                
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
                print(f"| Epoch: {epoch+1:<4}/{self.config['model_params']['num_epochs']:<4} | Train Loss: {losses[-1]:7.4f} | Test Loss: {test_losses[-1]:7.4f} | Best: {best_flag:<1} | Patience: {early_stop_counter:<3}/{self.config['model_params']['patience']} | Checkpoint: {checkpoint_flag:<1} |")

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