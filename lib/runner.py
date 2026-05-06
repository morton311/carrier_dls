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

        

        # get model info
        self._get_data()
        self._get_model()
        self._compile_model()

    def _init_paths_and_logging(self, config):
        is_init_path, paths = init.init_path(config)

        if config['mode'] != 'compare':
            if config['log'] == 'file':
                sys.stdout = open(paths.log_path, 'w')
                sys.stderr = open(paths.log_path, 'a')
        

        print(f'Using device: {self.device}')
        return paths
    

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
        self._latent_split()

        # load latent_config 
        with open(self.paths_bib.latent_path.replace('.h5', '_config.pkl'), 'rb') as f:
            self.l_config = pickle.load(f)

    def _compute_latent_coefficients(self):
        print("Computing latent coefficients...")
        # compute the latent coefficients from source data

        latent_config = dls.gfem_3d_compress_flexible(
                data_source = self.config['latent_params']['source_path'],
                field_name = 'UV',
                group_name = self.config['latent_params']['source_name'],
                patch_size = self.config['latent_params']['patch_size'],
                num_modes = self.config['latent_params']['num_modes'],
                latent_target = self.paths_bib.latent_path,
                batch_size = self.config['latent_params']['batch_size'],
            )
        
        with open(self.paths_bib.latent_path.replace('.h5', '_config.pkl'), 'wb') as f:
            pickle.dump(latent_config, f)
        print("Latent coefficient config saved")
        
        data_sources = ['train_data', 'eval_data']

        for data_source in data_sources:
            for key in self.config[data_source].keys():
                if self.config[data_source][key]['data_path'] == self.config['latent_params']['source_path']:
                    print(f"Source data {self.config['latent_params']['source_name']} found in {data_source}, skipping latent coefficient computation for this data.")
                    continue
                else:
                    print(f"Computing latent coefficients for {self.config[data_source][key]['data_name']}...")
                    dls.gfem_3d_compress_flexible(
                        data_source = self.config[data_source][key]['data_path'],
                        field_name = 'UV',
                        group_name = self.config[data_source][key]['data_name'],
                        patch_size = self.config['latent_params']['patch_size'],
                        num_modes = self.config['latent_params']['num_modes'],
                        latent_target = self.paths_bib.latent_path,
                        batch_size = self.config['latent_params']['batch_size'],
                        dls_config = latent_config
                    )
        