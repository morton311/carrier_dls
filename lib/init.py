from __future__ import annotations
from typing import Tuple
import logging
logger = logging.getLogger(__name__)

class pathsBib:
    """Stores all filesystem paths used by the runner."""

    def __init__(self, config: dict) -> None:
        """
        Initialize paths.
        """
        from pathlib import Path

        ############################
        # Data info initialization #
        ############################
        self.data_dir = 'data/'
        data_sources = ['train_data', 'eval_data']
        data_dict = {}
        for data_source in data_sources:
            data_dict[data_source] = {}
        self.data_dict = data_dict
        ##############################
        # Latent info initialization #
        ##############################

        source_name = config['latent_params'].get('source_name', None)
        if source_name is None:
            logger.info("Error: No source name specified in latent_params")
        source_path = config['latent_params'].get('source_path', None)
        if source_path is None:
            logger.info("Error: No source path specified in latent_params")

        if config['latent_params']['type'] == 'dls':
            self.latent_id = source_name + '_dls_p'  + str(config['latent_params']['patch_size']) + 'm' + str(config['latent_params']['num_modes'])
        elif config['latent_params']['type'] == 'bvae':
            filters_str = '_'.join(str(f) for f in config['latent_params']['filters'])
            lins_str = '_'.join(str(l) for l in config['latent_params']['linear'])
            
            self.latent_id = source_name + '_bvae_l' + str(config['latent_params']['latent_dim']) + '_b' + str(config['latent_params']['beta']) + '_f_' + filters_str + '_lin_' + lins_str
        elif config['latent_params']['type'] == 'pod':
            self.latent_id = source_name + '_pod_m' + str(config['latent_params']['num_modes'])
        else:
            self.latent_id = source_name + '_pod'
        self.latent_id = self.latent_id.replace('.', '_')

        self.source_path = self.data_dir + source_path + '.h5'
        

        #############################
        # Other path initialization #
        #############################
        self.model_id = config['name']
        self.model_id = self.model_id.replace('.', '_')
        self.config_dir = 'configs/'
        

        self.latent_dir = 'results/' + self.latent_id + '/'
        self.latent_path = self.latent_dir + 'latent_coeff.h5'
        self.latent_config_path = self.latent_dir + 'latent_config.pkl'
        self.latent_model_path = self.latent_dir + 'latent_model.pth'
        self.model_dir = self.latent_dir + self.model_id + '/'
        self.log_dir = self.model_dir + 'logs/'
        self.log_path = self.log_dir + config['mode'] + '.log'
        self.model_path = self.model_dir + 'model.pth'
        self.checkpoint_dir = self.model_dir + 'checkpoints/'
        self.checkpoint_path = self.checkpoint_dir + 'checkpoint.tar'
        self.pred_dir = self.model_dir + 'pred/'
        self.fig_dir = self.model_dir + 'figs/'
        self.anim_dir = self.model_dir + 'anim/'
        self.metrics_dir = self.model_dir + 'saved_metrics/'


def init_path(config: dict) -> Tuple[bool, pathsBib]:
    """Create all required directories and return a populated pathsBib.

    Returns:
        (is_init_path, paths_bib): True if all directories were created successfully.
    """
    import os
    from pathlib import Path

    paths_bib = pathsBib(config)
    is_init_path = False
    try:
        path_list = [v for k, v in paths_bib.__dict__.items()
                     if isinstance(v, str) and '/' in v and '_dir' in k]
        for pth in path_list:
            Path(pth).mkdir(exist_ok=True, parents=True)
        is_init_path = True
    except Exception as exc:
        logger.info(f"Error: Failed to create full path list: {exc}")

    return is_init_path, paths_bib