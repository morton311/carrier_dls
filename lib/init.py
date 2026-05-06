class pathsBib: 
    """
    Class to store paths.
    """ 
    def __init__(self, config):
        """
        Initialize paths.
        """
        from pathlib import Path

        ############################
        # Data info initialization #
        ############################
        data_sources = ['train_data', 'eval_data']
        for data_source in data_sources:
            for key in config[data_source].keys():
                data_path = config[data_source][key].get('data_path')
                data_name = config[data_source][key].get('data_name')

                if data_path:
                    data_path_obj = Path(data_path).expanduser()
                    config[data_source][key]['data_path'] = str(data_path_obj)
                    if not data_name:
                        config[data_source][key]['data_name'] = data_path_obj.stem
                else:
                    if not data_name:
                        raise ValueError(f"Config for {data_source} '{key}' must include either 'data_name' or 'data_path'.")
                    config[data_source][key]['data_path'] = self.data_dir + data_name + '.h5'
        
        ##############################
        # Latent info initialization #
        ##############################

        if config['latent_params'].get('source_path') or config['latent_params'].get('source_name'):
            if not config['latent_params'].get('source_name'):
                source_path_obj = Path(config['latent_params']['source_path']).expanduser()
                source = source_path_obj.stem
                config['latent_params']['source_name'] = source
            elif not config['latent_params'].get('source_path'):
                config['latent_params']['source_path'] = self.data_dir + config['latent_params']['source_name'] + '.h5'

        else:
            print("No source specified for latent parameters, using first train_data as source.")
            keys = config['train_data'].keys()
            source = config['train_data'][list(keys)[0]]['data_name']
            config['latent_params']['source_name'] = source
            config['latent_params']['source_path'] = config['train_data'][list(keys)[0]]['data_path']

        if config['latent_type'] == 'dls':
            self.latent_id = source + '_dls_p'  + str(config['latent_params']['patch_size']) + 'm' + str(config['latent_params']['num_modes'])
        elif config['latent_type'] == 'bvae':
            filters_str = '_'.join(str(f) for f in config['latent_params']['filters'])
            lins_str = '_'.join(str(l) for l in config['latent_params']['linear'])
            
            self.latent_id = source + '_bvae_l' + str(config['latent_params']['latent_dim']) + '_b' + str(config['latent_params']['beta']) + '_f_' + filters_str + '_lin_' + lins_str
        else:
            self.latent_id = source + '_pod' # + str(config['latent_params']['num_modes'])
        self.latent_id = self.latent_id.replace('.', '_')
        

        #############################
        # Other path initialization #
        #############################
        self.model_id = config['name']
        self.model_id = self.model_id.replace('.', '_')
        self.config_dir = 'configs/'
        self.data_dir = 'data/'

        self.latent_dir = 'results/' + self.latent_id + '/'
        self.latent_path = self.latent_dir + 'latent_coeff.h5'
        self.latent_model_path = self.latent_dir + 'latent_model.pth'
        self.model_dir = self.latent_dir + self.model_id + '/'
        self.log_dir = self.model_dir + 'logs/'
        self.log_path = self.log_dir + config['mode'] + '.log'
        self.model_path = self.model_dir + 'model.pth'
        self.checkpoint_dir = self.model_dir + 'checkpoints/'
        self.checkpoint_path = self.checkpoint_dir + 'checkpoint.tar'
        self.predictions_dir = self.model_dir + 'pred/'
        self.fig_dir = self.model_dir + 'figs/'
        self.anim_dir = self.model_dir + 'anim/'
        self.metrics_dir = self.model_dir + 'saved_metrics/'


def init_path(config):
    """
    Initialisation of all the paths 

    Returns:
        pathsBib        :   (class) class containing all the paths
        is_init_path    :   (bool) if initialise success
    """
    import os 
    from pathlib import Path
    
    paths_bib = pathsBib(config)
    is_init_path = False
    try:
        # print(f"{'#'*20}\t{'Init paths...':<20}\t{'#'*20}")
        path_list =[i for key,i in paths_bib.__dict__.items() if type(i)==str and "/" in i and '_dir' in key]
        for pth in path_list:
            Path(pth).mkdir(exist_ok=True, parents=True)
            # print(f"INIT:\t{pth}\tDONE")
            
        is_init_path = True
    except:
        print(f"Error: Failed to create full path list")


    return is_init_path, paths_bib