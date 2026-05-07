import argparse
import os
import shutil
import sys
import torch
import json
from pathlib import Path
import lib.init as init
from lib.runner import runner

parser = argparse.ArgumentParser()
parser.add_argument('-c', type=str, default='config.json', help='Name or whole path of config file, must be in json format')
parser.add_argument('-o', choices=['l', 'm', 'r', 'x'], default='x', help="Overwrite: 'l' for latent space, 'm' for model, 'r' for results, or None")
parser.add_argument('-m', choices=['train', 'eval', 'pred', 'test','latent', 'anim'], default='test', help="Mode: 'train' for training, 'eval' for evaluation")
parser.add_argument('-log', choices=['file', 'terminal'], default='file', help="Log output to file or terminal")
parser.add_argument('-d', choices=['True', 'False'], default='False', help="Run in distributed mode")
args = parser.parse_args()

# Check if arg contains directory name and/or .json
if 'configs/' not in args.c:
    args.c = 'configs/' + args.c
if '.json' not in args.c:
    args.c = args.c + '.json'

# Check if the config file exists
if not os.path.isfile(args.c):
    raise FileNotFoundError(f"Config file {args.c} not found.")


if __name__ == "__main__":

    with open(args.c, "r") as f:
        config = json.load(f)

    config['overwrite'] = args.o
    config['mode'] = args.m 
    config['name'] = os.path.basename(args.c).replace('.json', '')
    config['log'] = args.log
    if args.d == 'True':
        config['distributed'] = True
    else:
        config['distributed'] = False

    device = ('cuda' if torch.cuda.is_available() else "cpu")
    config['device'] = device
    
    run = runner(config)
    if run.config['mode'] == 'train':
        run.train()
    elif run.config['mode'] == 'pred':
        run.pred()
    elif run.config['mode'] == 'eval':
        run.eval()
    elif run.config['mode'] == 'latent':
        from lib.dls import latent_eval
        latent_eval(run)
    # elif run.config['mode'] == 'anim':
    #     from lib.plotting import animate
    #     animate(run)
    
    # copy the config file to the model directory
    shutil.copy(args.c, run.paths_bib.model_dir + os.path.basename(args.c))
    
    
    print(f"{'#'*20}\t{'End of script':<20}\t{'#'*20}")
    # close the log file
    sys.stdout.close()
    sys.stderr.close()