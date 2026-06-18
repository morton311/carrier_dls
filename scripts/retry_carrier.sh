#!/bin/bash
# JOB HEADERS HERE
#SBATCH --job-name=no_skip_z
#SBATCH --account=NAWCP24632466
#SBATCH --qos=standard
#SBATCH --constraint=mla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 24:00:00
#SBATCH --output=out_skip.out

module use $HOME/my_modules
module load torch_module
source $HOME/.venv/bin/activate

torchrun main.py -c 'carrier/no_skip_z' -m 'train' -d "True"
