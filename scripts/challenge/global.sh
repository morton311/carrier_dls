#!/bin/bash
# JOB HEADERS HERE
#SBATCH --job-name=challenge_global
#SBATCH --account=NAWCP24632466
#SBATCH --qos=standard
#SBATCH --constraint=mla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 24:00:00
#SBATCH --output=out/challenge_global.out

module use $HOME/my_modules
module load torch_module
source $HOME/.venv/bin/activate

torchrun main.py -c 'challenge/global' -m 'train' -d "True"
python main.py -c challenge/global -m pred
python main.py -c challenge/global -m eval
