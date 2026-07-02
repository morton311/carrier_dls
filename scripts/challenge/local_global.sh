#!/bin/bash
# JOB HEADERS HERE
#SBATCH --job-name=challenge_lg
#SBATCH --account=NAWCP24632466
#SBATCH --qos=standard
#SBATCH --constraint=mla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 24:00:00
#SBATCH --output=out/challenge_lg.out

module use $HOME/my_modules
module load torch_module
source $HOME/.venv/bin/activate

torchrun main.py -c 'challenge/local_global' -m 'train' -d "True"
python main.py -c challenge/local_global -m pred
python main.py -c challenge/local_global -m eval
