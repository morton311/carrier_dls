#!/bin/bash
# JOB HEADERS HERE
#SBATCH --job-name=challenge
#SBATCH --account=AFMNG31652E99
#SBATCH --qos=debug
#SBATCH --constraint=mla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 1:00:00
#SBATCH --output=out/challenge.out

module use $HOME/my_modules
module load torch_module
source $HOME/.venv/bin/activate

# torchrun main.py -c 'challenge/challenge' -m 'train' -d "True"
# python main.py -c challenge/challenge -m pred
python main.py -c challenge/challenge -m eval
