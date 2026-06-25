#!/bin/bash
# JOB HEADERS HERE
#SBATCH --job-name=ldc
#SBATCH --account=AFMNG31652E99
#SBATCH --qos=standard
#SBATCH --constraint=mla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 12:00:00
#SBATCH --output=output/ldc.out

module use $HOME/my_modules
module load torch_module
source $HOME/.venv/bin/activate

# torchrun main.py -c 'ldc/ldc' -m 'train' -d "True"
python main.py -c 'ldc/ldc' -m 'pred'
python main.py -c 'ldc/ldc' -m 'eval'
