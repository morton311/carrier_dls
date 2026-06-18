#!/bin/bash
# JOB HEADERS HERE
#SBATCH --job-name=short
#SBATCH --account=NAWCP24632466
#SBATCH --qos=standard
#SBATCH --constraint=mla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 12:00:00
#SBATCH --output=output/short.out

module use $HOME/my_modules
module load torch_module
source $HOME/.venv/bin/activate

# torchrun main.py -c 'ldc/ldc_short' -m 'train' -d "True"
python main.py -c 'ldc/ldc_short' -m 'pred'
python main.py -c 'ldc/ldc_short' -m 'eval'
