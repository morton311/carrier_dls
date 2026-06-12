#!/bin/bash
# JOB HEADERS HERE
#SBATCH --job-name=ldc_lg_tr
#SBATCH --account=NAWCP24632466
#SBATCH --qos=standard
#SBATCH --constraint=mla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 12:00:00
#SBATCH --output=out/ldc_lg_tr.out

module use $HOME/my_modules
module load torch_module
source $HOME/.venv/bin/activate

# torchrun main.py -c 'ldc_lg_tr' -m 'train' -d "True"
python main.py -c 'ldc_lg_tr' -m 'pred'
python main.py -c 'ldc_lg_tr' -m 'eval'
