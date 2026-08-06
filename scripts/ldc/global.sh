#!/bin/bash
# JOB HEADERS HERE
#SBATCH --job-name=global_ldc_dls
#SBATCH --account=NAWCP24632466
#SBATCH --qos=standard
#SBATCH --constraint=mla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 24:00:00
#SBATCH --output=out/global_ldc_dls.out

# Prefer node-local scratch if Slurm provides it, otherwise create a job-specific temp dir.
export TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}/slurm-${SLURM_JOB_ID:-$$}}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

module use $HOME/my_modules
module load torch_module
source $HOME/.venv/bin/activate

python main.py -c ldc/global_dls2 -m train
python main.py -c ldc/global_dls2 -m pred
python main.py -c ldc/global_dls2 -m eval