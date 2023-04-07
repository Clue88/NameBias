#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-100
#SBATCH --cpus-per-task=2
#SBATCH --time=8:00:00
#SBATCH --mem=8g
#SBATCH --job-name=eec
#SBATCH --mail-type=END
#SBATCH --mail-user=liuchris@seas.upenn.edu
#SBATCH --output=slurm_out/eec%a.out

srun python3 eec_similarity.py
