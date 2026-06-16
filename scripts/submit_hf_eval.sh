#!/bin/bash
#SBATCH --job-name=fairness-hf-eval
#SBATCH --account=mihalcea98
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=7-00:00:00
#SBATCH --output=results/slurm_hf_eval_%j.out
#SBATCH --error=results/slurm_hf_eval_%j.err

module load singularity

bash "$(dirname "$0")/batch_evaluate_hf.sh"
