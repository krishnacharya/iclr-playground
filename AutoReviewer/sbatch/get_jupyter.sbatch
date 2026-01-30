#!/bin/bash
#SBATCH --job-name=get_jupyter
#SBATCH --output=logs/jupyter/%j.out
#SBATCH --error=logs/jupyter/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=150G
#SBATCH --time=8:00:00


cd /n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer
source .venv/bin/activate

uv run --with jupyter jupyter lab