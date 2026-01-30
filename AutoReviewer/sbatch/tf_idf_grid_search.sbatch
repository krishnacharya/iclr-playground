#!/bin/bash
#SBATCH --job-name=tfidf_grid
#SBATCH --output=logs/tf_idf/tfidf_%A_%a.out
#SBATCH --error=logs/tf_idf/tfidf_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=all
#SBATCH --array=0-5

echo "Starting TF-IDF grid search at $(date)"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "============================================================"

cd /n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer
source .venv/bin/activate

# Define grid: (text_type, target)
# 0: original, binary
# 1: original, decision
# 2: original, citation
# 3: clean, binary
# 4: clean, decision
# 5: clean, citation

TEXT_TYPES=("original" "original" "original" "clean" "clean" "clean")
TARGETS=("binary" "decision" "citation" "binary" "decision" "citation")

TEXT_TYPE=${TEXT_TYPES[$SLURM_ARRAY_TASK_ID]}
TARGET=${TARGETS[$SLURM_ARRAY_TASK_ID]}

echo "Running: text_type=$TEXT_TYPE, target=$TARGET"
echo "============================================================"

python scripts/tf_idf_analysis.py \
    --text-type $TEXT_TYPE \
    --target $TARGET \
    --dataset-path data/iclr_2020_2025_80_20_split \
    --output-dir notebooks/generated_tf_idf_analysis

echo "============================================================"
echo "Task $SLURM_ARRAY_TASK_ID completed at $(date)"
