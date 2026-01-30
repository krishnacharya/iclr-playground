#!/bin/bash
#SBATCH --job-name=tfidf_sect
#SBATCH --output=logs/tf_idf/tfidf_sect_%A_%a.out
#SBATCH --error=logs/tf_idf/tfidf_sect_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=all
#SBATCH --array=0-17

echo "Starting TF-IDF sections grid search at $(date)"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "============================================================"

cd /n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer
source .venv/bin/activate

# Define grid: 6 text types × 3 targets = 18 jobs
# Text types: clean-title, original-abstract, clean-abstract,
#             clean-introduction, clean-results, clean-conclusion
# Targets: binary, decision, citation

TEXT_TYPES=(
    "clean-title" "clean-title" "clean-title"
    "original-abstract" "original-abstract" "original-abstract"
    "clean-abstract" "clean-abstract" "clean-abstract"
    "clean-introduction" "clean-introduction" "clean-introduction"
    "clean-results" "clean-results" "clean-results"
    "clean-conclusion" "clean-conclusion" "clean-conclusion"
)
TARGETS=(
    "binary" "decision" "citation"
    "binary" "decision" "citation"
    "binary" "decision" "citation"
    "binary" "decision" "citation"
    "binary" "decision" "citation"
    "binary" "decision" "citation"
)

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
