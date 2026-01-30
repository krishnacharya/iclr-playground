#!/bin/bash
#SBATCH --job-name=tfidf_dyn
#SBATCH --output=logs/tf_idf_dynamic/tfidf_dyn_%A_%a.out
#SBATCH --error=logs/tf_idf_dynamic/tfidf_dyn_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=all
#SBATCH --array=0-26

echo "Starting TF-IDF dynamic grid search at $(date)"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "============================================================"

cd /n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer
source .venv/bin/activate

# 9 text types × 3 targets = 27 jobs (all with --dynamically-create-clean-md)
# Index 0-2:   original × (binary, decision, citation)
# Index 3-5:   clean × (binary, decision, citation)
# Index 6-8:   clean-title × (binary, decision, citation)
# Index 9-11:  original-abstract × (binary, decision, citation)
# Index 12-14: clean-abstract × (binary, decision, citation)
# Index 15-17: clean-introduction × (binary, decision, citation)
# Index 18-20: clean-results × (binary, decision, citation)
# Index 21-23: clean-conclusion × (binary, decision, citation)
# Index 24-26: clean-references × (binary, decision, citation)

TEXT_TYPES=(
    "original" "original" "original"
    "clean" "clean" "clean"
    "clean-title" "clean-title" "clean-title"
    "original-abstract" "original-abstract" "original-abstract"
    "clean-abstract" "clean-abstract" "clean-abstract"
    "clean-introduction" "clean-introduction" "clean-introduction"
    "clean-results" "clean-results" "clean-results"
    "clean-conclusion" "clean-conclusion" "clean-conclusion"
    "clean-references" "clean-references" "clean-references"
)
TARGETS=(
    "binary" "decision" "citation"
    "binary" "decision" "citation"
    "binary" "decision" "citation"
    "binary" "decision" "citation"
    "binary" "decision" "citation"
    "binary" "decision" "citation"
    "binary" "decision" "citation"
    "binary" "decision" "citation"
    "binary" "decision" "citation"
)

TEXT_TYPE=${TEXT_TYPES[$SLURM_ARRAY_TASK_ID]}
TARGET=${TARGETS[$SLURM_ARRAY_TASK_ID]}

echo "Running: text_type=$TEXT_TYPE, target=$TARGET, dynamically-create-clean-md=True"
echo "============================================================"

python scripts/tf_idf_analysis.py \
    --text-type $TEXT_TYPE \
    --target $TARGET \
    --dataset-path data/iclr_2020_2025_80_20_split \
    --output-dir notebooks/generated_tf_idf_dynamic \
    --dynamically-create-clean-md

echo "============================================================"
echo "Task $SLURM_ARRAY_TASK_ID completed at $(date)"
