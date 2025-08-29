#!/bin/bash
#SBATCH --job-name=multimodal_ocr
#SBATCH --output=ocr_err.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00
#SBATCH --mem=32G

# TODO: Add your paths
WORK_DIR=""
CONTAINER=""
SCRIPT_DIR=""

input_dir=$1

OUTPUT_DIR=$SCRIPT_DIR/output/$input_dir
mkdir -p $OUTPUT_DIR

DATA_DIR=$WORK_DIR/corpuses/$input_dir

workers=$SLURM_CPUS_PER_TASK
echo "Running local OCR with $workers workers."

srun \
  --output="$SCRIPT_DIR/logs/$input_dir.txt" \
  --container-image $CONTAINER \
  --container-mounts $DATA_DIR:/data,$OUTPUT_DIR:/output,$SCRIPT_DIR:/script \
  --container-workdir /script \
      python3 ocr_with_local_model.py --workers=$workers --skip_existing
