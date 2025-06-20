#!/bin/bash
#SBATCH --job-name=vllm_ocr
#SBATCH --output=ocr_err.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --mem=256G

# TODO: add your paths
WORK_DIR=
SCRIPT_DIR=
CONTAINER=
MODELS_DIR=
DATA_DIR=

input_dir=$1
batch_size=$2
echo "Running vLLM OCR with batch size $batch_size."

OUTPUT_DIR=$SCRIPT_DIR/output/$input_dir
mkdir -p $OUTPUT_DIR

DATA_DIR=$DATA_DIR/$input_dir

mkdir -p $SCRIPT_DIR/logs/$input_dir

srun \
  --output="$SCRIPT_DIR/logs/$input_dir.txt" \
  singularity exec \
    --nv \
    -B $DATA_DIR:/data,$OUTPUT_DIR:/output,$SCRIPT_DIR:/script,$MODELS_DIR:/models \
    --pwd /script \
    $CONTAINER \
      python3 ocr_with_vllm.py --batch_size=$batch_size --skip_existing
