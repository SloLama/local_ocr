#!/bin/bash
#SBATCH --job-name=vllm_ocr
#SBATCH --output=logs/ocr_err_chunk.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=1-00
#SBATCH --mem=256G

# TODO: add your paths
WORK_DIR=
SCRIPT_DIR=
CONTAINER=
MODELS_DIR=
DATA_DIR=

input_dir=$1
batch_size=$2
n_chunks=$3
chunk_idx=$4
echo "Running vLLM OCR on chunk ${chunk_idx+1}/${n_shards} with batch size $batch_size."

OUTPUT_DIR=$SCRIPT_DIR/output/$input_dir
mkdir -p $OUTPUT_DIR

DATA_DIR=$WORK_DIR/corpora/$input_dir

mkdir -p $SCRIPT_DIR/logs/$input_dir

srun \
  --output="$SCRIPT_DIR/logs/$input_dir/$chunk_idx.txt" \
  singularity exec \
    --nv \
    -B $DATA_DIR:/data,$OUTPUT_DIR:/output,$SCRIPT_DIR:/script,$MODELS_DIR:/models \
    --pwd /script \
    $CONTAINER \
      python3 ocr_with_vllm.py --batch_size=$batch_size --skip_existing --n_chunks=$n_chunks --chunk_index=$chunk_idx
