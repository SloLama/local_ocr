#!/bin/bash
#SBATCH --job-name=ocr_postprocessing
#SBATCH --output=logs/postprocess_err_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --array=0-15

model=$1
input_dir=$2
batch_size=$3

# TODO: Add your paths
WORK_DIR=""
CONTAINER=""
DATA_DIR=""
MODELS_DIR=""
SCRIPT_DIR=""

OUTPUT_DIR=$SCRIPT_DIR/output/$input_dir
LOG_DIR=$SCRIPT_DIR/postporcessing_logs/$input_dir

SCRIPT_LOG_DIR=$SCRIPT_DIR/logs/$input_dir

mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR
mkdir -p $SCRIPT_LOG_DIR

tp_size=${SLURM_GPUS_ON_NODE}

srun \
    --cpu-bind=verbose \
    --output=$SCRIPT_LOG_DIR/$SLURM_ARRAY_TASK_ID.txt \
    singularity exec \
        --nv \
        -B $SCRIPT_DIR:/script,$OUTPUT_DIR:/output,$LOG_DIR:/log,$DATA_DIR:/data,$MODELS_DIR:/models \
        --pwd /script \
        $CONTAINER \
        python3 postprocess_ocr.py \
            --model_path=$model \
            --input_dir=/data/$input_dir \
            --output_dir=/output \
            --log_dir=/log \
            --tp_size=$tp_size \
            --batch_size=$batch_size \
            --skip_existing \
            --n_chunks=$SLURM_ARRAY_TASK_COUNT \
            --chunk_index=$SLURM_ARRAY_TASK_ID
