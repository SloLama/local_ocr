#!/bin/bash
#SBATCH --job-name=ocr_postprocessing
#SBATCH --output=postprocess_err.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=256G

model=$1
input_dir=$2

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
    --output=$SCRIPT_LOG_DIR.txt \
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
            --skip_existing
