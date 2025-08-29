#!/bin/bash
#SBATCH --job-name=vllm_server
#SBATCH --output=run_err.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=8
#SBATCH --mem=0
#SBATCH --exclusive

# TODO: Add your paths
WORK_DIR=""
CONTAINER=""
MODELS_DIR=""
SCRIPT_DIR=""

model_name=$1

node_ip=$(srun --nodes=1 --ntasks=1 hostname --ip-address)

sleep 5

# If model is stored locally, copy it to the node's local storage for faster loading
if [[ "$model_name" == /models/* ]]; then
    base_name=$(basename "$model_name")

    echo "Starting weights copy at $(date)"
    cp -r $MODELS_DIR/$base_name $SCRATCH
    echo "Copy finished at $(date)"

    model_name=$SCRATCH/$base_name
    echo "Changed model name to $model_name"
else
    echo "Model name does not start with '/models'. Weight copying is not performed."
fi

# Run with Enroot
srun \
    --output=logs/${model_name}.txt \
    --container-mounts $MODELS_DIR:/models \
    --container-image $CONTAINER \
    --container-name ${SLURM_JOB_NAME} \
    --container-workdir $SCRIPT_DIR \
    ./run_server.sh $model_name $node_ip
