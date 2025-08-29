#!/bin/bash

model_name=$1
head_node_ip=$2

echo "Number of GPUs per task ${SLURM_GPUS_PER_TASK}"
echo "Number of CPUs per task ${SLURM_CPUS_PER_TASK}"
vllm_port_number=65535
SERVER_ADDR="http://${head_node_ip}:${vllm_port_number}/v1"
echo "Server address: $SERVER_ADDR"

nvidia-smi

python3 -m vllm.entrypoints.openai.api_server \
    --model ${model_name} \
    --host "0.0.0.0" \
    --port ${vllm_port_number} \
    --pipeline-parallel-size 1 \
    --tensor-parallel-size ${SLURM_GPUS_ON_NODE} \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --max-model-len 128000
