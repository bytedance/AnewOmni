# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/bin/bash
########## Instruction ##########
# This script takes three optional environment variables:
# GPU / ADDR / PORT
# e.g. Use gpu 0, 1 and 4 for training, set distributed training
# master address and port to localhost:9901, the command is as follows:
#
# GPU="0,1,4" ADDR=localhost PORT=9901 bash train.sh
#
# Default value: GPU=-1 (use cpu only), ADDR=localhost, PORT=9901
# Note that if your want to run multiple distributed training tasks,
# either the addresses or ports should be different between
# each pair of tasks.
######### end of instruction ##########


########## setup project directory ##########
CODE_DIR=`realpath $(dirname "$0")/..`
echo "Locate the project folder at ${CODE_DIR}"


########## parsing yaml configs ##########
if [ -z $1 ]; then
    echo "Config missing. Usage example: GPU=0,1 bash $0 <config> [optional arguments]"
    exit 1;
fi


########## setup  distributed training ##########
GPU="${GPU:--1}" # default using CPU
MASTER_ADDR="${ADDR:-localhost}"
MASTER_PORT="${PORT:-9901}"
echo "Using GPUs: $GPU"
echo "Master address: ${MASTER_ADDR}, Master port: ${MASTER_PORT}"
# Multi-Node distributed training
if [ -z $NNODE ]; then
    echo "NNODE not given. Default assuming one node for training."
    NNODE=1
fi

export CUDA_VISIBLE_DEVICES=$GPU
GPU_ARR=(`echo $GPU | tr ',' ' '`)

if [ $NNODE -gt 1 ]; then   # some frameworks only support the depricated API torch.distributed.launch for multi-node training
    echo "Number of nodes: ${NNODE}. Multi-node training enabled."
    if [ -z ${NODE_RANK} ]; then
        echo "Environment variable NODE_RANK missing, which is necessary for multi-node distributed training!"
        exit 1;
    fi
    export OMP_NUM_THREADS=2
    PREFIX="torchrun --nproc_per_node=${#GPU_ARR[@]} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --nnodes=${NNODE} --node_rank=${NODE_RANK} --max-restarts=10"
elif [ ${#GPU_ARR[@]} -gt 1 ]; then
    export OMP_NUM_THREADS=2
	PREFIX="torchrun --nproc_per_node=${#GPU_ARR[@]} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} --nnodes=1 --max-restarts=10"
else
    PREFIX="python"
fi


########## start training ##########
cd $CODE_DIR
${PREFIX} train.py --gpu "${!GPU_ARR[@]}" --config $@
