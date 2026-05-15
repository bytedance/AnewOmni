# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/bin/bash
########## setup project directory ##########
CODE_DIR=`realpath $(dirname "$0")/..`
echo "Locate the project folder at ${CODE_DIR}"
cd ${CODE_DIR}

######### check number of args ##########
HELP="Usage example: GPU=0,1,2,3 bash $0 <path> <AE config> <LDM config> <confidence config> [mode: e.g. 111]"
if [ -z $1 ]; then
    echo "Experiment saving path missing. ${HELP}"
    exit 1;
else
    SAVE_PATH=$1
fi
if [ -z $2 ]; then
    echo "Autoencoder config missing. ${HELP}"
    exit 1;
else
    AECONFIG=$2
fi
if [ -z $3 ]; then
    echo "LDM config missing. ${HELP}"
    exit 1;
else
    LDMCONFIG=$3
fi
if [ -z $4 ]; then
    echo "Confidence module config missing. ${HELP}"
else
    CONFCONFIG=$4
fi
if [ -z $5 ]; then
    MODE=111
else
    MODE=$5
fi

echo "Mode: $MODE, [train AE] / [train LDM] / [train Confidence Module]"
TRAIN_AE_FLAG=${MODE:0:1}
TRAIN_LDM_FLAG=${MODE:1:1}
TRAIN_CONF_FLAG=${MODE:2:1}

SUFFIX=`basename ${SAVE_PATH}`

AE_SAVE_DIR=$SAVE_PATH/AE_${SUFFIX}
LDM_SAVE_DIR=$SAVE_PATH/LDM_${SUFFIX}
CONF_SAVE_DIR=$SAVE_PATH/Conf_${SUFFIX}
OUTLOG=$SAVE_PATH/output.log


########## Handle existing directories ##########
update_max_epoch() {
    local config_file=$1
    local ckpt_file=$2
    config_max_epoch=`cat $config_file | grep -E "max_epoch: [0-9]+" | grep -oE "[0-9]+"`
    current_epoch=`basename $ckpt_file | grep -oE "epoch[0-9]+" | grep -oE "[0-9]+"`
    echo $((config_max_epoch - current_epoch - 1))
}

get_latest_checkpoint() {
    local save_dir=$1
    latest_ckpt=`cat ${save_dir}/version_0/checkpoint/topk_map.txt | head -n 1 | awk -F " " '{print $2}'`
    echo $latest_ckpt
}


organize_folder() {
    local save_dir=$1
    if [ -e $save_dir/old ]; then   # clean old checkpoints
        rm -r $save_dir/old
    fi
    mv ${save_dir}/version_0 ${save_dir}/old
    echo ${save_dir}/old
}


if [[ ! -e $SAVE_PATH ]]; then
    mkdir -p $SAVE_PATH
fi

# if [[ -e $AE_SAVE_DIR ]] && [ "$TRAIN_AE_FLAG" = "1" ]; then
#     echo "Directory ${AE_SAVE_DIR} exisits! But training flag is 1!"
#     LATEST_AE_CKPT=$(get_latest_checkpoint $AE_SAVE_DIR)
#     if [ -n "$LATEST_AE_CKPT" ]; then
#         echo "Found Autoencoder checkpoint: $LATEST_AE_CKPT"
#         AE_UPDATE_MAX_EPOCH=$(update_max_epoch $AECONFIG $LATEST_AE_CKPT)
#         echo "Updated max epoch to $AE_UPDATE_MAX_EPOCH"
#         echo "Moved old checkpoints to $(organize_folder $AE_SAVE_DIR)"
#         LATEST_AE_CKPT=${LATEST_AE_CKPT/version_0/old}
#         AE_CONTINUE_ARGS="--load_ckpt $LATEST_AE_CKPT --trainer.config.max_epoch $AE_UPDATE_MAX_EPOCH"
#     else
#         echo "No checkpoint found. Training will start from scratch."
#         AE_CONTINUE_ARGS=""
#     fi
# fi

# if [[ -e $LDM_SAVE_DIR ]] && [ "$TRAIN_LDM_FLAG" = "1" ]; then
#     echo "Directory ${LDM_SAVE_DIR} exisits! But training flag is 1!"
#     LATEST_LDM_CKPT=$(get_latest_checkpoint $LDM_SAVE_DIR)
#     if [ -n "$LATEST_LDM_CKPT" ]; then
#         echo "Found LDM checkpoint: $LATEST_LDM_CKPT"
#         LDM_UPDATE_MAX_EPOCH=$(update_max_epoch $LDMCONFIG $LATEST_LDM_CKPT)
#         echo "Updated max epoch to $LDM_UPDATE_MAX_EPOCH"
#         echo "Moved old checkpoints to $(organize_folder $LDM_SAVE_DIR)"
#         LATEST_LDM_CKPT=${LATEST_LDM_CKPT/version_0/old}
#         LDM_CONTINUE_ARGS="--load_ckpt $LATEST_LDM_CKPT --trainer.config.max_epoch $LDM_UPDATE_MAX_EPOCH"
#     else
#         echo "No checkpoint found. Training will start from scratch."
#         LDM_CONTINUE_ARGS=""
#     fi
# fi

# if [[ -e $CONF_SAVE_DIR ]] && [ "$TRAIN_CONF_FLAG" = "1" ]; then
#     echo "Directory ${CONF_SAVE_DIR} exisits! But training flag is 1!"
#     LATEST_CONF_CKPT=$(get_latest_checkpoint $CONF_SAVE_DIR)
#     if [ -n "$LATEST_CONF_CKPT" ]; then
#         echo "Found confidence module checkpoint: $LATEST_CONF_CKPT"
#         CONF_UPDATE_MAX_EPOCH=$(update_max_epoch $CONFCONFIG $LATEST_CONF_CKPT)
#         echo "Updated max epoch to $CONF_UPDATE_MAX_EPOCH"
#         echo "Moved old checkpoints to $(organize_folder $CONF_SAVE_DIR)"
#         LATEST_CONF_CKPT=${LATEST_CONF_CKPT/version_0/old}
#         CONF_CONTINUE_ARGS="--load_ckpt $LATEST_CONF_CKPT --trainer.config.max_epoch $CONF_UPDATE_MAX_EPOCH"
#     else
#         echo "No checkpoint found. Training will start from scratch."
#         CONF_CONTINUE_ARGS=""
#     fi
# fi

########## train autoencoder ##########
echo "Training Autoencoder with config $AECONFIG:" > $OUTLOG
cat $AECONFIG >> $OUTLOG
echo "Overwriting args $ARGS1"
if [ "$TRAIN_AE_FLAG" = "1" ]; then
    bash scripts/train.sh $AECONFIG --trainer.config.save_dir=$AE_SAVE_DIR $ARGS1
    if [ $? -eq 0 ]; then
        echo "Succeeded in training AutoEncoder"
    else
        echo "Error encountered during training AutoEncoder"
        exit 1;
    fi
    # if  [ -z $AE_UPDATE_MAX_EPOCH ] || [ $AE_UPDATE_MAX_EPOCH -gt 0 ]; then
    #     bash scripts/train.sh $AECONFIG --trainer.config.save_dir=$AE_SAVE_DIR $ARGS1 $AE_CONTINUE_ARGS
    #     if [ $? -eq 0 ]; then
    #         echo "Succeeded in training AutoEncoder"
    #     else
    #         echo "Failed to train AutoEncoder"
    #         exit 1;
    #     fi
    # else
    #     echo "AutoEncoder already finished training"
    # fi
fi

########## train ldm ##########
echo "Training LDM with config $LDMCONFIG:" >> $OUTLOG
cat $LDMCONFIG >> $OUTLOG
echo "Overwriting args $ARGS2"
AE_CKPT=`cat ${AE_SAVE_DIR}/version_0/checkpoint/topk_map.txt | head -n 1 | awk -F " " '{print $2}'`
echo "Using Autoencoder checkpoint: ${AE_CKPT}" >> $OUTLOG
if [ "$TRAIN_LDM_FLAG" = "1" ]; then
    bash scripts/train.sh $LDMCONFIG --trainer.config.save_dir=$LDM_SAVE_DIR --model.autoencoder_ckpt=$AE_CKPT $ARGS2
    if [ $? -eq 0 ]; then
        echo "Succeeded in training LDM"
    else
        echo "Error encountered during training LDM"
        exit 1;
    fi
    # if  [ -z $LDM_UPDATE_MAX_EPOCH ] || [ $LDM_UPDATE_MAX_EPOCH -gt 0 ]; then
    #     bash scripts/train.sh $LDMCONFIG --trainer.config.save_dir=$LDM_SAVE_DIR --model.autoencoder_ckpt=$AE_CKPT $ARGS2 $LDM_CONTINUE_ARGS
    # else
    #     echo "LDM already finished training"
    # fi
fi

########## train confidence module ##########
echo "Training confidence module with config $CONFCONFIG:" >> $OUTLOG
cat $CONFCONFIG >> $OUTLOG
echo "Overwriting args $ARGS3"
LDM_CKPT=`cat ${LDM_SAVE_DIR}/version_0/checkpoint/topk_map.txt | head -n 1 | awk -F " " '{print $2}'`
echo "Using LDM checkpoint: ${LDM_CKPT}" >> $OUTLOG
if [ "$TRAIN_CONF_FLAG" = "1" ]; then
    bash scripts/train.sh $CONFCONFIG --trainer.config.save_dir=$CONF_SAVE_DIR --model.base_model_path=$LDM_CKPT $ARGS3
    if [ $? -eq 0 ]; then
        echo "Succeeded in training Confidence Module"
    else
        echo "Error encountered during training Confidence Module"
        exit 1;
    fi
    # if  [ -z $CONF_UPDATE_MAX_EPOCH ] || [ $CONF_UPDATE_MAX_EPOCH -gt 0 ]; then
    #     bash scripts/train.sh $CONFCONFIG --trainer.config.save_dir=$CONF_SAVE_DIR --model.base_model_path=$LDM_CKPT $ARGS3 $CONF_CONTINUE_ARGS
    # else
    #     echo "Confidence module already finished training"
    # fi
fi