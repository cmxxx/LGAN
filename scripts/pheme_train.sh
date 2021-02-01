#!/bin/bash

# event_inputs: 'NOT_HOLD-ONE', 'charliehebdo', 'ferguson', 'germanwings-crash', 'ottawashooting','sydneysiege'
event=$1 
gpu=$2

classify_output=sigmoid
mask=4

#training
CUDA_VISIBLE_DEVICES=$gpu nohup python -u train.py \
--TRAIN_MODE \
--WORKING_SPACE ./ \
--DATA_DIR ./data \
--DATASET PHEMEv5 --DATA_EVENT_SPECIFY "${event}" \
--MAX_SEQUENCE_LENGTH 50 --MASK_SIZE ${mask} --BATCH_SIZE 16 \
--IS_OVER_SAMPLING True \
--GLOVE_DIM 100 --NB_WORDS 15000 \
--FIRST_EPOCHS 200 --EARLY_STOP_EPOCH 15 \
--D_CLASSIFY_OUTPUT ${classify_output} \
--G_R_LR 1e-4 --G_W_LR 1e-4 \
--D_C_LR 1e-4 --D_W_LR_1 1e-4 \
--G_STEP 1 --D_STEP 3 --D_STEP_K 1 \
--D_CLASSIFY_IS_TRAIN_FIRST_EPOCH \
--D_WHERE_IS_TRAIN_FIRST_EPOCH \
--G_TRAIN_PATTERN R-1_N-0 \
--D_CLASSIFY_TRAIN_PATTERN N-1_R-0_N-m-3_R-m-2 \
--IS_LSTM_D_CLASSIFY \ # mute to do LEX-CNNs
--VERSION ${event}_LGAN_4_class_R_mask_${mask}.${classify_output}.v1 > log/PHEMEv5_${event}.LEX_LSTM.2_class.R_mask_${mask}.${classify_output}.v1.log1 2>&1 &