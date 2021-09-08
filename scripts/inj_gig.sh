#!/bin/bash

set -x

CODEHOME=${@:(-1):1}

DATASET=gigawords
NLU_WEIGHT=2.0
CLS_WEIGHT=2.0

DATAHOME=${CODEHOME}/dataset
EXEHOME=${CODEHOME}/codes
SAVEPATH=${CODEHOME}/outputs_nlu${NLU_WEIGHT}_cls${CLS_WEIGHT}/${DATASET}

mkdir -p ${SAVEPATH}

cd ${EXEHOME}

python ${EXEHOME}/train.py \
       -save_path ${SAVEPATH} \
       -log_home ${SAVEPATH} \
       -online_process_data \
       -enc_rnn_size 256 \
       -eval_per_batch 500 \
       -batch_size 32 \
       -epochs 30 \
       -learning_rate 0.001 \
       -gpus 0 \
       -extra_shuffle \
       -seed 12345 \
       -cuda_seed 12345 \
       -train_src ${DATAHOME}/train/${DATASET}_type_train.txt.source.txt \
       -train_guide_src ${DATAHOME}/train/${DATASET}_mention_train.txt.source.txt \
       -train_guide_tgt ${DATAHOME}/train/${DATASET}_mention_train.txt.target.txt \
       -train_tgt ${DATAHOME}/train/${DATASET}_type_train.txt.target.txt \
       -src_vocab ${DATAHOME}/train/${DATASET}_type_vocab_src.txt.20k \
       -guide_src_vocab ${DATAHOME}/train/${DATASET}_mention_vocab_src.txt.20k \
       -tgt_vocab ${DATAHOME}/train/${DATASET}_type_vocab_src.txt.20k \
       -dev_input_src ${DATAHOME}/dev/${DATASET}_type_dev.txt.shuffle.dev.source.txt \
       -dev_guide_src ${DATAHOME}/dev/${DATASET}_mention_dev.txt.shuffle.dev.source.txt \
       -dev_guide_tgt ${DATAHOME}/dev/${DATASET}_mention_dev.txt.shuffle.dev.target.txt \
       -dev_ref ${DATAHOME}/dev/${DATASET}_type_dev.txt.shuffle.dev.target.txt \
       -test_input_src ${DATAHOME}/test/${DATASET}_type_test.txt.shuffle.test.source.txt \
       -test_guide_src ${DATAHOME}/test/${DATASET}_mention_test.txt.shuffle.test.source.txt \
       -test_guide_tgt ${DATAHOME}/test/${DATASET}_mention_test.txt.shuffle.test.target.txt \
       -test_ref ${DATAHOME}/test/${DATASET}_type_test.txt.shuffle.test.target.txt \
       -ofn_prefix output_${DATASET} \
       -cls ${CLS_WEIGHT} \
       -nlu ${NLU_WEIGHT} \