#!/bin/bash
#$ -M wyu1@nd.edu
#$ -m abe
#$ -q gpu@qa-p100-007
#$ -pe smp 1
#$ -l gpu=0

set -x

CODEHOME=/afs/crc.nd.edu/group/dmsquare/vol4/wyu1/A_EMNLP_2021_InjType/itype

DATASET=gigawords
NLU_WEIGHT=2.0
CLS_WEIGHT=2.0

DATAHOME=${CODEHOME}/dataset 
EXEHOME=${CODEHOME}/codes
SAVEPATH=${CODEHOME}/outputs_nlu${NLU_WEIGHT}_cls${CLS_WEIGHT}/${DATASET}

mkdir -p ${SAVEPATH}

cd ${EXEHOME}

CUDA_VISIBLE_DEVICES=3 /afs/crc.nd.edu/user/w/wyu1/anaconda3/envs/itt/bin/python train.py \
       -save_path ${SAVEPATH} \
       -log_home ${SAVEPATH} \
       -online_process_data \
       -enc_rnn_size 256 \
       -eval_per_batch 500 \
       -batch_size 32 \
       -epochs 60 \
       -learning_rate 0.001 \
       -gpus 0 \
       -extra_shuffle \
       -seed 12345 \
       -cuda_seed 12345 \
       -train_src ${DATAHOME}/train/gigawords_type_train.txt.source.txt \
       -train_guide_src ${DATAHOME}/train/gigawords_mention_train.txt.source.txt \
       -train_tgt ${DATAHOME}/train/gigawords_type_train.txt.target.txt \
       -src_vocab ${DATAHOME}/train/gigawords_type_vocab_src.txt.20k \
       -guide_src_vocab ${DATAHOME}/train/gigawords_mention_vocab_src.txt.20k \
       -tgt_vocab ${DATAHOME}/train/gigawords_type_vocab_src.txt.20k \
       -dev_input_src ${DATAHOME}/dev/gigawords_type_dev.txt.shuffle.dev.source.txt \
       -dev_guide_src ${DATAHOME}/dev/gigawords_mention_dev.txt.shuffle.dev.source.txt \
       -dev_ref ${DATAHOME}/dev/gigawords_type_dev.txt.shuffle.dev.target.txt \
       -test_input_src ${DATAHOME}/test/gigawords_type_test.txt.shuffle.test.source.txt \
       -test_guide_src ${DATAHOME}/test/gigawords_mention_test.txt.shuffle.test.source.txt \
       -test_ref ${DATAHOME}/test/gigawords_type_test.txt.shuffle.test.target.txt \
       -ofn_prefix output_${DATASET} \
       -cls ${CLS_WEIGHT} \
       -nlu ${NLU_WEIGHT} \

