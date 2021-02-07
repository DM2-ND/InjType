#!/bin/bash
#$ -M wyu1@nd.edu
#$ -m abe
#$ -q gpu
#$ -l gpu_card=1


set -x

DATAHOME=${@:(-2):1}
EXEHOME=${@:(-1):1}

SAVEPATH=${DATAHOME}/models/

mkdir -p ${SAVEPATH}

cd ${EXEHOME}

python train.py \
       -save_path ${SAVEPATH} \
       -log_home ${SAVEPATH} \
       -online_process_data \
       -layers 1 \
       -enc_rnn_size 512 \
       -brnn \
       -word_vec_size 300 \
       -dropout 0.5 \
       -batch_size 32 \
       -beam_size 1 \
       -epochs 60 \
       -optim adam \
       -learning_rate 0.001 \
       -gpus 0 \
       -curriculum 0 \
       -extra_shuffle \
       -start_eval_batch 2000 \
       -eval_per_batch 500 \
       -halve_lr_bad_count 3 \
       -seed 12345 \
       -cuda_seed 12345 \
       -log_interval 10 \
       -train_src ${DATAHOME}/train/nyt_type_train.txt.source.txt \
       -train_guide_src ${DATAHOME}/train/nyt_mention_train.txt.source.txt \
       -src_vocab ${DATAHOME}/train/nyt_type_vocab_src.txt.20k \
       -guide_src_vocab ${DATAHOME}/train/nyt_mention_vocab_src.txt.20k \
       -train_tgt ${DATAHOME}/train/nyt_type_train.txt.target.txt \
       -tgt_vocab ${DATAHOME}/train/nyt_type_vocab_src.txt.20k \
       -dev_input_src ${DATAHOME}/dev/nyt_type_dev.txt.shuffle.dev.source.txt \
       -dev_guide_src ${DATAHOME}/dev/nyt_mention_dev.txt.shuffle.dev.source.txt \
       -dev_ref ${DATAHOME}/dev/nyt_type_dev.txt.shuffle.dev.target.txt \
       -test_input_src ${DATAHOME}/test/nyt_type_test.txt.shuffle.test.source.txt \
       -test_guide_src ${DATAHOME}/test/nyt_mention_test.txt.shuffle.test.source.txt \
       -test_ref ${DATAHOME}/test/nyt_type_test.txt.shuffle.test.target.txt \
       -ofn_prefix nyt_inj

