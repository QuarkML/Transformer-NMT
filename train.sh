#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO

torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    train.py \
    --distributed \
    --fp16 \
    --src_vocab_size 32000 \
    --tgt_vocab_size 32000 \
    --d_model 256 \
    --num_heads 8 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --d_ff 1024 \
    --dropout 0.2 \
    --max_seq_length 128 \
    --shared_embeddings \
    --num_epochs 100 \
    --batch_size 64 \
    --learning_rate 3e-4 \
    --beta1 0.9 \
    --beta2 0.98 \
    --epsilon 1e-9 \
    --weight_decay 0.0001 \
    --max_grad_norm 1.0 \
    --label_smoothing 0.1 \
    --val_split 0.1 \
    --total_steps 354800 \
    --num_workers 4 \
    --save_dir models \
    --log_dir runs/tnmt_training \
    --log_interval 100 \
    --save_interval 1 \
    --translation_interval 500 \
    --seed 42