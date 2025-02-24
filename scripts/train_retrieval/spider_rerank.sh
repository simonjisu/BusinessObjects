#!/bin/bash
uv run train_retrieval_model.py --task rerank --ds spider \
    --per_device_train_batch_size 256 --per_device_eval_batch_size 128 --logging_steps 1 \
    --save_steps 10 --eval_steps 10 --num_train_epochs 5 \
    --exp_name "pipeline_exp"
