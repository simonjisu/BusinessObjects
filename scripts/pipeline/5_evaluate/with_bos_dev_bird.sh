#!/bin/bash
uv run run_bo_sql.py \
    --ds bird \
    --type dev \
    --task evaluate \
    --exp_name pipeline_exp \
    --eval_target fill_in \
    --num_cpus 3 \
    --with_bos