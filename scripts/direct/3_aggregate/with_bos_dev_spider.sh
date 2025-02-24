#!/bin/bash
uv run run_bo_sql.py \
    --ds spider \
    --type dev \
    --task aggregate \
    --eval_target direct \
    --exp_name direct_exp \
    --with_bos