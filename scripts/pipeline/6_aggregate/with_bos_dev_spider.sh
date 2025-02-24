#!/bin/bash
uv run run_bo_sql.py \
    --ds spider \
    --type dev \
    --task aggregate \
    --eval_target fill_in \
    --exp_name pipeline_exp \
    --with_bos