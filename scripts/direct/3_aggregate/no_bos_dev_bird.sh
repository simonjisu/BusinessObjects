#!/bin/bash
uv run run_bo_sql.py \
    --ds bird \
    --type dev \
    --task aggregate \
    --eval_target fill_in \
    --exp_name pipeline_exp
