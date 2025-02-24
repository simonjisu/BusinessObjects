#!/bin/bash
uv run run_bo_sql.py \
    --ds spider \
    --type test \
    --task aggregate \
    --exp_name pipeline_exp \
    --eval_target fill_in
    
