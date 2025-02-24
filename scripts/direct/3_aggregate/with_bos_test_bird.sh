#!/bin/bash
uv run run_bo_sql.py \
    --ds bird \
    --type test \
    --task aggregate \
    --eval_target direct \
    --exp_name direct_exp \
    --with_bos