#!/bin/bash
uv run run_bo_sql.py \
    --ds bird \
    --type test \
    --task aggregate \
    --exp_name pipeline_exp \
    --eval_target fill_in \
    --with_bos