#!/bin/bash
uv run run_bo_sql.py \
    --ds spider \
    --type test \
    --task evaluate \
    --exp_name direct_exp \
    --eval_target direct \
    --num_cpus 3 \
    --prefix "x-test-no_bos-"