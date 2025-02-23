#!/bin/bash
uv run run_bo_sql.py \
    --ds bird \
    --type test \
    --task evaluate \
    --exp_name direct_exp \
    --eval_target direct \
    --num_cpus 1 \
    --prefix "x-test-no_bos-"