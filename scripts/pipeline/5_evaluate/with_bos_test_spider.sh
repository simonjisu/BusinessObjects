#!/bin/bash
uv run run_bo_sql.py \
    --ds spider \
    --type test \
    --task evaluate \
    --exp_name pipeline_exp \
    --eval_target fill_in \
    --num_cpus 3 \
    --with_bos \
    --prefix "x-test-with_bos-"