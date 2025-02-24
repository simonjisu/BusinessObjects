#!/bin/bash
uv run run_bo_sql.py \
    --ds spider \
    --type test \
    --task direct \
    --exp_name direct_exp \
    --with_bos \
    --prefix "x-test-with_bos-"