#!/bin/bash
uv run run_bo_sql.py \
    --ds spider \
    --type test \
    --task direct \
    --exp_name direct_exp \
    --prefix "x-test-no_bos-"