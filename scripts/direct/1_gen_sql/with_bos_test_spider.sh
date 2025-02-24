#!/bin/bash
uv run run_bo_sql.py \
    --ds bird \
    --type test \
    --task direct \
    --exp_name direct_exp \
    --with_bos \
    --prefix "x-test-with_bos-"