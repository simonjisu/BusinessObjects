#!/bin/bash
uv run run_bo_sql.py \
    --ds spider \
    --type dev \
    --task direct \
    --exp_name direct_exp \
    --with_bos \
    --prefix "x-dev-with_bos-"