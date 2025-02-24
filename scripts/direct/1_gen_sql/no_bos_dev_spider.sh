#!/bin/bash
uv run run_bo_sql.py \
    --ds spider \
    --type dev \
    --task direct \
    --exp_name direct_exp \
    --prefix "x-dev-no_bos-"