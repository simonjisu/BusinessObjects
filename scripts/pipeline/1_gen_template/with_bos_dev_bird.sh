#!/bin/bash
uv run run_bo_sql.py \
    --ds bird \
    --type dev \
    --task gen_template \
    --exp_name pipeline_exp \
    --with_bos \
    --prefix "x-dev-with_bos-"