#!/bin/bash
uv run run_bo_sql.py \
    --ds bird \
    --type dev \
    --task fill_in \
    --exp_name pipeline_exp \
    --prefix "x-dev-no_bos-"