#!/bin/bash
uv run run_bo_sql.py \
    --ds spider \
    --type dev \
    --task keyword_extraction \
    --exp_name pipeline_exp \
    --prefix "x-dev-no_bos-"