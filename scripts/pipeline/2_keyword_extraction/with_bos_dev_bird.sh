#!/bin/bash
uv run run_bo_sql.py \
    --ds bird \
    --type dev \
    --task keyword_extraction \
    --exp_name pipeline_exp \
    --with_bos