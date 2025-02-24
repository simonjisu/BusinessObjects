#!/bin/bash
uv run run_bo_sql.py \
    --ds spider \
    --type dev \
    --task valid_bo \
    --exp_name pipeline_exp