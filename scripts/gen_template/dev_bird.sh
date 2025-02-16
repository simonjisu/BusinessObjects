#!/bin/bash
uv run run_bo_sql.py \
    --ds bird \
    --type dev \
    --task gen_template \
    --description_file bird_description.json