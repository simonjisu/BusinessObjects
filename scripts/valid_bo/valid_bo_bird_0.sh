#!/bin/bash
uv run run_bo_sql.py \
    --ds bird \
    --type dev \
    --task valid_bo \
    --description_file bird_description.json \
    --db_id_group 0 
