#!/bin/bash
uv run run_bo_sql.py \
    --ds spider \
    --type dev \
    --task valid_bo \
    --description_file description.json \
    --db_id_group 6

