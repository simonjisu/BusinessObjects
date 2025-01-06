#!/bin/bash
uv run run_bo_sql.py \
    --ds bird \
    --type train \
    --task create_bo \
    --description_file bird_description.json