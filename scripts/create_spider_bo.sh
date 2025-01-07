#!/bin/bash
uv run run_bo_sql.py \
    --ds spider \
    --type train \
    --task create_bo \
    --description_file description.json