#!/bin/bash
uv run run_evaluation.py \
    --ds spider \
    --description_file description.json \
    --type dev \
    --task zero_shot
    