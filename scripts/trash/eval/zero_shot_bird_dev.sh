#!/bin/bash
uv run run_evaluation.py \
    --ds bird \
    --description_file bird_description.json \
    --type dev \
    --task zero_shot
    