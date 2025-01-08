#!/bin/bash
uv run run_evaluation.py \
    --ds bird \
    --description_file bird_description.json \
    --type test \
    --task zero_shot_hint \
    --scenario 0
    