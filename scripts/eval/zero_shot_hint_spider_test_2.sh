#!/bin/bash
uv run run_evaluation.py \
    --ds spider \
    --description_file description.json \
    --type test \
    --task zero_shot_hint \
    --scenario 2
    