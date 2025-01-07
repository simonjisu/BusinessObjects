#!/bin/bash
uv run run_zeroshot.py \
    --ds spider \
    --type dev \
    --model gpt-4o-mini \
    --task zero_shot \
    --description_file description.json