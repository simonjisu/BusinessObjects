#!/bin/bash
uv run run_zeroshot.py \
    --ds bird \
    --type test \
    --model gpt-4o-mini \
    --task zero_shot \
    --description_file bird_description.json