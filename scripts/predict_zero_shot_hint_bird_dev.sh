#!/bin/bash
uv run run_bo_sql.py \
    --ds bird \
    --type dev \
    --task zero_shot_hint \
    --description_file bird_description.json \
    --k_retrieval 8 \
    --n_retrieval 2 \
    --score_threshold 0.6 \
    --use_reranker 