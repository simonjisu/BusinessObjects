#!/bin/bash
uv run run_bo_sql.py \
    --ds spider \
    --type test \
    --task zero_shot_hint \
    --description_file description.json \
    --k_retrieval 5 \
    --n_retrieval 1 \
    --score_threshold 0.6 \
    --scenario 1 \
    --use_reranker