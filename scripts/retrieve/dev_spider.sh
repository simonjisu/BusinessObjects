#!/bin/bash
uv run run_bo_sql.py \
    --ds spider \
    --type dev \
    --task retrieve \
    --description_file description.json \
    --k_retrieval 5 \
    --n_retrieval 1 \
    --use_reranker \
    --embedding_model custom \
    --reranker_model custom