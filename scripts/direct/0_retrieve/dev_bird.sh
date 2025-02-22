#!/bin/bash
uv run run_bo_sql.py \
    --ds bird \
    --type dev \
    --task retrieve \
    --exp_name direct_exp \
    --k_retrieval 10 \
    --n_retrieval 3 \
    --use_reranker \
    --embedding_model custom \
    --reranker_model custom \
    --with_bos