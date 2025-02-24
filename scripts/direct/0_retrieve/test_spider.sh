#!/bin/bash
uv run run_bo_sql.py \
    --ds spider \
    --type test \
    --task retrieve \
    --exp_name direct_exp \
    --k_retrieval 10 \
    --n_retrieval 1 \
    --use_reranker \
    --embedding_model custom \
    --reranker_model custom \
    --with_bos \
    --prefix "x-test-"