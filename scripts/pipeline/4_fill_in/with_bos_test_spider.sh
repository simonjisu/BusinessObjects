#!/bin/bash
uv run run_bo_sql.py \
    --ds spider \
    --type test \
    --task fill_in \
    --exp_name pipeline_exp \
    --with_bos \
    --prefix "x-test-with_bos-"