#!/bin/bash
uv run run_bo_sql.py \
    --ds spider \
    --type test \
    --task fill_in \
    --exp_name pipeline_exp \
    --prefix "x-test-no_bos-"