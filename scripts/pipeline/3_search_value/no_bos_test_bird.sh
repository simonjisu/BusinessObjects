#!/bin/bash
uv run run_bo_sql.py \
    --ds bird \
    --type test \
    --task search_value \
    --exp_name pipeline_exp \
    --prefix "x-test-"