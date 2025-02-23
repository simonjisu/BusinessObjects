#!/bin/bash
uv run run_bo_sql.py \
    --ds bird \
    --type test \
    --task gen_template \
    --exp_name pipeline_exp \
    --prefix "x-test-no_bos-"