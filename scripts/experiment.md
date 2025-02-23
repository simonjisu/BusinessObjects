# experiments

```
no_bos_direct: without BOs + direct inference
with_bos_direct: with BOs + direct inference
no_bos_pipeline: without BOs + pipeline inference
with_bos_pipeline: with BOs + pipeline inference

** with_bos: run `0_retrieve` / no_bos: do not run `0_retrieve`
** pipeline: 1_gen_templates > 2_keyword_extraction > 3_search_value > 4_fill_in
```

experiment folder should be two: `direct_exp` and `pipeline_exp`

if run with_bos, only gen_template will have different prompts and input arguments than others.

BIRD / Spider

* direct inference: 
    * dev-no_bos: 
        * 1_gen_sql: Yes /
        * 2_evaluate: Yes /
        * 3_aggregate: Yes /
    * dev-with_bos:
        * 0_retrieve: Yes /
        * 1_gen_sql: Yes / 
        * 2_evaluate: Yes /
        * 3_aggregate: Yes /
        * 4_valid_bo: Yes / 
    * test-no_bos:
        * 1_gen_sql: Yes /
        * 2_evaluate: Yes /
        * 3_aggregate: Yes /
    * test-with_bos:
        * 0_retrieve: Yes /
        * 1_gen_sql: Yes /
        * 2_evaluate: 
        * 3_aggregate: 
* pipeline inference
    * dev-no_bos:
        * 1_gen_templates: Yes / 
        * 2_keyword_extraction: Yes / 
        * 3_search_value: Yes /
        * 4_fill_in: Yes /
        * 5_evaluate: Yes /
        * 6_aggregate:  Yes /
    * dev-with_bos:
        * 0_retrieve: Yes / 
        * 1_gen_templates: Yes / 
        * 2_keyword_extraction: Yes /
        * 3_search_value: Yes /
        * 4_fill_in: Yes / 
        * 5_evaluate: Yes / --> 문제 있음 다시 돌리는중
        * 6_aggregate: 
        * 7_valid_bo: 
    * test-no_bos:
        * 1_gen_templates: Yes / 
        * 2_keyword_extraction: Yes / 
        * 3_search_value: Yes / 
        * 4_fill_in: Yes / 
        * 5_evaluate: 
        * 6_aggregate: 
    * test-with_bos:
        * 0_retrieve: 
        * 1_gen_templates: 
        * 2_keyword_extraction: 
        * 3_search_value: 
        * 4_fill_in: 
        * 5_evaluate: 
        * 6_aggregate: 
```