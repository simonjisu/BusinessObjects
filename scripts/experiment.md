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
        * 1_gen_sql: Yes / Yes
        * 2_evaluate: Yes / Yes
        * 3_aggregate: Yes / Yes
    * dev-with_bos:
        * 0_retrieve: Yes / Yes
        * 1_gen_sql: Yes / Yes
        * 2_evaluate: Yes / Yes
        * 3_aggregate: Yes / Yes
        * 4_valid_bo: Yes / Yes
    * test-no_bos:
        * 1_gen_sql: Yes / Yes
        * 2_evaluate: Yes / Yes
        * 3_aggregate: Yes / Yes
    * test-with_bos:
        * 0_retrieve: Yes / Yes
        * 1_gen_sql: Yes / Yes
        * 2_evaluate: Yes / Yes
        * 3_aggregate: Yes / Yes
* pipeline inference
    * dev-no_bos:
        * 1_gen_templates: Yes / Yes
        * 2_keyword_extraction: Yes / Yes
        * 3_search_value: Yes / Yes
        * 4_fill_in: Yes / Yes
        * 5_evaluate: Yes / Yes
        * 6_aggregate:  Yes / Yes
    * dev-with_bos:
        * 0_retrieve: Yes / Yes
        * 1_gen_templates: Yes / Yes
        * 2_keyword_extraction: Yes / Yes
        * 3_search_value: Yes / Yes 
        * 4_fill_in: Yes / Yes
        * 5_evaluate: Yes / Yes 
        * 6_aggregate: Yes / 
        * 7_valid_bo: Yes / 
    * test-no_bos:
        * 1_gen_templates: Yes / Yes
        * 2_keyword_extraction: Yes / Yes
        * 3_search_value: Yes / Yes
        * 4_fill_in: Yes / Yes
        * 5_evaluate: Yes / Yes
        * 6_aggregate: Yes / Yes
    * test-with_bos:
        * 0_retrieve: Yes / Yes 
        * 1_gen_templates: Yes / Yes 
        * 2_keyword_extraction: Yes / Yes 
        * 3_search_value: Yes / Yes 
        * 4_fill_in: Yes / Yes 
        * 5_evaluate: Yes / Yes
        * 6_aggregate: Yes / Yes 
```