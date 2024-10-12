# Business Objects: A Practical Data-Centric AI Approach to Improving Usability of NL2SQL for Enterprises

Using natural language (NL) as an interface for querying databases has gained significant attention in the database research community, as it enhances usability by allowing end users to directly access databases. While many model-centric AI approaches have been developed to improve accuracy, they have yet to be widely deployed in real-world enterprise applications. A key bottleneck is the lack of rich metadata for AI models to work with. In this paper, we introduce a complementary, practical, data-centric AI approach: Business Objects (BOs), which enhances the usability of NL2SQL in enterprise settings. BOs reduce query complexity by integrating virtual tables and business abstracts, allowing large language models (LLMs) to generate more accurate SQL queries. This cost-effective and scalable approach focuses on continuous data refinement rather than costly model modifications. Through extensive experiments using the TPC-H benchmark, BOs achieved nearly 2-times improvement in execution accuracy for complex real-world business queries. Additionally, with the Spider dataset, we demonstrate how SQL complexity—both in structure and number of tables—affects NL2SQL performance, suggesting complexity as a criterion for selecting BOs. Finally, a qualitative case study on BO selection highlights both sides of this approach in enterprise settings.

## Setup

Download the spider dataset from [here](https://yale-lily.github.io/spider) and place it in the `data` directory. We need the `tables.json` file and `database` folder for execution. 

Please install `uv` for running all the files ([link](https://docs.astral.sh/uv/getting-started/installation/)). 

```bash
$ uv venv --python 3.11
$ uv sync
```

You will need to have OpenAI api-key and Vertexai(Gemini) to run the code. Please create a `.env` file in the root directory and add the following line:

```
OPENAI_API_KEY = "" 
VERTEXAI_PROJECT_NAME = ""
```

### TPC-H

You can download pre-processed TPC-H data from [here](https://drive.google.com/file/d/1DdBnQorpF6gkD2n89ySP1wTkXRy0fVeL/view?usp=sharing) to create the TPC-H database. Place it under the `data / tpch` folder. Then you can run `python src/tpch_preprocess.py` to create the TPC-H database.

The structure of the `data / tpch` folder should be as follows:

```
data
└── tpch
    ├── bo/
    ├── gold/
    ├── tables/  # TPC-H tables
    ├── ...
    └── tpch.db  # after funning `python src/tpch_preprocess.py`
```

# Codes

## Zero-shot Inference: Spider

```bash
$ python run_zero_shot.py \ 
    --ds "spider" \
    --table_file "tables.json" \
    --description_file "description.json" \
    --type "train" \
    --model "gpt-4o-mini" \
    --task "zero_shot" 
```

* **task**: run as following procedure (1) `zero_shot`, (2) `post_process`, and (3) `output_result_plus` for each train and dev
* **model**: `gpt-4o-mini` available

## Analysis for Zero-shot Inference: Spider

Check the code for the analysis of the zero-shot inference results in `analysis.py`. Please check our paper for more details explanation.

### Relationship between Execution Accuracy and Complexities

<image src="./figs/complexity_vs_score.png" width=80%>

### Relationship between Number of Tables and Structural Complexities

<image src="./figs/n_tbls_sc.png" width=80%>

## Structural Complexity Score Function

We explored the best-fit structural complexity function. The best-fit function is using normalized `tanh` function: 

$$F = \text{tanh} \bigg( \log ( 1 + \sum_i \dfrac{\Vert f_i \Vert}{k} ) \bigg)$$

### Factor Distribution given two numbers: a and b

<image src="./figs/score_function_test.png" width=70%>

### Input and Output Values plots at k=6

<image src="./figs/score_function_test2.png" width=70%>

* Left: Sum of features(x) with normalized values $f(x)$
* Mid: normalized values $f(x)$ and $\log(1+x)$
* Right: $\log(1+x)$ and $\text{tanh} \big(\log(1+x) \big)$

## Train and Test set split

We set the train dataset with the corrected samples in zero-shot inference and the unique queries(`gold_sql`) for BOs creation. The rest of incorrect samples are used for the test set. 

* Number of unique train queries(BOs): 3,489
* Number of test queries: 2,126

Following figures are the distribution of the score by the number of tables and structural complexities at zero-shot inference stage.

### Score Distribution by Number of Tables Complexity (160 DBs)

<image src="./figs/score_dist_num_tbls.png">

### Score Distribution by Structural Complexity (160 DBs)

<image src="./figs/score_dist_strutral.png">

## Business Object Experiment

### BO Creation

Create the BO with train set.

```bash
$ python bo_creation.py
```

## Zero-shot Inference with BOs: Spider

```bash
$ python run_bo_sql.py \ 
    --type "zero_shot_hint" \
    --n_retrieval 3 \
    --score_threshold 0.6 \
    --percentile 100
```

* **task**: run as following procedure (1) `zero_shot_hint` and (2) `bo_eval`.
* **percentile**: Filter by the ranking of percentile of partial matching score in train set: 25, 50, 75, any other will not call this filter.

Check the code for the analysis of the zero-shot inference with BOs results in `result_analysis.ipynb`. Please check our paper for more details explanation.

### Comparison of Execution Accuracy in the test set by Retrieval Augmented Generation and the Number of BOs

<image src="./figs/num_bo.png">


