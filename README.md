# Business Objects: A Practical Data-Centric AI Approach to Improving Usability of NL2SQL for Enterprises

Using natural language (NL) as an interface for querying databases has gained significant attention in the database research community, as it enhances usability by allowing end users to directly access databases. While many model-centric AI approaches have been developed to improve accuracy, they have yet to be widely deployed in real-world enterprise applications. A key bottleneck is the lack of rich metadata for AI models to work with. In this paper, we introduce a complementary, practical, data-centric AI approach: Business Objects (BOs), which enhances the usability of NL2SQL in enterprise settings. BOs reduce query complexity by integrating virtual tables and business abstracts, allowing large language models (LLMs) to generate more accurate SQL queries. This cost-effective and scalable approach focuses on continuous data refinement rather than costly model modifications. 
We demonstrate how to select BOs from many candidates by using training and development set to select BOs with new metric called Merit, which can capture both structural and semantic similarity score between two SQLs. Our evaluation on benchmark datasets (BIRD and Spider) demonstrates that BOs improve execution accuracy by 5.79\% and 4.58\%, respectively, when compared to queries generated without BOs. Our transition analysis reveals that an average of 44.87\% of databases across both model-centric methods experience a 25\% relative improvement in execution accuracy when BOs are introduced, highlighting their effectiveness in enhancing SQL query performance.

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
```

### Pretrained Retrieval and Reranking models

You can download the pretrained retrieval and reranking models from [here](https://drive.google.com/file/d/1kxXleTQnfjTbShB-MmUANBgG0DX_DiWq/view?usp=drive_link) and place them in the `models` directory.

Pleae check `train_retrieval_model.py` to train the custom models.

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
    └── tpch.db  # after running `python src/tpch_preprocess.py`
```

# Codes

## run_bo_sql.py

```bash
$ python run_bo_sql.py --help
```

please check the `scripts` directory for the detailed scripts for the experiments.


## TPC-H Preliminary Study

```bash
$ python eval_tpch.py \
    --llm gpt-4o-mini \
    --ref_query all \
    --option 0
```

* **llm**: The LLM to use, it can be `gpt-4o`, `gpt-4o-mini`, `gemini-1.5-pro`,...
* **ref_query**: The TPC-H query to work on, 'all' for all queries
* **option**: 0: database schema only, 1: with business abstract, 2: with Specific BO, 3: with General BO

Check the code for preliminary study with TPC-H in `eval_tpch.py`. Please check our paper for more details explanation.

### TPC-H Preliminary Study Error Study

<image src="./figs/preliminary_error_analysis.png">