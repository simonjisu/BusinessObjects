import json
import pandas as pd
from typing import Optional
from itertools import pairwise

from src.spider_sparc_preprocess import process_all_tables
from pathlib import Path
from src.db_utils import get_schema_str
from tqdm import tqdm
from src.spider_sparc_preprocess import DatabaseModel, SpiderSample, SpiderSample, QuestionSQL, BusinessObject
from dotenv import load_dotenv, find_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

_ = load_dotenv(find_dotenv())
proj_path = Path('.').resolve()

def filter_by_pm_score(x: pd.Series, df_pm_stats: pd.DataFrame, percentile: int):
    rank_criteria = df_pm_stats.loc[x['db_id'], f'{percentile}%']
    return x['pm_score_rank'] < rank_criteria

def get_vector_store(proj_path, percentile: Optional[int]=100):
    df_train = pd.read_csv(proj_path / 'data' / 'split_in_domain' / f'spider_bo_desc_train.csv')
    if percentile in [25, 50, 75]:
        df_pm_stats = df_train.groupby(['db_id'])['pm_score_rank'].describe().loc[:, ['25%', '50%', '75%']]
        pm_idx = df_train.apply(lambda x: filter_by_pm_score(x, df_pm_stats, percentile), axis=1)
        df_train = df_train.loc[pm_idx].reset_index(drop=True)

    documents = []
    for i, row in df_train.iterrows():
        doc = Document(
            doc_id=row['sample_id'],
            page_content=row['description'],
            metadata={
                'sample_id': row['sample_id'],
                'db_id': row['db_id'],
                'cate_gold_c': row['cate_gold_c'],
                'cate_len_tbls': row['cate_len_tbls'],
                'virtual_table': row['virtual_table']
            }
        )
        documents.append(doc)

    embeddings_model = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(
        documents, 
        embedding = embeddings_model,
        distance_strategy = DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    return vectorstore


class Response(BaseModel):
    output: str = Field(description='The full SQL query.')
    rationale: list[str] = Field(description='The step-by-step reasoning to generate the SQL query. ')

template = '''### TASK
You are tasked with generating a SQL query(in a SQLite Database) according to a user input NL question.
You should work in step-by-step reasoning before coming to the full SQL query.

### SCHEMA
You are working with the following schema in a SQLite Database:
{schema}

### HINT
You will be provided a hint to help you. It is called "virtual table".
You will get either descriptions of the virtual tables or descriptions and templates of virtual tables together.
You can use or modify the hint to generate the full SQL query.

### FORMATTING
Your output should be of the following JSON format:
{{
    "rationale": "<list[str]: the step-by-step reasoning to generate the SQL query>",
    "full_sql_query": "<str: the full SQL query>"
}}

### OUTPUT
<INPUT QUERY>: {input_query}
<HINT>: {hint}
<OUTPUT>: 
'''

prompt = PromptTemplate(
    template=template,
    input_variables=['schema', 'input_query', 'hint'],
)

model_openai = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0.0,
    frequency_penalty=0.1,
)

model = model_openai.with_structured_output(Response)
chain = (prompt | model)

# -----------------------------------------------------------------
n_retrieval = 3  # 1, 3 
score_threshold = 0.60
percentile = 25  # 25, 50, 75, any other will not call this filter
# -----------------------------------------------------------------
if percentile in [25, 50, 75]:
    exp_name = f'test_exp2_{percentile}'
else:
    exp_name = 'test_exp1'
if not (proj_path / 'experiments' / exp_name).exists():
    (proj_path / 'experiments' / exp_name).mkdir(parents=True)

with (proj_path / 'data' / 'spider' / f'tables.json').open() as f:
    tables = json.load(f)

with (proj_path / 'data' / 'description.json').open() as f:
    all_descriptions = json.load(f)

spider_tables = process_all_tables(tables, descriptions=all_descriptions)
vectorstore = get_vector_store(proj_path, percentile=percentile)

final_outputs = []
df_test = pd.read_csv(proj_path / 'data' / 'split_in_domain' / 'test.csv')
df_test.reset_index(drop=True, inplace=True)

# restart from checkpoint
if list((proj_path / 'experiments' / exp_name).glob('*.json')) != []:
    row_index = [int(file.stem.split('_')[-1]) for file in sorted(list((proj_path / 'experiments' / exp_name).glob('*.json')))]
    df_test = df_test.iloc[row_index[-1]:]
    final_outputs = json.load((proj_path / 'experiments' / exp_name / f'{df_test.iloc[0]["sample_id"]}_{row_index[-1]}.json').open())

iterator = tqdm(df_test.iterrows(), total=len(df_test))
for i, row in iterator:
    o = {'sample_id': row['sample_id']}

    db_schema = get_schema_str(
        schema=spider_tables[row['db_id']].db_schema, 
        foreign_keys=spider_tables[row['db_id']].foreign_keys,
        col_explanation=spider_tables[row['db_id']].col_explanation
    )
    
    # Experiment Complexity: low, mid, high
    iterator.set_description(f"Processing {row['sample_id']}: Complexity - low, mid, high")
    filter_key = 'cate_gold_c'
    for filter_value in ['low', 'mid', 'high']:
        retriever = vectorstore.as_retriever(
            search_kwargs={'k': n_retrieval, 'score_threshold': score_threshold, 'filter': {filter_key: filter_value, 'db_id': row['db_id']}}
        )
        docs = retriever.invoke(row['question'])
        hint = 'Descriptions and Virtual Tables:\n'
        hint += json.dumps({j: {'description': doc.page_content, 'virtual_table': doc.metadata['virtual_table']} for j, doc in enumerate(docs)}, indent=4)
        hint += '\n'
        input_data = {'schema': db_schema, 'input_query': row['question'], 'hint': hint}
        output = chain.invoke(input=input_data)

        o[f'c_{filter_value}'] = output.output
        o[f'c_{filter_value}_hint'] = hint

    # Experiment Complexity: 1, 2, 3+
    iterator.set_description(f"Processing {row['sample_id']}: Complexity - 1, 2, 3+")
    filter_key = 'cate_len_tbls'
    for filter_value in ['1', '2', '3+']:
        retriever = vectorstore.as_retriever(
            search_kwargs={'k': n_retrieval, 'score_threshold': score_threshold, 'filter': {filter_key: filter_value, 'db_id': row['db_id']}}
        )
        docs = retriever.invoke(row['question'])
        hint = 'Descriptions and Virtual Tables:\n'
        hint += json.dumps({j: {'description': doc.page_content, 'virtual_table': doc.metadata['virtual_table']} for j, doc in enumerate(docs)}, indent=4)
        hint += '\n'
        input_data = {'schema': db_schema, 'input_query': row['question'], 'hint': hint}
        output = chain.invoke(input=input_data)

        o[f't_{filter_value}'] = output.output
        o[f't_{filter_value}_hint'] = hint
    final_outputs.append(o)

    if i % 100 == 0:
        with (proj_path / 'experiments' / exp_name / f'{row["sample_id"]}_{i}.json').open('w') as f:
            json.dump(final_outputs, f, indent=4)
    
df_final = pd.DataFrame(final_outputs).to_csv(proj_path / 'experiments' / 'bo_evals' / f'{exp_name}.csv', index=False)