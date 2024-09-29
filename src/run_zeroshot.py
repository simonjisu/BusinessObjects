import json
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from src.db_utils import get_schema_str
from src.database import SqliteDatabase
from src.spider_sparc_preprocess import (
    load_spider_sparc_data,
    process_all_tables, 
    load_samples_spider,
    load_samples_sparc,
    filter_samples_by_count_sparc,
    filter_samples_by_count_spider, 
    process_samples_sparc,
    process_samples_spider, 
    split_train_dev_test,
    save_samples_spider
)


import os 
from .spider_sparc_preprocess import DatabaseModel, SpiderSample
from dotenv import load_dotenv, find_dotenv
from collections import defaultdict
from tqdm import tqdm
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
# from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Models
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

_ = load_dotenv(find_dotenv())

class Response(BaseModel):
    full_sql_query: str = Field(description='The full SQL query.')
    rationale: list[str] = Field(description='The step-by-step reasoning to generate the SQL query. Each step has ')

template = '''### TASK
You are tasked with generating a SQL query(in a SQLite Database) according to a user input request.
You should work in step-by-step reasoning before coming to the full SQL query.

You will be provided an input NL query.

### SCHEMA
You are working with the following schema in a SQLite Database:
{schema}

### FORMATTING
Your output should be of the following JSON format:
{{
    "rationale": "<list[str]: the step-by-step reasoning to generate the SQL query>",
    "full_sql_query": "<str: the full SQL query>"
}}

### OUTPUT
<INPUT QUERY>: {input_query}
<OUTPUT>: 
'''

def predict_sql(samples: list[SpiderSample], spider_tables: dict[str, DatabaseModel], chain: RunnableSequence, k: int = 500, file_name: str = 'full_sql_output') -> list[dict]:
    all_full_sql = list()
    for i, data in tqdm(enumerate(samples), total=len(samples)):
        db_schema = get_schema_str(
            schema=spider_tables[data.db_id].db_schema, 
            foreign_keys=spider_tables[data.db_id].foreign_keys,
            col_explanation=spider_tables[data.db_id].col_explanation
        )
        input_data = {'schema': db_schema, 'input_query': data.final.question}
        output = chain.invoke(input=input_data)

        full_sql_output = {}
        full_sql_output['sample_id'] = data.sample_id
        full_sql_output['db_id'] = data.db_id
        full_sql_output['question'] = data.final.question
        full_sql_output['rationale'] = output.rationale
        full_sql_output['pred_sql'] = output.full_sql_query
        full_sql_output['gold_sql'] = data.final.sql
        full_sql_output['source_tables'] = data.final.source_tables
        all_full_sql.append(full_sql_output)

        if len(all_full_sql) == k:
            with open(proj_path / 'experiments' / f'{file_name}_{i//k}.jsonl', 'w') as f:
                for d in all_full_sql:
                    f.write(json.dumps(d) + '\n')
            all_full_sql = list()

    if len(all_full_sql) > 0:
        with open(proj_path / 'experiments' / f'{file_name}_{i//k}.jsonl', 'w') as f:
            for d in all_full_sql:
                f.write(json.dumps(d) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot SQL generation with OpenAI')
    parser.add_argument('--ds', type=str, default='spider', help='Dataset to use for training.')
    parser.add_argument('--table_file', type=str, default='tables.json', help='File containing the tables.')
    parser.add_argument('--description_file', type=str, default='description.json', help='File containing the descriptions.')
    parser.add_argument('--type', type=str, default='train', help='Type of data to use for .')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='Model to use for training.')
    args = parser.parse_args()

    proj_path = Path('.').resolve()

    model_dict = {
        'gpt-4o-mini': ChatOpenAI,
        'google-generative-ai': ChatGoogleGenerativeAI
    }

    with (proj_path / 'data' / args.ds / args.table_file).open() as f:
        tables = json.load(f)

    with (proj_path / 'data' / args.description_file).open() as f:
        all_descriptions = json.load(f)
    spider_tables = process_all_tables(tables, descriptions=all_descriptions)

    samples = load_samples_spider(proj_path / 'data' / f'{args.ds}_{args.type}.json')
    print(f'{args.ds}-{args.type} samples loaded: {len(samples)}')    
    prompt = PromptTemplate(
        template=template,
        input_variables=['schema', 'input_query']
    )



    model_openai = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0.0,
    )

    model = model_openai.with_structured_output(Response)
    chain = (prompt | model)


