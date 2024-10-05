import json
from tqdm import tqdm
from .db_utils import get_schema_str, get_data_dict
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from pydantic import BaseModel, Field
# from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Optional
from collections import defaultdict

import sqlglot
import sqlglot.expressions as exp
from sqlglot.optimizer import optimize
# https://github.com/tobymao/sqlglot/blob/main/sqlglot/optimizer/optimizer.py

from sqlglot.optimizer.annotate_types import annotate_types
from sqlglot.optimizer.canonicalize import canonicalize
from sqlglot.optimizer.eliminate_ctes import eliminate_ctes
from sqlglot.optimizer.eliminate_joins import eliminate_joins
from sqlglot.optimizer.eliminate_subqueries import eliminate_subqueries
from sqlglot.optimizer.merge_subqueries import merge_subqueries
from sqlglot.optimizer.normalize import normalize
from sqlglot.optimizer.optimize_joins import optimize_joins
from sqlglot.optimizer.pushdown_predicates import pushdown_predicates
from sqlglot.optimizer.pushdown_projections import pushdown_projections
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.qualify_columns import quote_identifiers
from sqlglot.optimizer.simplify import simplify
from sqlglot.optimizer.unnest_subqueries import unnest_subqueries
from sqlglot.schema import ensure_schema

RULES = (
    # qualify,
    # pushdown_projections,
    # normalize,
    # unnest_subqueries,
    # pushdown_predicates,
    # optimize_joins,
    # eliminate_subqueries,
    # merge_subqueries,
    # # eliminate_joins,
    # eliminate_ctes,
    # quote_identifiers,
    # annotate_types,
    # canonicalize,
    # simplify,
)


class DatabaseModel(BaseModel):
    db_id: str
    db_schema: dict[str, dict[str, str]]
    col_explanation: dict[str, dict[str, str]]
    foreign_keys: list[str]
    primary_keys: list[str]

class QuestionSQL(BaseModel):
    question: str
    sql: str
    source_tables: list[str] = []

class SparcSample(BaseModel):
    sample_id: int = -1
    db_id: str
    interactions: list[QuestionSQL]
    final: QuestionSQL
    
class BusinessObject(BaseModel):
    obj_id: str
    obj_name: Optional[str] = None
    virtual_table: str
    description: str

class SpiderSample(BaseModel):
    sample_id: int|str = None
    db_id: str
    final: QuestionSQL
    bo: Optional[BusinessObject] = None


def load_spider_sparc_data(data_path: Path, load_test: bool=False) -> tuple:
    with (data_path / f'tables.json').open() as f:
        data_tables = json.load(f)
    with (data_path / f'train.json').open() as f:
        train_data = json.load(f)
    with (data_path / f'dev.json').open() as f:
        dev_data = json.load(f)

    if load_test:
        with (data_path / f'test.json').open() as f:
            test_data = json.load(f)
        return data_tables, train_data, dev_data, test_data
    return data_tables, train_data, dev_data

def preprocess_sql(sql: str) -> str:
    return sql.replace('"', "'").strip()
    # return sql.strip()

def process_all_tables(tables: list, descriptions: Optional[dict[str, dict[str, str]]]=None) -> dict[str, DatabaseModel]:
    database = defaultdict(DatabaseModel)
    for table in tables:
        db_id = table['db_id']
        data_dict = get_data_dict(table)
        if descriptions is not None:
            col_exps = descriptions[db_id]
        else:
            col_exps = data_dict['col_explanation']
        database[db_id] = DatabaseModel(
            db_id=db_id,
            db_schema=data_dict['schema'],
            col_explanation=col_exps,
            foreign_keys=data_dict['foreign_keys'],
            primary_keys=data_dict['primary_keys']
        )
    return database

def filter_samples_by_count_sparc(all_data: dict, n: int=5) -> list:
    counter = defaultdict(int)
    for data in all_data:
        db_id = data['database_id']
        counter[db_id] += 1
    all_data = list(filter(lambda x: counter[x['database_id']] >= n, all_data))
    return all_data

def extract_used_table(sql: str, schema: dict) -> list[str]:
    sql = sqlglot.parse_one(sql, read='sqlite')
    # sql = optimize(sqlglot.parse_one(sql, read='sqlite'), schema=schema, rules=RULES)
    tbls = set([x.this.this.lower() for x in list(sql.find_all(exp.Table))])
    return tbls

def process_samples_sparc(all_data: list[dict], tables: dict[str, DatabaseModel], skip: Optional[list]=[]) -> dict[str, list[SparcSample]]:
    data_by_db_id = defaultdict(list)
    for i, data in tqdm(enumerate(all_data), total=len(all_data)):
        if i in skip:
            continue
        db_id = data['database_id']
        schema = tables[db_id].db_schema

        interactions = []
        for x in data['interaction']:
            sql = preprocess_sql(x['query'])
            tbls = extract_used_table(sql, schema)
            interactions.append(QuestionSQL(question=x['utterance'], sql=sql, source_tables=tbls))

        
        final_sql = preprocess_sql(data['final']['query'])
        try:
            final_tbls = extract_used_table(final_sql, schema)
        except Exception as e:
            print(f'Warning Skipped: {db_id} - {i}')
            continue

        final = QuestionSQL(question=data['final']['utterance'], sql=final_sql, source_tables=final_tbls)

        sample = SparcSample(
            sample_id=i,
            db_id=db_id,
            interactions=interactions,
            final=final
        )
        data_by_db_id[db_id].append(sample)

    return data_by_db_id

def filter_samples_by_count_spider(all_data: dict, n: int=5) -> list:
    counter = defaultdict(int)
    for data in all_data:
        db_id = data['db_id']
        counter[db_id] += 1
    all_data = list(filter(lambda x: counter[x['db_id']] >= n, all_data))
    return all_data

def process_samples_spider(all_data: list, tables: dict[str, DatabaseModel], skip: Optional[list]) -> dict[str, list[SparcSample]]:
    data_by_db_id = defaultdict(list)
    for i, data in tqdm(enumerate(all_data), total=len(all_data)):
        if i in skip:
            continue
        db_id = data['db_id']
        schema = tables[db_id].db_schema

        final_sql = preprocess_sql(data['query'])
        try:
            final_tbls = extract_used_table(final_sql, schema)
        except Exception as e:
            print(f'Warning Skipped: {db_id} - {i}')
            continue

        sample = SpiderSample(
            sample_id=i,
            db_id=db_id,
            final=QuestionSQL(question=data['question'], sql=final_sql, source_tables=final_tbls)
        )

        data_by_db_id[db_id].append(sample)
    return data_by_db_id

def split_train_dev_test(data_samples: dict, train_ratio: float=0.8, dev_ratio: float=0.1) -> tuple:
    assert 1.0 - train_ratio - dev_ratio > 0, 'Not enough samples for test set'
    train_samples = []
    dev_samples = []
    test_samples = []
    for db_id, samples in data_samples.items():
        n_train = int(len(samples) * train_ratio)
        n_dev = int(len(samples) * dev_ratio)
        n_test = len(samples) - n_train - n_dev
        assert n_test > 0, f'Not enough samples for test set: {db_id}'
        assert len(samples) == (len(samples[:n_train]) + len(samples[n_train:(n_dev+n_train)]) + len(samples[n_train+n_dev:])), f'Error: {db_id}'
        train_samples.extend(samples[:n_train])
        dev_samples.extend(samples[n_train:(n_dev+n_train)])
        test_samples.extend(samples[n_train+n_dev:])
    return train_samples, dev_samples, test_samples

def get_sparc_schema_description(proj_path: Path, sparc_tables: dict) -> dict:

    class Description(BaseModel):
        output: dict[str, dict[str, str]] = Field(description='Description of each column for all tables in the database')

    template = '''### Task
    You are tasked with writing one line short description for each column name in a database to help users understand the data better.
    You will be proveded a schema with table names and column names.

    ### Formatting
    Your output should be of the following JSON format with `output` key and value as a dictionary of table names and column names with their descriptions.:
    {{
        "<table_name1>" : {{
            "<column_name>": <str: the one line short description of column>,
            ...
        }},
        ...
    }} 

    ### Output
    <SCHEMA>:\n{schema}
    <OUTPUT>: 
    '''

    prompt = PromptTemplate(
        template=template,
        input_variables=['schema']
    )

    model_openai = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0.2,
    )

    chain = (prompt | model_openai | JsonOutputParser(pydantic_object=Description))

    all_descriptions = {}
    for db_id, database_model in tqdm(sparc_tables.items(), total=len(sparc_tables)):
        schema_desc = chain.invoke(input={'schema': get_schema_str(database_model.db_schema)})
        all_descriptions[db_id] = schema_desc

    with (proj_path / 'data' / 'description.json').open('w') as f:
        json.dump(all_descriptions, f, indent=4)

def save_samples_spider(samples: list[SpiderSample], path: Path):
    with path.open('w') as f:
        for s in samples:
            data = {
                'sample_id': s.sample_id,
                'db_id': s.db_id,
                'final': {
                    'question': s.final.question,
                    'sql': s.final.sql,
                    'source_tables': s.final.source_tables
                }
            }
            data_json = json.dumps(data)
            f.write(data_json+'\n')

def load_samples_spider(path: Path) -> list[SpiderSample]:
    samples = []
    with path.open() as f:
        for line in f:
            data = json.loads(line)
            sample = SpiderSample(
                sample_id=data['sample_id'],
                db_id=data['db_id'],
                final=QuestionSQL(
                    question=data['final']['question'],
                    sql=data['final']['sql'],
                    source_tables=data['final']['source_tables']
                )
            )
            samples.append(sample)
    return samples

def save_samples_sparc(samples: list[SparcSample], path: Path):
    with path.open('w') as f:
        for s in samples:
            data = {
                'sample_id': s.sample_id,
                'db_id': s.db_id,
                'interactions': [],
                'final': {
                    'question': s.final.question,
                    'sql': s.final.sql,
                    'source_tables': s.final.source_tables
                }
            }
            for i in s.interactions:
                data['interactions'].append({
                    'question': i.question,
                    'sql': i.sql,
                    'source_tables': i.source_tables
                })
            data_json = json.dumps(data)
            f.write(data_json+'\n')

def load_samples_sparc(path: Path) -> list[SparcSample]:
    samples = []
    with path.open() as f:
        for line in f:
            data = json.loads(line)
            interactions = []
            for i in data['interactions']:
                interactions.append(QuestionSQL(
                    question=i['question'],
                    sql=i['sql'],
                    source_tables=i['source_tables']
                ))
            sample = SparcSample(
                sample_id=data['sample_id'],
                db_id=data['db_id'],
                interactions=interactions,
                final=QuestionSQL(
                    question=data['final']['question'],
                    sql=data['final']['sql'],
                    source_tables=data['final']['source_tables']
                )
            )
            samples.append(sample)
    return samples

if __name__ == '__main__':
    _ = load_dotenv(find_dotenv())
    
    proj_path = Path('.').resolve()
    sparc_path = proj_path / 'data' / 'sparc'
    with (proj_path / 'data' / 'description.json').open() as f:
        all_descriptions = json.load(f)

    tables, train_data, dev_data = load_spider_sparc_data(sparc_path)
    print(f'Number of train: {len(train_data)} | Number of dev: {len(dev_data)}')
    
    sparc_tables = process_all_tables(tables)
    # filter samples by count, must have at least 5 samples
    all_data = filter_samples_by_count_sparc(train_data+dev_data, n=5)
    # process samples -> {db_id: list of samples}
    sparc_samples = process_samples_sparc(all_data)
    # change train/dev by sample
    train_samples, dev_samples, test_samples = split_train_dev_test(sparc_samples, train_ratio=0.8, dev_ratio=0.1)
    
    save_samples_sparc(train_samples, proj_path / 'data' / 'sparc_train.json')
    save_samples_sparc(dev_samples, proj_path / 'data' / 'sparc_dev.json')

    # get description
    get_sparc_schema_description(proj_path, sparc_tables)

    # --------------------------------------------------------------------------------------------
    # spider dataset
    spider_path = proj_path / 'data' / 'spider'
    tables, train_data, dev_data = load_spider_sparc_data(spider_path)

    with (proj_path / 'data' / 'description.json').open() as f:
        all_descriptions = json.load(f)
    spider_tables = process_all_tables(tables, descriptions=all_descriptions)

    all_data = filter_samples_by_count_spider(train_data+dev_data, n=10)
    # process samples -> {db_id: list of samples}
    skip = [3146, 4690, 4691]
    spider_samples = process_samples_spider(all_data, spider_tables, skip=skip)
    # change train/dev by sample
    train_samples, dev_samples, test_samples = split_train_dev_test(spider_samples, train_ratio=0.8, dev_ratio=0.1)
    print(f'Number of train: {len(train_samples)} | Number of dev: {len(dev_samples)}')

    save_samples_spider(train_samples, proj_path / 'data' / 'spider_train.json')
    save_samples_spider(dev_samples, proj_path / 'data' / 'spider_dev.json')