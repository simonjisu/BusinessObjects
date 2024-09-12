import json
from tqdm import tqdm
from .db_utils import get_schema_str, get_data_dict
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from collections import defaultdict
# from pydantic import BaseModel

class DatabaseModel(BaseModel):
    db_id: str
    db_schema: dict[str, dict[str, str]]
    col_explanation: dict[str, str]
    foreign_keys: list[str]
    primary_keys: list[str]

class QuestionSQL(BaseModel):
    question: str
    sql: str

class SparcSample(BaseModel):
    sample_id: int = -1
    db_id: str
    interactions: list[QuestionSQL]
    final: QuestionSQL

def load_sparc_data(data_path: Path):
    with (data_path / f'tables.json').open() as f:
        data_tables = json.load(f)
    with (data_path / f'train.json').open() as f:
        train_data = json.load(f)
    with (data_path / f'dev.json').open() as f:
        dev_data = json.load(f)
    return data_tables, train_data, dev_data

def preprocess_sql(sql: str) -> str:
    return sql.replace('"', "'").strip()

def process_all_tables(tables: list) -> dict[str, DatabaseModel]:
    database = defaultdict(DatabaseModel)
    for table in tables:
        db_id = table['db_id']
        data_dict = get_data_dict(table)
        database[db_id] = DatabaseModel(
            db_id=db_id,
            db_schema=data_dict['schema'],
            col_explanation=data_dict['col_explanation'],
            foreign_keys=data_dict['foreign_keys'],
            primary_keys=data_dict['primary_keys']
        )
    return database

def filter_samples_by_count(all_data: dict, n: int=5) -> list:
    counter = defaultdict(int)
    for data in all_data:
        db_id = data['database_id']
        counter[db_id] += 1
    all_data = list(filter(lambda x: counter[x['database_id']] >= n, all_data))
    return all_data

def process_samples(all_data: list) -> dict[str, list[SparcSample]]:
    data_by_db_id = defaultdict(list)
    for i, data in enumerate(all_data):
        db_id = data['database_id']
        sample = SparcSample(
            sample_id=i,
            db_id=db_id,
            interactions=[
                QuestionSQL(
                    question=x['utterance'], 
                    sql=preprocess_sql(x['query'])) for x in data['interaction']
            ],
            final=QuestionSQL(
                question=data['final']['utterance'], 
                sql=preprocess_sql(data['final']['query']), 
            )
        )
        data_by_db_id[db_id].append(sample)
    return data_by_db_id

def split_train_dev(sparc_samples: dict, ratio: float=0.8):
    train_samples = []
    dev_samples = []
    for db_id, samples in sparc_samples.items():
        n_train = int(len(samples) * ratio)
        assert len(samples[n_train:]) > 0, f'Not enough samples for dev set: {db_id}'
        train_samples.extend(samples[:n_train])
        dev_samples.extend(samples[n_train:])
    return train_samples, dev_samples

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

    with (proj_path / 'db_data' / 'sparc_description.json').open('w') as f:
        json.dump(all_descriptions, f, indent=4)


if __name__ == '__main__':
    _ = load_dotenv(find_dotenv())
    
    proj_path = Path('.').resolve()
    sparc_path = proj_path / 'data' / 'sparc'

    tables, train_data, dev_data = load_sparc_data(sparc_path)
    print(f'Number of train: {len(train_data)} | Number of dev: {len(dev_data)}')
    
    sparc_tables = process_all_tables(tables)
    # filter samples by count, must have at least 5 samples
    all_data = filter_samples_by_count(train_data+dev_data, n=5)
    # process samples -> {db_id: list of samples}
    sparc_samples = process_samples(all_data)
    # change train/dev by sample
    train_samples, dev_samples = split_train_dev(sparc_samples, ratio=0.8)


    get_sparc_schema_description(proj_path, sparc_tables)