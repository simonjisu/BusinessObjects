import json
import sqlparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

from src.db_utils import get_schema_str
from src.data_preprocess import process_all_tables
from src.prompts import Prompts
from src.parsing_sql import tks, Statement
from src.pymodels import (
    DatabaseModel, 
    SpiderSample, 
    QuestionSQL, 
    BusinessObject,
    BODescription
)

_ = load_dotenv(find_dotenv())
LITERALS = [tks.Literal.String.Single, tks.Literal.String.Symbol, tks.Literal.Number.Integer, tks.Literal.Number.Float, tks.Literal.Number.Hexadecimal]
LITERAL_TYPES = dict([(t, str(t.__dict__['parent']).split('.')[-1].upper()) for t in LITERALS])

def get_virtual_table(sql: str):    
    parsed = sqlparse.parse(sql)[0]
    tokens = []
    for token in parsed.flatten():
        if token.ttype in LITERALS:
            token.value = f"[PLACEHOLDER-TYPE:{LITERAL_TYPES[token.ttype]}]"
        tokens.append(token)
    stmt = Statement(tokens)
    return str(stmt)

def bo_desc_gen(
        samples: list[SpiderSample], 
        spider_tables: dict[str, DatabaseModel], 
        chain: RunnableSequence, 
        task: str, 
        k: int = 500, 
        file_name: str = 'full_sql_output',
    ) -> list[dict]:
    # check index exists, then start from the last index
    check_files = list((proj_path / 'experiments' / task).glob(f'{file_name}_*.jsonl'))
    check_ids = [int(x.stem.split('_')[-1]) for x in check_files]
    if len(check_ids) > 0:
        i = (max(check_ids)+1) * k
        samples = samples[i:]
    else:
        i = 0
    
    all_outputs = list()
    for i, data in tqdm(enumerate(samples, i), total=len(samples)):
        db_schema = get_schema_str(
            schema=spider_tables[data.db_id].db_schema, 
            foreign_keys=spider_tables[data.db_id].foreign_keys,
            col_explanation=spider_tables[data.db_id].col_explanation
        )
        final_output = {}
        input_data = {'schema': db_schema, 'virtual_table': data.bo.virtual_table}
        output = chain.invoke(input=input_data)

        final_output['sample_id'] = data.sample_id
        final_output['rationale'] = output.rationale

        final_output['description'] = output.output
        all_outputs.append(final_output)

        if len(all_outputs) == k:
            with open(proj_path / 'experiments' / task / f'{file_name}_{i//k}.jsonl', 'w') as f:
                for d in all_outputs:
                    f.write(json.dumps(d) + '\n')
            all_outputs = list()

    if len(all_outputs) > 0:
        with open(proj_path / 'experiments' / task / f'{file_name}_{i//k}.jsonl', 'w') as f:
            for d in all_outputs:
                f.write(json.dumps(d) + '\n')


def get_bo_sample_data(df):
    data = []
    for i, row in df.iterrows():
        sample = SpiderSample(
            sample_id=row['sample_id'],
            db_id=row['db_id'],
            final=QuestionSQL(
                question=row['question'],
                sql=row['gold_sql'],
                source_tables=[],
            ),
            bo=BusinessObject(
                obj_id=row['sample_id'],
                virtual_table=row['virtual_table'],
                description='' if row.get('description') is None else row['description']
            )
        )
        data.append(sample)
    return data

def run_bo_desc_gen(proj_path, task, spider_tables, chain, type_exp='train', output_file='spider_bo_desc'):    
    df = pd.read_csv(proj_path / 'data' / 'split_in_domain' / f'{type_exp}.csv')
    df['virtual_table'] = df['gold_sql'].apply(get_virtual_table)
    samples = get_bo_sample_data(df)
    bo_desc_gen(
        samples, 
        spider_tables, 
        chain, 
        task, 
        k=100, 
        file_name=f'bo_desc_output_{type_exp}'
    )

    bos = []
    for p in sorted((proj_path / 'experiments' / task).glob(f'bo_desc_output_{type_exp}_*.jsonl'), key=lambda x: int(x.stem.split('_')[-1])):
        with p.open() as f:
            for line in f:
                bos.append(json.loads(line))

    with (proj_path / 'experiments' / f'{output_file}_{type_exp}.jsonl').open('w') as f:
        for bo in bos:
            f.write(json.dumps(bo) + '\n')

    # load the data
    descs = dict()
    with (proj_path / 'experiments' / f'{output_file}_{type_exp}.jsonl').open('r') as f:
        for line in f:
            x = json.loads(line)
            descs[x['sample_id']] = x['description']

    df['description'] = df['sample_id'].map(descs)

    # ranking the pm scores
    pm_cols = ['s_sel', 's_cond', 's_agg', 's_nest', 's_oth']
    df['pm_score'] = df.loc[:, pm_cols].sum(axis=1)
    # ranking with the pm scores
    df['pm_score_rank'] = df.groupby(['db_id'])['pm_score'].rank(
        method='min', ascending=True).astype(np.int64)
    
    df.to_csv(proj_path / 'data' / 'split_in_domain' / f'{output_file}_{type_exp}.csv', index=False)
    

if __name__ == '__main__':
    proj_path = Path('.').resolve()
    assert proj_path.name == 'BusinessObjects', f'Expected project path to be BusinessObjects, but got {proj_path.name}'

    task = f'bo_desc_gen'

    with (proj_path / 'data' / 'spider' / f'tables.json').open() as f:
        tables = json.load(f)

    with (proj_path / 'data' / 'description.json').open() as f:
        all_descriptions = json.load(f)
        
    spider_tables = process_all_tables(tables, descriptions=all_descriptions)

    if not (proj_path / 'experiments' / task).exists():
        (proj_path / 'experiments' / task).mkdir()

    prompt = PromptTemplate(
        template=Prompts.bo_description,
        input_variables=['schema', 'sql_template'],
    )

    model_openai = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0.0,
    )

    model = model_openai.with_structured_output(BODescription)
    chain = (prompt | model)

    run_bo_desc_gen(
        proj_path, 
        task, 
        spider_tables, 
        chain, 
        type_exp='train', 
        output_file='spider_bo_desc'
    )