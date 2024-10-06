import json
import pandas as pd
from typing import Optional

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

# ----------------------------------------------
def predict_sql(
        samples: list[SpiderSample], 
        spider_tables: dict[str, DatabaseModel], 
        chain: RunnableSequence, 
        task: str, 
        k: int = 500, 
        file_name: str = 'full_sql_output',
        vectorstore: Optional[FAISS] = None,
        n_retrieval: int = 1,
        score_threshold: float = 0.65,
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
        if task == 'sql_gen_zero_shot':
            input_data = {'schema': db_schema, 'input_query': data.final.question}
        elif 'bo_desc_gen' in task:
            input_data = {'schema': db_schema, 'virtual_table': data.bo.virtual_table}
        elif 'sql_gen_hint' in task:
            # name hint: 'sql_gen_hint_top{n_retrieval}_[low/mid/high or 1/2/3+]_[desc/descvt]'
            assert vectorstore is not None, 'vectorstore is required for hint generation task.'
            assert len(task.split('_')) == 6, f'Invalid task - {task} name hint: sql_gen_hint_top{n_retrieval}_[low/mid/high or 1/2/3+]_[desc/descvt]"'
            hint_type = task.split('_')[-1]
            filter_type = task.split('_')[-2]
            assert filter_type in ['low', 'mid', 'high', '1', '2', '3+'], f'Invalid filter type: {filter_type}'
            assert hint_type in ['desc', 'descvt'], f'Invalid hint type: {hint_type}'

            retriever = vectorstore.as_retriever(
                search_kwargs={'k': n_retrieval, 'score_threshold': score_threshold, 'filter': {'level': filter_type, 'db_id': data.db_id}}
            )
            docs = retriever.invoke(data.final.question)
            hint = ''
            if len(docs) != 0:
                if hint_type == 'desc':
                    hint += 'Descriptions:\n'
                    hint += json.dumps({j: doc.page_content for j, doc in enumerate(docs)}, indent=4)
                elif hint_type == 'descvt':
                    hint += 'Descriptions and Virtual Tables:\n'
                    hint += json.dumps({j: {'description': doc.page_content, 'virtual_table': doc.metadata['virtual_table']} for j, doc in enumerate(docs)}, indent=4)
            hint += '\n'
            input_data = {'schema': db_schema, 'input_query': data.final.question, 'hint': hint}
        else:
            raise ValueError(f'Invalid task: {task}')
        
        output = chain.invoke(input=input_data)

        final_output['sample_id'] = data.sample_id
        final_output['db_id'] = data.db_id
        final_output['question'] = data.final.question
        final_output['rationale'] = output.rationale
        final_output['gold_sql'] = data.final.sql
        final_output['source_tables'] = data.final.source_tables

        if task == 'sql_gen_zero_shot':
            final_output['pred_sql'] = output.output
        elif 'bo_desc_gen' in task:
            final_output['description'] = output.output
            final_output['virtual_table'] = data.bo.virtual_table
        elif 'sql_gen_hint' in task:
            final_output['pred_sql'] = output.output
            final_output['hint'] = input_data['hint']
        else:
            raise ValueError(f'Invalid task: {task}')
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
                source_tables=eval(row['source_tables']),
            ),
            bo=BusinessObject(
                obj_id=row['sample_id'],
                virtual_table=row['virtual_table'],
                description='' if row.get('description') is None else row['description']
            )
        )
        data.append(sample)
    return data

def run_bo_sql_gen(
        proj_path, task, spider_tables, chain, type_exp,
        vectorstore: FAISS, n_retrieval: int = 3, score_threshold: float = 0.65
    ):
    df = pd.read_csv(proj_path / 'data' / 'spilt_in_domain' / f'spider_bo_desc{type_exp}.csv')
    samples = get_bo_sample_data(df)
    predict_sql(samples, spider_tables, chain, task, 
                k=100, 
                file_name=f'bo_sql_output{type_exp}',
                vectorstore=vectorstore,
                n_retrieval=n_retrieval,
                score_threshold=score_threshold)

    bos = []
    for p in sorted((proj_path / 'experiments' / task).glob(f'bo_sql_output{type_exp}_*.jsonl'), key=lambda x: int(x.stem.split('_')[-1])):
        with p.open() as f:
            for line in f:
                bos.append(json.loads(line))

    with (proj_path / 'experiments' / f'{task}.jsonl').open('w') as f:
        for bo in bos:
            f.write(json.dumps(bo) + '\n')

def get_vector_store(proj_path, typ):
    df_train = pd.read_csv(proj_path / 'data' / 'spilt_in_domain' / f'spider_bo_desc{typ}_train.csv')
    documents = []
    for i, row in df_train.iterrows():
        if typ == '_c':
            if row['need_low|wrong']:
                level = 'low'
            elif row['need_mid|wrong']:
                level = 'mid'
            elif row['need_high|wrong']:
                level = 'high'
            else:
                raise ValueError('The complexity level is not defined.')
        elif typ == '_t':
            if row['need_1|wrong']:
                level = '1'
            elif row['need_2|wrong']:
                level = '2'
            elif row['need_3+|wrong']:
                level = '3+'
            else:
                raise ValueError('The complexity level is not defined.')
        else:
            raise ValueError('Invalid type (`typ`)')

        doc = Document(
            doc_id=row['sample_id'],
            page_content=row['description'],
            metadata={
                'sample_id': row['sample_id'],
                'db_id': row['db_id'],
                'question': row['question'],
                'gold_sql': row['gold_sql'],
                'source_tables': row['source_tables'],
                'cate_len_tbls': row['cate_len_tbls'],
                'gold_c': row['gold_c'],
                'level': level,
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
typ = '_c'  # '_t', '_c'
score_threshold = 0.65
iterator = ['low', 'mid', 'high'] if typ == '_c' else ['1', '2', '3+']
# -----------------------------------------------------------------

with (proj_path / 'data' / 'spider' / f'tables.json').open() as f:
    tables = json.load(f)

with (proj_path / 'data' / 'description.json').open() as f:
    all_descriptions = json.load(f)

spider_tables = process_all_tables(tables, descriptions=all_descriptions)
vectorstore = get_vector_store(proj_path, typ)

for typ2 in ['desc', 'descvt']:
    for n_retrieval in [1, 3]:
        for level in iterator:
            # ----------------------------------------------
            if typ2 == 'desc' or n_retrieval == 1:
                continue
            # ----------------------------------------------
            task = f'sql_gen_hint_top{n_retrieval}_{level}_{typ2}'
            print(task)
            # name hint: 'sql_gen_hint_top{n_retrieval}_[low/mid/high or 1/2/3+]_[desc/descvt]'
            if not (proj_path / 'experiments' / task).exists():
                (proj_path / 'experiments' / task).mkdir()
            run_bo_sql_gen(
                proj_path, task, spider_tables, chain, 
                type_exp=f'{typ}_test',
                vectorstore=vectorstore,
                n_retrieval=n_retrieval,
                score_threshold=score_threshold
            )