import json
import pandas as pd
import argparse
import sqlparse
from tqdm import tqdm
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnableSequence
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate

from src.prompts import Prompts
from src.pymodels import SQLResponse
from src.db_utils import get_schema_str
from src.spider_sparc_preprocess import process_all_tables
from src.pymodels import SQLResponse, DatabaseModel
from src.database import SqliteDatabase
from src.eval import result_eq, check_if_exists_orderby
from src.eval_complexity import eval_all
from src.process_sql import get_schema, Schema
from src.parsing_sql import (
    extract_selection, 
    extract_condition, 
    extract_aggregation, 
    extract_nested_setoperation, 
    extract_others,
    extract_aliases,
)

_ = load_dotenv(find_dotenv())

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

def run_bo_test_sql(
        proj_path: Path,
        spider_tables: dict[str, DatabaseModel], 
        chain: RunnableSequence,
        vectorstore: FAISS,
        exp_name: str,
        n_retrieval: int = 3,
        score_threshold: float = 0.60
    ):
    final_outputs = []
    df_test = pd.read_csv(proj_path / 'data' / 'split_in_domain' / 'test.csv')
    df_test.reset_index(drop=True, inplace=True)

    # restart from checkpoint
    if list((proj_path / 'experiments' / exp_name).glob('*.json')) != []:
        row_index = sorted([int(file.stem.split('_')[-1]) for file in sorted(list((proj_path / 'experiments' / exp_name).glob('*.json')))])
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
        
    pd.DataFrame(final_outputs).to_csv(proj_path / 'experiments' / 'bo_evals' / f'{exp_name}.csv', index=False)

def get_error_infos(df_test: pd.DataFrame) -> dict:

    iterator = tqdm(df_test.iterrows(), total=len(df_test))
    error_infos = {
        'pred_exec': [],
        'result': [],
        'parsing_sql': [],
        'error_samples': set(),
    }

    test_cols = ['c_low', 'c_mid', 'c_high', 't_1',  't_2',  't_3+']
    for i, x in iterator:
        has_error = False
        schema = get_schema(str(proj_path / 'data' / 'spider' / 'database' / x['db_id'] / f'{x["db_id"]}.sqlite'))
        schema = Schema(schema)
        
        for test_col in test_cols:
            try:
                sql = x[test_col]
                statement = sqlparse.parse(sql.strip())[0]
                aliases = extract_aliases(statement)
                selection = extract_selection(statement, aliases, schema)
                condition = extract_condition(statement, aliases, schema)
                aggregation = extract_aggregation(statement, aliases, schema)
                nested = extract_nested_setoperation(statement)
                others = extract_others(statement, aliases, schema)

            except Exception as e:
                has_error = True
                error_infos['parsing_sql'].append((x['sample_id'], test_col, str(e)))
                error_infos['error_samples'].add(x['sample_id'])
                break
        
        if has_error:
            continue

        iterator.set_description_str(f'error samples {len(error_infos["error_samples"])}')

    print(f'Parsing SQL errors: {len(error_infos["parsing_sql"])}')
    return error_infos

def bo_eval(proj_path: Path, df_test: pd.DataFrame, error_infos: dict):
    test_cols = ['c_low', 'c_mid', 'c_high', 't_1',  't_2',  't_3+']
    eval_cols = ['score', 's_sel', 's_cond', 's_agg', 's_nest', 's_oth']

    df = df_test.loc[~df_test['sample_id'].isin(error_infos['error_samples'])].reset_index(drop=True)
    for test_col in test_cols:
        df_exp = df.loc[:, ['sample_id', 'db_id', 'gold_sql', test_col]]
        iterator = tqdm(df_exp.iterrows(), total=len(df_exp), desc=f'Processing {test_col}')
        # init task eval results
        task_results = {'sample_id': []}
        for col in eval_cols:
            task_results[f'{test_col}_{col}'] = []

        for i, x in iterator:
            task_results['sample_id'].append(x['sample_id'])
            # parsing sql
            schema = get_schema(str(proj_path / 'data' / 'spider' / 'database' / x['db_id'] / f'{x["db_id"]}.sqlite'))
            schema = Schema(schema)
            
            # partial & complexity eval
            parsed_result = {}
            for k in ['gold', 'pred']:
                sql = x[test_col] if k == 'pred' else x['gold_sql']
                statement = sqlparse.parse(sql.strip())[0]
                aliases = extract_aliases(statement)
                selection = extract_selection(statement, aliases, schema)
                condition = extract_condition(statement, aliases, schema)
                aggregation = extract_aggregation(statement, aliases, schema)
                nested = extract_nested_setoperation(statement)
                others = extract_others(statement, aliases, schema)

                parsed_result[k + '_selection'] = selection
                parsed_result[k + '_condition'] = condition
                parsed_result[k + '_aggregation'] = aggregation
                parsed_result[k + '_nested'] = nested
                parsed_result[k + '_others'] = {
                    'distinct': others['distinct'], 
                    'order by': others['order by'], 
                    'limit': others['limit']
                }

            eval_res = eval_all(parsed_result, k=6)
            task_results[f'{test_col}_s_sel'].append(eval_res['score']['selection'])
            task_results[f'{test_col}_s_cond'].append(eval_res['score']['condition'])
            task_results[f'{test_col}_s_agg'].append(eval_res['score']['aggregation'])
            task_results[f'{test_col}_s_nest'].append(eval_res['score']['nested'])
            task_results[f'{test_col}_s_oth'].append(eval_res['score']['others'])
            
            # execution eval
            database = SqliteDatabase(
                str(proj_path / 'data' / 'spider' / 'database' / x['db_id'] / f'{x["db_id"]}.sqlite')
            )
            error_info = ''
            try:
                pred_result = database.execute(x[test_col], rt_pandas=False)
            except Exception as e:
                pred_result = []
                error_info = 'Predction Execution Error:' + str(e)
                score = 0

            try:
                gold_result = database.execute(x['gold_sql'], rt_pandas=False)
            except Exception as e:
                error_info = 'Gold Execution Error:' + str(e)

            if 'Gold Execution Error' in error_info:
                continue
            elif 'Predction Execution Error' in error_info:
                task_results[f'{test_col}_score'].append(score)
                continue
            else:
                exists_orderby = check_if_exists_orderby(x['gold_sql'])
                score = int(result_eq(pred_result, gold_result, order_matters=exists_orderby))
                task_results[f'{test_col}_score'].append(score)

        df_temp = pd.DataFrame(task_results)
        df_test = pd.merge(df_test, df_temp, on='sample_id', how='left')
        df_temp.to_csv(proj_path / 'experiments' / 'bo_evals' / f'{test_col}.csv', index=False)
    df_test.to_csv(proj_path / 'experiments' / 'bo_evals' / f'all_{exp_name}.csv', index=False)

if __name__ == '__main__':
    proj_path = Path('.').resolve()
    assert proj_path.name == 'BusinessObjects', f'Expected project path to be BusinessObjects, but got {proj_path.name}'
    
    parser = argparse.ArgumentParser(description='Zero-shot SQL generation with OpenAI')
    parser.add_argument('--task', type=str, default='zero_shot_hint', help='`zero_shot_hint`, `bo_eval`')
    parser.add_argument('--n_retrieval', type=int, default=3, help='Number of retrievals to consider')
    parser.add_argument('--score_threshold', type=float, default=0.60, help='Score threshold for retrieval')
    parser.add_argument('--percentile', type=int, default=50, help='Percentile to filter by: 25, 50, 75, any other will not call this filter')
    args = parser.parse_args()

    prompt = PromptTemplate(
        template=Prompts.zero_shot_hints_inference,
        input_variables=['schema', 'input_query', 'hint'],
    )

    model_openai = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0.0,
        frequency_penalty=0.1,
    )

    model = model_openai.with_structured_output(SQLResponse)
    chain = (prompt | model)

    # -----------------------------------------------------------------
    # n_retrieval = 3  # 1, 3 
    # score_threshold = 0.60
    # percentile = 50  # 25, 50, 75, any other will not call this filter
    # -----------------------------------------------------------------
    if args.percentile in [25, 50, 75]:
        exp_name = f'test_exp2_{args.percentile}'
    else:
        exp_name = 'test_exp1'
    if not (proj_path / 'experiments' / exp_name).exists():
        (proj_path / 'experiments' / exp_name).mkdir(parents=True)

    if not (proj_path / 'experiments' / 'bo_evals').exists():
        (proj_path / 'experiments' / 'bo_evals').mkdir(parents=True)
    
    if args.task == 'zero_shot_hint':
        with (proj_path / 'data' / 'spider' / f'tables.json').open() as f:
            tables = json.load(f)

        with (proj_path / 'data' / 'description.json').open() as f:
            all_descriptions = json.load(f)

        spider_tables = process_all_tables(tables, descriptions=all_descriptions)
        vectorstore = get_vector_store(proj_path, percentile=args.percentile)

        run_bo_test_sql(
            proj_path, 
            spider_tables, 
            chain, 
            vectorstore, 
            exp_name, 
            n_retrieval=args.n_retrieval, 
            score_threshold=args.score_threshold
        )
    elif args.task == 'bo_eval':
        df_train = pd.read_csv(proj_path / 'data' / 'split_in_domain' / 'spider_bo_desc_train.csv')
        df_test = pd.read_csv(proj_path / 'data' / 'split_in_domain' / 'test.csv')
        df_pred = pd.read_csv(proj_path / 'experiments' / 'bo_evals' / f'{exp_name}.csv')
        df_test = pd.merge(df_test, df_pred, on='sample_id')
        error_infos = get_error_infos(df_test)
        bo_eval(proj_path, df_test, error_infos)

    