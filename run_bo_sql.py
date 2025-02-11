import json
import numpy as np
import pandas as pd
import argparse
import sqlglot
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from itertools import product

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnableSequence
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.callbacks.manager import get_openai_callback
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

from src.prompts import Prompts
from src.pymodels import SQLResponse, DatabaseModel, BirdSample, SpiderSample, BODescription
from src.db_utils import get_schema_str
from src.data_preprocess import process_all_tables, load_raw_data, load_samples_spider_bird
from src.database import SqliteDatabase
from src.parsing_sql import (
    Schema, extract_all, extract_aliases, _format_expression
)
from src.eval_utils import get_complexity

_ = load_dotenv(find_dotenv())

from typing import Iterator
from itertools import product, islice

def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch

class Sampler():
    def __init__(self, bos: dict[str, list[dict]]):
        self.bos = bos
        self.df = _get_df_from_bos(bos)

    def _get_sample_batch(self, db_id: str, n_sample: int=1, n_stop: int=20, seed: int=42):
        np.random.seed(seed)
        sampled = []
        sample_batch = []
        n_sampled = 0
        df_db_id = self.df.loc[(self.df['db_id'] == db_id) & ~self.df['sample_id'].isin(sampled)]
        while df_db_id['category'].nunique() > 0 and n_sampled < n_stop:
            groupby_statement = df_db_id.groupby('category')['sample_id']
            sample_ids = groupby_statement.apply(lambda x: x.sample(min(len(x), n_sample))).tolist()
            if n_sampled + len(sample_ids) > n_stop:
                sample_ids = sample_ids[:(n_stop - n_sampled)]
            sampled.extend(sample_ids)
            sample_batch.append(sample_ids)
            n_sampled += len(sample_ids)
            df_db_id = self.df.loc[(self.df['db_id'] == db_id) & ~self.df['sample_id'].isin(sampled)]

        return sample_batch
    
    def sample(self, db_id: str, n_sample: int=1, n_stop: int=20, seed: int=42, rt_idx: bool=False) -> Iterator:
        sample_batch = self._get_sample_batch(db_id, n_sample, n_stop, seed)
        if rt_idx:
            for sample_ids in sample_batch:
                yield sample_ids
        else:
            for sample_ids in sample_batch:
                s = self.df.loc[(self.df['db_id'] == db_id) & self.df['sample_id'].isin(sample_ids)]
                s = s.to_dict(orient='records')
                for x in s:
                    x['category'] = str(x['category'])
                yield s

def _format_interval(x: pd.Interval):
    return pd.Interval(
            left=int(np.floor(x.left)), 
            right=int(np.floor(x.right)),
            closed=x.closed
    )

def _get_categories(s: pd.Series):
    if s.nunique() == 1:
        return pd.qcut(s, q=[0, 1], duplicates='drop')
    else:
        tiles = [0, 0.2, 0.4, 0.6, 0.8, 1]
        return pd.qcut(s, q=tiles, duplicates='drop')

def _get_categories_with_same_name(s: pd.Series):
    if s.nunique() == 1:
        s = pd.qcut(s, q=[0, 1], duplicates='drop')
        s = s.map({s.cat.categories[i]: str(i) for i in range(len(s.cat.categories))})
        return s
    else:
        tiles = [0, 0.2, 0.4, 0.6, 0.8, 1]
        s = pd.qcut(s, q=tiles, duplicates='drop')
        s = s.map({s.cat.categories[i]: str(i) for i in range(len(s.cat.categories))})
        return s

def _get_df_from_bos(bos):
    df = []
    for db_id, bs in bos.items():
        for b in bs:
            res = {'db_id': db_id}
            res.update(b)
            df.append(res)
    df = pd.DataFrame(df)
    df_cates = df.groupby('db_id')['gold_complexity'].apply(_get_categories)
    df_cates = df_cates.rename('category').apply(_format_interval)
    df = df.merge(df_cates.reset_index('db_id', drop=True), left_index=True, right_index=True)
    return df

def create_vt_ba(
        samples: list[SpiderSample|BirdSample], 
        tables: dict[DatabaseModel],
        chain: RunnableSequence,
        prediction_path: Path,
        file_name: str,
        split_k: int = 3,
    ) -> dict[str, list[dict]]:
    processed_db_ids = [p.stem.split('_', split_k)[-1] for p in prediction_path.glob(f'{file_name}_*')]
    # restart from checkpoint
    if processed_db_ids:
        samples = [sample for sample in samples if sample.db_id not in processed_db_ids]
        print(f'Skip some processed db_ids: {len(processed_db_ids)} {processed_db_ids[-5:]}')

    samples_by_db_id = defaultdict(list)
    for sample in samples:
        samples_by_db_id[sample.db_id].append(sample)

    for db_id, samples in samples_by_db_id.items():
        results = []
        for sample in tqdm(samples, total=len(samples), desc=f"{db_id}"):
            bo = {
                'sample_id': sample.sample_id,
                'gold_sql': sample.final.sql,
            }
            # get complexity
            schema = Schema(tables[db_id].db_schema)
            output = extract_all(sample.final.sql, schema)
            bo['gold_complexity'] = get_complexity(output)

            # get virtual table
            parsed_sql = sqlglot.parse_one(sample.final.sql)
            aliases = extract_aliases(parsed_sql)
            _, expr = _format_expression(parsed_sql, aliases, schema, 
                                        remove_alias=False, anonymize_literal=True)
            bo['vt'] = str(expr)
            
            # get description
            db_schema = get_schema_str(
                schema=tables[db_id].db_schema, 
                foreign_keys=tables[db_id].foreign_keys,
                col_explanation=tables[db_id].col_explanation
            )
            input_data = {'schema': db_schema, 'virtual_table': str(expr)}
            o = chain.invoke(input_data)
            bo['ba'] = o.description

            results.append(bo)
            
        with (prediction_path / f'{file_name}_{db_id}.json').open('w') as f:
            json.dump(results, f, indent=4)

def get_vector_store(bos: dict[str, list[dict[str, str]]], is_question_query: bool=False) -> FAISS:
    documents = []
    for db_id, samples in bos.items():
        for x in samples:
            doc = Document(
                page_content=x['ba'] if not is_question_query else x['question'],
                metadata={
                    'sample_id': x['sample_id'],
                    'db_id': db_id,
                    'vt': x['vt']
                }
            )
            documents.append(doc)
    embeddings_model = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(
        documents, 
        embedding = embeddings_model,
    )
    return vector_store

def get_retriever(
        vectorstore: FAISS,
        db_id: str='',
        cross_encoder: HuggingFaceCrossEncoder=None,
        n_retrieval: int = 3,
        k_retrieval: int = 10,
        score_threshold: float = 0.60,
        use_reranker: bool = False
    ) -> BaseRetriever:
    k_retrieval = k_retrieval if use_reranker else n_retrieval
    base_retriever = vectorstore.as_retriever(
        search_type='similarity_score_threshold',
        search_kwargs={
            'k': k_retrieval, 
            'score_threshold': score_threshold, 
            'filter': {'db_id': db_id},
        }
    )
    if use_reranker:
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=n_retrieval)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
    else:
        retriever = base_retriever
    return retriever

def valid_bo(
        samples: list[SpiderSample|BirdSample],
        tables: dict[DatabaseModel],
        bos: dict[str, list[dict[str, str]]],
        chain: RunnableSequence,
        prediction_path: Path,
        file_name: str = '[args.ds]_[args.type]',
        split_k: int = 2,
    ):
    # "[file_name]_[db_id]-[idx_bo]"
    # restart from checkpoint
    processed_files = [p.stem.split('_', split_k)[-1] for p in prediction_path.glob(f'{file_name}_*')]
    if processed_files:
        bos = dict([x for x in bos.items() if x[0] not in processed_files])
        print(f'Skip some processed db_ids: {len(processed_files)} {processed_files[-5:]}')

    for detail_file_name, train_bos in bos.items():
        set_llm_cache(SQLiteCache(database_path=f"./cache/{prediction_path.stem}_{detail_file_name}.db"))
        train_bos = train_bos['train_bos']
        db_id = detail_file_name.split('-')[0]    
        schema_str = get_schema_str(
            schema=tables[db_id].db_schema, 
            foreign_keys=tables[db_id].foreign_keys,
            col_explanation=tables[db_id].col_explanation
        )
        results = []
        x_samples = list(filter(lambda x: x.db_id == db_id, samples))
        iterator = list(product(train_bos, x_samples))
        iterator = tqdm(iterator, total=len(iterator), desc=f"{detail_file_name}")
        for bo, sample in iterator:
            res = {
                'sample_id': sample.sample_id,
                'gold_sql': sample.final.sql,
                'question': sample.final.question,
                'retrieved': bo['sample_id'],
            }
            question = sample.final.question
            hint = '\nDescriptions and Virtual Tables:\n'
            hint += json.dumps({'description': bo['ba'], 'virtual_table': bo['vt']}, indent=4)
            hint += '\n'
            input_data = {'schema': schema_str, 'input_query': question, 'hint': hint}
            with get_openai_callback() as cb:
                output = chain.invoke(input=input_data)

            res.update({
                'rationale': output.rationale,
                'pred_sql': output.full_sql_query,
                'token_usage': {'tokens': cb.total_tokens, 'cost': cb.total_cost}
            })
            results.append(res)

        with open(prediction_path / f'{file_name}_{detail_file_name}.json', 'w') as f:
            json.dump(results, f, indent=4)

def predict_sql_bo(
        samples: list[SpiderSample|BirdSample],
        tables: dict[DatabaseModel],
        test_bos: dict[str, dict[str, list[str]|int]],
        chain: RunnableSequence,
        prediction_path: Path,
        file_name: str = '[args.ds]_[args.type]_[args.scenario]',
        split_k: int = 3,
        k_retrieval: int = 5,  # for test
        n_retrieval: int = 1,   # for test
        score_threshold: float = 0.65,
        use_reranker: bool = False,
        is_question_query: bool = False
    ):
    processed_db_ids = [p.stem.split('_', split_k)[-1] for p in prediction_path.glob(f'{file_name}_*')]
    # restart from checkpoint
    if processed_db_ids:
        samples = [sample for sample in samples if sample.db_id not in processed_db_ids]
        print(f'Skip some processed db_ids: {len(processed_db_ids)} {processed_db_ids[-5:]}')

    samples_by_db_id = defaultdict(list)
    for sample in samples:
        samples_by_db_id[sample.db_id].append(sample)

    if use_reranker:
        cross_encoder = HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
    else:
        cross_encoder = None

    # use train set to evaluate development set
    for db_id, samples in samples_by_db_id.items():
        set_llm_cache(SQLiteCache(database_path=f"./cache/{prediction_path.stem}_{file_name}_{db_id}.db"))
        print(f'[{db_id}] Creating vector store...')
        vectorstore = get_vector_store(test_bos, is_question_query)
        retriever = get_retriever(
            vectorstore, db_id, cross_encoder,
            n_retrieval, k_retrieval, score_threshold, use_reranker
        )
        schema_str = get_schema_str(
            schema=tables[db_id].db_schema, 
            foreign_keys=tables[db_id].foreign_keys,
            col_explanation=tables[db_id].col_explanation
        )
        results = []
        for sample in tqdm(samples, total=len(samples), desc=f"{db_id}"):
            question = sample.final.question
            docs = retriever.invoke(question)
            hint = '\nDescriptions and Virtual Tables:\n'
            hint += json.dumps({j: {'description': doc.page_content, 'virtual_table': doc.metadata['vt']} for j, doc in enumerate(docs)}, indent=4)
            hint += '\n'
            input_data = {'schema': schema_str, 'input_query': question, 'hint': hint}
    
            with get_openai_callback() as cb:
                output = chain.invoke(input=input_data)
            
            res = {}
            res['sample_id'] = sample.sample_id
            res['rationale'] = output.rationale
            res['pred_sql'] = output.full_sql_query
            res['retrieved'] = [doc.metadata['sample_id'] for doc in docs]
            res['token_usage'] = {'tokens': cb.total_tokens, 'cost': cb.total_cost}
            results.append(res)

        with open(prediction_path / f'{file_name}_{db_id}.json', 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot SQL generation with OpenAI')
    parser.add_argument('--task', type=str, default='zero_shot_hint', help='`create_bo`, `zero_shot_hint`, `bo_eval`')
    parser.add_argument('--ds', type=str, default='spider', help='Dataset to use for training. spider or bird') 
    parser.add_argument('--type', type=str, default='train', help='Type of data to use for .')
    parser.add_argument('--description_file', type=str, default='description.json', help='File containing the descriptions.')
    parser.add_argument('--k_retrieval', type=int, default=5, help='Number of retrievals for reranker to consider')
    parser.add_argument('--n_retrieval', type=int, default=1, help='Number of retrievals to consider')
    parser.add_argument('--n_sample', type=int, default=3, help='[type=dev] Number of samples to consider')
    parser.add_argument('--n_stop', type=int, default=50, help='[type=dev] Number of samples to stop')
    parser.add_argument('--score_threshold', type=float, default=0.60, help='Score threshold for retrieval')
    parser.add_argument('--use_reranker', action='store_true', help='Whether to use reranker or not')
    # valid_bo
    parser.add_argument('--db_id_group', type=int, default=-1, help='Group of db_ids to consider, < 0 means use all')
    # test_bo
    parser.add_argument('--scenario', type=int, default='0', help='Scenario to consider')
    # scenario
    # (ba-query) 0: 10 | 1: 15 | 2: 25 | (question-query) 4: 25
    
    args = parser.parse_args()

    proj_path = Path('.').resolve()
    assert proj_path.name == 'BusinessObjects', f'Expected project path to be BusinessObjects, but got {proj_path.name}'
    
    experiment_folder = proj_path / 'experiments' / args.ds
    prediction_path = experiment_folder / 'predictions' / args.task
    eval_path = experiment_folder / 'evals' / args.task
    for p in [prediction_path, eval_path]:
        if not p.exists():
            p.mkdir(parents=True)

    # must load components
    tables, *_ = load_raw_data(proj_path / 'data' / args.ds, load_test=False)
    with (proj_path / 'data' / args.description_file).open() as f:
        all_descriptions = json.load(f)
    tables = process_all_tables(tables, descriptions=all_descriptions)

    model_openai = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0.0,
        frequency_penalty=0.1,
    )

    if args.task == 'create_bo':
        samples = load_samples_spider_bird(proj_path / 'data' / f'{args.ds}_{args.type}.json')

        prompt = PromptTemplate(
            template=Prompts.bo_description,
            input_variables=['schema', 'virtual_table']
        )
        
        model = model_openai.with_structured_output(BODescription)
        chain = (prompt | model)

        create_vt_ba(
            samples=samples, tables=tables, chain=chain,
            prediction_path=prediction_path, file_name=f'{args.ds}_{args.type}_bo', split_k=3)
        
        bos = defaultdict(list)
        for p in prediction_path.glob(f'{args.ds}_{args.type}_bo_*.json'):
            with p.open() as f:
                temp = json.load(f)
            bos[p.stem.split('_', 3)[-1]] = temp

        with (prediction_path / f'final_{args.ds}_{args.type}_bo.json').open('w') as f:
            json.dump(bos, f, indent=4)

    elif args.task == 'valid_bo_prepare_batch_run':
        samples = load_samples_spider_bird(proj_path / 'data' / f'{args.ds}_{args.type}.json')
        df = pd.read_csv(experiment_folder / 'evals' / 'zero_shot' / f'bird_dev.csv')
        df_score = df.loc[:, ['sample_id', 'db_id', 'exec_result']]
        df_error = df_score.loc[df_score['exec_result'] == 0, ['db_id', 'sample_id']]
        error_ids = df_error['sample_id'].tolist()
        samples = list(filter(lambda x: x.sample_id in error_ids, samples))
        
        with open(experiment_folder / f'partial_{args.ds}_db_ids.json') as f:
            partial_db_ids = json.load(f)
        
        bo_path = experiment_folder / 'predictions' / 'create_bo' / f'final_{args.ds}_train_bo.json'
        assert bo_path.exists(), 'Run with the `task=create_bo, type=train` first'
        with bo_path.open() as f:
            bos = json.load(f)

        with open(experiment_folder / f'partial_{args.ds}_db_ids.json', 'w') as f:
            json.dump(partial_db_ids, f, indent=4)
        
        sampler = Sampler(bos)
        
        sampled_bos = {}
        for db_id_group in partial_db_ids:
            sampled_bos[str(db_id_group)] = defaultdict()
            for db_id in partial_db_ids[str(db_id_group)]:
                x_samples = list(filter(lambda x: x.db_id == db_id, samples))
                for idx_bos, train_bos in enumerate(sampler.sample(db_id, args.n_sample, args.n_stop, rt_idx=False)):
                    # print(f'{db_id}-{idx_bos} :', f'{len(train_bos)}', f'{len(list(product(train_bos, x_samples)))}')
                    sampled_bos[str(db_id_group)][f'{db_id}-{idx_bos}'] = {
                        'train_bos': train_bos,
                        'n_iter': len(list(product(train_bos, x_samples))), 
                        'total_bos_in_batch': len(train_bos),
                        'total_samples_in_batch': len(x_samples)
                    }

        with (experiment_folder / f'partial_{args.ds}_batch.json').open('w') as f:
            json.dump(sampled_bos, f, indent=4)

    elif args.task == 'valid_bo':
        # use error sample to validate
        samples = load_samples_spider_bird(proj_path / 'data' / f'{args.ds}_{args.type}.json')
        df = pd.read_csv(experiment_folder / 'evals' / 'zero_shot' / f'{args.ds}_dev.csv')
        df_error = df.loc[df['exec_result'] == 0]
        error_ids = df_error['sample_id'].tolist()
        samples = list(filter(lambda x: x.sample_id in error_ids, samples))

        bo_path = experiment_folder / 'predictions' / 'create_bo' / f'final_{args.ds}_train_bo.json'
        assert bo_path.exists(), 'Run with the `task=create_bo, type=train` first'
        with bo_path.open() as f:
            bos = json.load(f)

        batch_file_path = experiment_folder / f'partial_{args.ds}_batch.json'
        # assert batch_file_path.exists(), 'Run with the `task=create_bo, type=train` first'
        if (args.db_id_group >= 0) and batch_file_path.exists():
            with batch_file_path.open() as f:
                sampled_bos = json.load(f)[str(args.db_id_group)]  
            # dict["db_id-idx_bo", dict["train_bos", "n_iter", "total_bos_in_batch", "total_samples_in_batch"]]
        else:
            raise KeyError('Run with the `task=valid_bo_prepare_batch_run` first')
        
        # filter samples with db_ids gorup
        with open(experiment_folder / f'partial_{args.ds}_db_ids.json') as f:
            partial_db_ids = json.load(f)
        samples = list(filter(lambda x: x.db_id in partial_db_ids[str(args.db_id_group)], samples))
        print(f'{args.ds}-{args.type} samples loaded: {len(samples)}')
        
        prompt = PromptTemplate(
            template=Prompts.zero_shot_hints_inference,
            input_variables=['schema', 'input_query', 'hint'],
        )

        model = model_openai.with_structured_output(SQLResponse)
        chain = (prompt | model)

        valid_bo(
            samples=samples, 
            tables=tables, 
            bos=sampled_bos, 
            chain=chain,
            prediction_path=prediction_path, 
            file_name=f'{args.ds}_{args.type}', 
            split_k=2,
        )

    elif args.task == 'zero_shot_hint':
        bo_path = experiment_folder / 'predictions' / 'create_bo' / f'final_{args.ds}_train_bo.json'
        with bo_path.open() as f:
            all_bos = json.load(f)

        # test_scenarios
        with (experiment_folder / 'test_scenarios.json').open('r') as f:
                test_scenarios = json.load(f)
            
        sce = {0: "10", 1: "15", 2: "25", 3: "25"}[args.scenario]
        test_bo_ids = test_scenarios[sce]
        test_bos = defaultdict(list)
        for db_id, bos in all_bos.items():
            if db_id in test_bo_ids:
                bo_ids = test_bo_ids[db_id]
                test_bos[db_id].extend(list(filter(lambda x: x['sample_id'] in bo_ids, bos)))
        
        # (bo-query)
        if args.scenario in (0, 1, 2):
            is_question_query = False
            print('bo-query scenario')
        # (question-query)
        elif args.scenario in (3,):
            is_question_query = True
            print('question-query scenario')
        else:
            print(f'Invalid scenario: {args.scenario}')
            raise ValueError('Invalid scenario')
        
        # args.type == test
        samples = load_samples_spider_bird(proj_path / 'data' / f'{args.ds}_{args.type}.json')
        samples = [x for x in samples if x.db_id in test_bo_ids]
        print(f'{args.ds}-{args.type} samples loaded: {len(samples)}')
        
        prompt = PromptTemplate(
            template=Prompts.zero_shot_hints_inference,
            input_variables=['schema', 'input_query', 'hint'],
        )

        model = model_openai.with_structured_output(SQLResponse)
        chain = (prompt | model)

        predict_sql_bo(
            samples=samples, 
            tables=tables, 
            test_bos=test_bos, 
            chain=chain,
            prediction_path=prediction_path, 
            file_name=f'{args.ds}_{args.type}_{args.scenario}', 
            split_k=3,
            k_retrieval=args.k_retrieval,
            n_retrieval=args.n_retrieval,
            score_threshold=args.score_threshold,
            use_reranker=args.use_reranker,
            is_question_query=is_question_query
        )