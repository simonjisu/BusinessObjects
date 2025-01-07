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
    tiles = [0, 0.2, 0.4, 0.6, 0.8, 1]
    df = pd.qcut(s, q=tiles, duplicates='drop')
    return df

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

def get_vector_store(bos: dict[str, list[dict[str, str]]]):
    documents = []
    for db_id, samples in bos.items():
        for x in samples:
            doc = Document(
                page_content=x['ba'],
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
        cross_encoder: HuggingFaceCrossEncoder,
        db_id: str,
        sample_ids: list[int],
        n_retrieval: int = 3,
        k_retrieval: int = 10,
        score_threshold: float = 0.60,
        use_reranker: bool = True
    ) -> BaseRetriever:
    k_retrieval = k_retrieval if use_reranker else n_retrieval
    base_retriever = vectorstore.as_retriever(
        search_type='similarity_score_threshold',
        search_kwargs={
            'k': k_retrieval, 
            'score_threshold': score_threshold, 
            'filter': {'db_id': db_id, 'sample_id': {'$in' : sample_ids}},
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
        bos = [x for x in bos.items() if x[0] not in processed_files]
        print(f'Skip some processed db_ids: {len(processed_files)} {processed_files[-5:]}')

    for detail_file_name, train_bos in bos.items():
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
        bos: dict[str, dict[str, list[str]|int]],
        chain: RunnableSequence,
        prediction_path: Path,
        file_name: str = '[args.ds]_[args.type]',
        split_k: int = 2,
        k_retrieval: int = 5,  # for test
        n_retrieval: int = 1,   # for test
        score_threshold: float = 0.65,
        use_reranker: bool = True
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
        cross_encoder = HuggingFaceCrossEncoder(model='cross-encoder/ms-marco-MiniLM-L-6-v2')
    else:
        cross_encoder = None


    # use train set to evaluate development set
    for db_id, samples in samples_by_db_id.items():
        vectorstore = get_vector_store(bos)
        retriever = get_retriever(
            vectorstore, cross_encoder, db_id, 
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
            # full_sql_output = 1
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
    
    parser.add_argument('--db_id_group', type=int, default=-1, help='Group of db_ids to consider, < 0 means use all')
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
        dev_samples = load_samples_spider_bird(proj_path / 'data' / f'{args.ds}_{args.type}.json')
        with open(experiment_folder / f'partial_{args.ds}_db_ids.json') as f:
            partial_db_ids = json.load(f)
        
        bo_path = experiment_folder / 'predictions' / 'create_bo' / f'final_{args.ds}_train_bo.json'
        assert bo_path.exists(), 'Run with the `task=create_bo, type=train` first'
        with bo_path.open() as f:
            bos = json.load(f)

        db_ids = list(bos.keys())
        partial_db_ids = {}
        n = 20
        for i in range(30):
            if db_ids[i*n:(i+1)*n]:
                partial_db_ids[i] = db_ids[i*n:(i+1)*n]

        with open(experiment_folder / f'partial_{args.ds}_db_ids.json', 'w') as f:
            json.dump(partial_db_ids, f, indent=4)
        
        sampler = Sampler(bos)
        
        sampled_bos = {}
        for db_id_group in partial_db_ids:
            sampled_bos[str(db_id_group)] = defaultdict()
            for db_id in partial_db_ids[str(db_id_group)]:
                x_samples = list(filter(lambda x: x.db_id == db_id, dev_samples))
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
        samples = load_samples_spider_bird(proj_path / 'data' / f'{args.ds}_{args.type}.json')
        
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
        bo_path = experiment_folder / 'predictions' / 'valid_bo' / f'final_{args.ds}_bo.json'
        assert bo_path.exists(), f'Run with the `task=valid_bo, type=dev` first'
        with bo_path.open() as f:
            bos = json.load(f)

        samples = load_samples_spider_bird(proj_path / 'data' / f'{args.ds}_{args.type}.json')
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
            bos=bos, 
            chain=chain,
            prediction_path=prediction_path, 
            file_name=f'{args.ds}_{args.type}', 
            split_k=2,
            k_retrieval=args.k_retrieval,
            n_retrieval=args.n_retrieval,
            score_threshold=args.score_threshold,
            use_reranker=args.use_reranker
        )

        # -----------------------------------------------------------------
        # n_retrieval = 3  # 1, 3 
        # score_threshold = 0.60
        # percentile = 50  # 25, 50, 75, any other will not call this filter
        # -----------------------------------------------------------------
        # if args.percentile in [25, 50, 75]:
        #     exp_name = f'test_exp2_{args.percentile}'
        # else:
        #     exp_name = 'test_exp1'
        # if not (proj_path / 'experiments' / exp_name).exists():
        #     (proj_path / 'experiments' / exp_name).mkdir(parents=True)

        # if not (proj_path / 'experiments' / 'bo_evals').exists():
        #     (proj_path / 'experiments' / 'bo_evals').mkdir(parents=True)
        
        # if args.task == 'zero_shot_hint':
        #     with (proj_path / 'data' / 'spider' / f'tables.json').open() as f:
        #         tables = json.load(f)

        #     with (proj_path / 'data' / 'description.json').open() as f:
        #         all_descriptions = json.load(f)

        #     spider_tables = process_all_tables(tables, descriptions=all_descriptions)
        #     vectorstore = get_vector_store(proj_path, percentile=args.percentile)

        #     run_bo_test_sql(
        #         proj_path, 
        #         spider_tables, 
        #         chain, 
        #         vectorstore, 
        #         exp_name, 
        #         n_retrieval=args.n_retrieval, 
        #         score_threshold=args.score_threshold
        #     )

        #     predictions = []
        #     for p in prediction_path.glob(f'{args.ds}_{args.type}_*.json'):
        #         with open(p) as f:
        #             pred = json.load(f)
        #             new_pred = []
        #             for x in pred:
        #                 x.pop('rationale')
        #                 new_pred.append(x)
        #             predictions.extend(new_pred)

        #     with open(prediction_path / f'final_{args.ds}_{args.type}.jsonl', 'w') as f:
        #         for p in predictions:
        #             f.write(json.dumps(p) + '\n')
        # elif args.task == 'bo_eval':
        #     df_train = pd.read_csv(proj_path / 'data' / 'split_in_domain' / 'spider_bo_desc_train.csv')
        #     df_test = pd.read_csv(proj_path / 'data' / 'split_in_domain' / 'test.csv')
        #     df_pred = pd.read_csv(proj_path / 'experiments' / 'bo_evals' / f'{exp_name}.csv')
        #     df_test = pd.merge(df_test, df_pred, on='sample_id')
        #     error_infos = get_error_infos(df_test)
        #     bo_eval(proj_path, df_test, error_infos)

# def get_vector_store(proj_path, percentile: Optional[int]=100):
#     df_train = pd.read_csv(proj_path / 'data' / 'split_in_domain' / f'spider_bo_desc_train.csv')
#     if percentile in [25, 50, 75]:
#         df_pm_stats = df_train.groupby(['db_id'])['pm_score_rank'].describe().loc[:, ['25%', '50%', '75%']]
#         pm_idx = df_train.apply(lambda x: filter_by_pm_score(x, df_pm_stats, percentile), axis=1)
#         df_train = df_train.loc[pm_idx].reset_index(drop=True)

#     documents = []
#     for i, row in df_train.iterrows():
#         doc = Document(
#             doc_id=row['sample_id'],
#             page_content=row['description'],
#             metadata={
#                 'sample_id': row['sample_id'],
#                 'db_id': row['db_id'],
#                 'cate_gold_c': row['cate_gold_c'],
#                 'cate_len_tbls': row['cate_len_tbls'],
#                 'virtual_table': row['virtual_table']
#             }
#         )
#         documents.append(doc)

#     embeddings_model = OpenAIEmbeddings()
#     vectorstore = FAISS.from_documents(
#         documents, 
#         embedding = embeddings_model,
#         distance_strategy = DistanceStrategy.EUCLIDEAN_DISTANCE
#     )
#     return vectorstore

# def run_bo_test_sql(
#         proj_path: Path,
#         spider_tables: dict[str, DatabaseModel], 
#         chain: RunnableSequence,
#         vectorstore: FAISS,
#         exp_name: str,
#         n_retrieval: int = 3,
#         score_threshold: float = 0.60
#     ):
#     final_outputs = []
#     df_test = pd.read_csv(proj_path / 'data' / 'split_in_domain' / 'test.csv')
#     df_test.reset_index(drop=True, inplace=True)

#     # restart from checkpoint
#     if list((proj_path / 'experiments' / exp_name).glob('*.json')) != []:
#         row_index = sorted([int(file.stem.split('_')[-1]) for file in sorted(list((proj_path / 'experiments' / exp_name).glob('*.json')))])
#         df_test = df_test.iloc[row_index[-1]:]
#         final_outputs = json.load((proj_path / 'experiments' / exp_name / f'{df_test.iloc[0]["sample_id"]}_{row_index[-1]}.json').open())

#     iterator = tqdm(df_test.iterrows(), total=len(df_test))
#     for i, row in iterator:
#         o = {'sample_id': row['sample_id']}

#         db_schema = get_schema_str(
#             schema=spider_tables[row['db_id']].db_schema, 
#             foreign_keys=spider_tables[row['db_id']].foreign_keys,
#             col_explanation=spider_tables[row['db_id']].col_explanation
#         )
        
#         # Experiment Complexity: low, mid, high
#         iterator.set_description(f"Processing {row['sample_id']}: Complexity - low, mid, high")
#         filter_key = 'cate_gold_c'
#         for filter_value in ['low', 'mid', 'high']:
#             retriever = vectorstore.as_retriever(
#                 search_kwargs={'k': n_retrieval, 'score_threshold': score_threshold, 'filter': {filter_key: filter_value, 'db_id': row['db_id']}}
#             )
#             docs = retriever.invoke(row['question'])
#             hint = 'Descriptions and Virtual Tables:\n'
#             hint += json.dumps({j: {'description': doc.page_content, 'virtual_table': doc.metadata['virtual_table']} for j, doc in enumerate(docs)}, indent=4)
#             hint += '\n'
#             input_data = {'schema': db_schema, 'input_query': row['question'], 'hint': hint}
#             output = chain.invoke(input=input_data)

#             o[f'c_{filter_value}'] = output.output
#             o[f'c_{filter_value}_hint'] = hint

#         # Experiment Complexity: 1, 2, 3+
#         iterator.set_description(f"Processing {row['sample_id']}: Complexity - 1, 2, 3+")
#         filter_key = 'cate_len_tbls'
#         for filter_value in ['1', '2', '3+']:
#             retriever = vectorstore.as_retriever(
#                 search_kwargs={'k': n_retrieval, 'score_threshold': score_threshold, 'filter': {filter_key: filter_value, 'db_id': row['db_id']}}
#             )
#             docs = retriever.invoke(row['question'])
#             hint = 'Descriptions and Virtual Tables:\n'
#             hint += json.dumps({j: {'description': doc.page_content, 'virtual_table': doc.metadata['virtual_table']} for j, doc in enumerate(docs)}, indent=4)
#             hint += '\n'
#             input_data = {'schema': db_schema, 'input_query': row['question'], 'hint': hint}
#             output = chain.invoke(input=input_data)

#             o[f't_{filter_value}'] = output.output
#             o[f't_{filter_value}_hint'] = hint
#         final_outputs.append(o)

#         if i % 100 == 0:
#             with (proj_path / 'experiments' / exp_name / f'{row["sample_id"]}_{i}.json').open('w') as f:
#                 json.dump(final_outputs, f, indent=4)
        
#     pd.DataFrame(final_outputs).to_csv(proj_path / 'experiments' / 'bo_evals' / f'{exp_name}.csv', index=False)

# def get_error_infos(df_test: pd.DataFrame) -> dict:

#     iterator = tqdm(df_test.iterrows(), total=len(df_test))
#     error_infos = {
#         'pred_exec': [],
#         'result': [],
#         'parsing_sql': [],
#         'error_samples': set(),
#     }

#     test_cols = ['c_low', 'c_mid', 'c_high', 't_1',  't_2',  't_3+']
#     for i, x in iterator:
#         has_error = False
#         schema = get_schema(str(proj_path / 'data' / 'spider' / 'database' / x['db_id'] / f'{x["db_id"]}.sqlite'))
#         schema = Schema(schema)
        
#         for test_col in test_cols:
#             try:
#                 sql = x[test_col]
#                 statement = sqlparse.parse(sql.strip())[0]
#                 aliases = extract_aliases(statement)
#                 selection = extract_selection(statement, aliases, schema)
#                 condition = extract_condition(statement, aliases, schema)
#                 aggregation = extract_aggregation(statement, aliases, schema)
#                 nested = extract_nested_setoperation(statement)
#                 others = extract_others(statement, aliases, schema)

#             except Exception as e:
#                 has_error = True
#                 error_infos['parsing_sql'].append((x['sample_id'], test_col, str(e)))
#                 error_infos['error_samples'].add(x['sample_id'])
#                 break
        
#         if has_error:
#             continue

#         iterator.set_description_str(f'error samples {len(error_infos["error_samples"])}')

#     print(f'Parsing SQL errors: {len(error_infos["parsing_sql"])}')
#     return error_infos

# def bo_eval(proj_path: Path, df_test: pd.DataFrame, error_infos: dict):
#     test_cols = ['c_low', 'c_mid', 'c_high', 't_1',  't_2',  't_3+']
#     eval_cols = ['score', 's_sel', 's_cond', 's_agg', 's_nest', 's_oth']

#     df = df_test.loc[~df_test['sample_id'].isin(error_infos['error_samples'])].reset_index(drop=True)
#     for test_col in test_cols:
#         df_exp = df.loc[:, ['sample_id', 'db_id', 'gold_sql', test_col]]
#         iterator = tqdm(df_exp.iterrows(), total=len(df_exp), desc=f'Processing {test_col}')
#         # init task eval results
#         task_results = {'sample_id': []}
#         for col in eval_cols:
#             task_results[f'{test_col}_{col}'] = []

#         for i, x in iterator:
#             task_results['sample_id'].append(x['sample_id'])
#             # parsing sql
#             schema = get_schema(str(proj_path / 'data' / 'spider' / 'database' / x['db_id'] / f'{x["db_id"]}.sqlite'))
#             schema = Schema(schema)
            
#             # partial & complexity eval
#             parsed_result = {}
#             for k in ['gold', 'pred']:
#                 sql = x[test_col] if k == 'pred' else x['gold_sql']
#                 statement = sqlparse.parse(sql.strip())[0]
#                 aliases = extract_aliases(statement)
#                 selection = extract_selection(statement, aliases, schema)
#                 condition = extract_condition(statement, aliases, schema)
#                 aggregation = extract_aggregation(statement, aliases, schema)
#                 nested = extract_nested_setoperation(statement)
#                 others = extract_others(statement, aliases, schema)

#                 parsed_result[k + '_selection'] = selection
#                 parsed_result[k + '_condition'] = condition
#                 parsed_result[k + '_aggregation'] = aggregation
#                 parsed_result[k + '_nested'] = nested
#                 parsed_result[k + '_others'] = {
#                     'distinct': others['distinct'], 
#                     'order by': others['order by'], 
#                     'limit': others['limit']
#                 }

#             eval_res = eval_all(parsed_result, k=6)
#             task_results[f'{test_col}_s_sel'].append(eval_res['score']['selection'])
#             task_results[f'{test_col}_s_cond'].append(eval_res['score']['condition'])
#             task_results[f'{test_col}_s_agg'].append(eval_res['score']['aggregation'])
#             task_results[f'{test_col}_s_nest'].append(eval_res['score']['nested'])
#             task_results[f'{test_col}_s_oth'].append(eval_res['score']['others'])
            
#             # execution eval
#             database = SqliteDatabase(
#                 str(proj_path / 'data' / 'spider' / 'database' / x['db_id'] / f'{x["db_id"]}.sqlite')
#             )
#             error_info = ''
#             try:
#                 pred_result = database.execute(x[test_col], rt_pandas=False)
#             except Exception as e:
#                 pred_result = []
#                 error_info = 'Predction Execution Error:' + str(e)
#                 score = 0

#             try:
#                 gold_result = database.execute(x['gold_sql'], rt_pandas=False)
#             except Exception as e:
#                 error_info = 'Gold Execution Error:' + str(e)

#             if 'Gold Execution Error' in error_info:
#                 continue
#             elif 'Predction Execution Error' in error_info:
#                 task_results[f'{test_col}_score'].append(score)
#                 continue
#             else:
#                 exists_orderby = check_if_exists_orderby(x['gold_sql'])
#                 score = int(result_eq(pred_result, gold_result, order_matters=exists_orderby))
#                 task_results[f'{test_col}_score'].append(score)

#         df_temp = pd.DataFrame(task_results)
#         df_test = pd.merge(df_test, df_temp, on='sample_id', how='left')
#         df_temp.to_csv(proj_path / 'experiments' / 'bo_evals' / f'{test_col}.csv', index=False)
#     df_test.to_csv(proj_path / 'experiments' / 'bo_evals' / f'all_{exp_name}.csv', index=False)