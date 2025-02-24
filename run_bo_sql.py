import json
import numpy as np
import pandas as pd
import argparse
import sqlglot
import pickle
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from itertools import product
import logging
import hashlib

from copy import deepcopy
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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from langchain_community.callbacks.manager import get_openai_callback
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

from src.prompts import Prompts
from src.pymodels import (
    SQLResponse, 
    GenTemplateResponse, 
    KeywordExtractionResponse, 
    DatabaseModel, 
    BirdSample, 
    SpiderSample,
    BODescription,
    DirectSQLResponse
)
from src.db_utils import get_schema_str, get_db_file
from src.data_preprocess import process_all_tables, load_raw_data, load_samples_spider_bird
from src.database import SqliteDatabase
from src.parsing_sql import (
    Schema, extract_all, extract_aliases, _format_expression, STRING_TYPE, NUMERIC_TYPE
)
from src.eval_utils import (
    get_complexity, 
    run_sqls_parallel,
    get_all_structural_score,
    get_all_semantic_score,
    # run_sqls,
    SKIP_DB_IDS
)

_ = load_dotenv(find_dotenv())

from typing import Iterator, Generator, Any
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
        tables: dict[str, DatabaseModel],
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

def remove_duplicate_bos(
        bos: dict[str, list[dict[str, str]]]
    ) -> dict[str, list[dict[str, str]]]:
    hash_map = defaultdict(set)
    for db_id, xs in bos.items():
        for x in xs:
            hash_map[x['vt']].add(x['sample_id'])

    ids = list(map(lambda x: list(x)[0], hash_map.values()))
    new_bos = {}
    for db_id, xs in bos.items():
        new_bos[db_id] = [x for x in xs if x['sample_id'] in ids]

    return new_bos

def get_vector_store(
        bos: dict[str, list[dict[str, str]]], 
        embeddings_model: Embeddings,
        is_question_query: bool=False) -> FAISS:
    documents = []
    for db_id, samples in bos.items():
        for x in samples:
            doc = Document(
                page_content=x['ba'] if not is_question_query else x['question'],
                metadata={
                    'sample_id': x['sample_id'],
                    'db_id': db_id,
                    'vt': x['vt'],
                    'complexity': x['gold_complexity']
                }
            )
            documents.append(doc)
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
        use_reranker: bool = False
    ) -> BaseRetriever:
    k_retrieval = k_retrieval if use_reranker else n_retrieval
    base_retriever = vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={
            'k': k_retrieval, 
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

def task_retrieve(
        samples: list[SpiderSample|BirdSample],
        bos: dict[str, list[dict[str, str]]],
        prediction_path: Path,
        embedding_model: str = 'openai',
        reranker_model: str = 'msmarco',
        prefix: str = 'x-',
        k_retrieval: int = 5,
        n_retrieval: int = 1,
        use_reranker: bool = False,
        is_question_query: bool = False,
        is_test: bool = False,
        custom_retriever_model_name: str='',
        custom_reranker_model_name: str=''
    ):
    if embedding_model == 'openai':
        embedder = OpenAIEmbeddings()
    elif embedding_model == 'custom':
        embedder = HuggingFaceEmbeddings(model_name=custom_retriever_model_name)
    else:
        raise KeyError('Invalid embedding model for `embedding_model`, either `openai` or `custom`')
    if reranker_model == 'msmarco':
        cross_encoder = HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
    elif reranker_model == 'custom':
        cross_encoder = HuggingFaceCrossEncoder(model_name=custom_reranker_model_name)
    else:
        raise KeyError('Invalid embedding model for `reranker_model`, either `msmarco` or `custom`')
    
    # filename pattern [prefix]-[db_id].json
    processed_db_ids = [p.stem.split('-')[-1] for p in prediction_path.glob(f'{prefix}*.json')]
    if processed_db_ids:
        bos = dict([(db_id, bo_list) for db_id, bo_list in bos.items() if db_id not in processed_db_ids])
        print(f'Skip some processed db_ids: {len(processed_db_ids)} {processed_db_ids[-5:]}...')

    samples_by_db_id = defaultdict(list)
    for sample in samples:
        samples_by_db_id[sample.db_id].append(sample)

    for db_id, samples in samples_by_db_id.items():
        vector_store = get_vector_store(
            bos=bos, 
            embeddings_model=embedder, 
            is_question_query=is_question_query)
        retriever = get_retriever(
            vectorstore=vector_store,
            db_id=db_id,
            cross_encoder=cross_encoder,
            n_retrieval=n_retrieval,
            k_retrieval=k_retrieval,
            use_reranker=use_reranker,
        )

        x_samples = list(filter(lambda x: x.db_id == db_id, samples))
        iterator = tqdm(x_samples, total=len(x_samples), desc=f"{db_id}")
        results = []
        for sample in iterator:
            question = sample.final.question
            docs = retriever.invoke(question)
            doc_ids = [doc.metadata['sample_id'] for doc in docs]
            res = {
                'db_id': db_id,
                'sample_id': sample.sample_id,
                'retrieved': doc_ids,
            }
            results.append(res)
        
        with open(prediction_path / f'{prefix}{db_id}.json', 'w') as f:
            json.dump(results, f)

    file_name = 'with_bos-test.json' if is_test else 'with_bos-dev.json'
    with open(prediction_path / file_name, 'w') as file:
        all_results = []
        for p in prediction_path.glob(f'{prefix}*.json'):
            with p.open() as f:
                temp = json.load(f)
            all_results.extend(temp)
        json.dump(all_results, file, indent=4)

    # remove temp_files
    for p in prediction_path.glob(f'{prefix}*.json'):
        p.unlink()

def task_gen_template(
        samples: list[SpiderSample|BirdSample],
        tables: dict[str, DatabaseModel],
        bos: dict[str, list[dict[str, str]]],
        retrieved: dict[str, int|list[int]],
        chain: RunnableSequence,
        prediction_path: Path,
        with_bos: bool = False,
        prefix: str = 'x-',
        is_test: bool = False
    ):
    n_batch = 32 if is_test else 2 # batch size
    n_batch = n_batch if with_bos else 32
    def create_hint(id2bo, doc_ids):
        docs = []
        # doc_ids could be empty
        if not doc_ids:
            return 'No Hints'
        docs = [{
            'descrption': id2bo[doc_id]['ba'], 
            'virtual_table': id2bo[doc_id]['vt']
        } for doc_id in doc_ids]
        
        hint = '\nDescriptions and Virtual Tables:\n'
        hint += json.dumps({j: doc for j, doc in enumerate(docs)}, indent=4)
        hint += '\n'
        return hint

    # filename pattern [prefix][db_id].json
    # retrieved:
    # test: [{db_id, sample_id, retrieved, }] or empty
    # dev: {db_id: [bo_id1, ...]}
    processed_db_ids = [p.stem.split('-')[-1] for p in prediction_path.glob(f'{prefix}*.json')]
    if processed_db_ids:
        samples = [sample for sample in samples if sample.db_id not in processed_db_ids]
        print(f'Skip some processed db_ids: {len(processed_db_ids)} {processed_db_ids[-5:]}')

    if with_bos and is_test:
        retrieved_by_db_id = defaultdict(dict)
        for sample in retrieved:
            db_id = sample['db_id']
            sample_id = sample['sample_id']
            retrieved_by_db_id[db_id][sample_id] = sample['retrieved']

    samples_by_db_id = defaultdict(list)
    for sample in samples:
        samples_by_db_id[sample.db_id].append(sample)

    for db_id, samples in samples_by_db_id.items():
        set_llm_cache(SQLiteCache(database_path=f"./cache/{prefix}{prediction_path.stem}_{db_id}.db"))
        schema_str = get_schema_str(
            schema=tables[db_id].db_schema, 
            foreign_keys=tables[db_id].foreign_keys,
            col_explanation=tables[db_id].col_explanation
        )
        if with_bos:
            # one-to-one mapping in BOs
            id2bo = {}
            retrieved_bos = bos.get(db_id, [])
            for x in retrieved_bos:
                id2bo[x['sample_id']] = x

        results = []
        batched_samples: list[BirdSample|SpiderSample] = list(batched(samples, n_batch))
        for batch in tqdm(batched_samples, total=len(batched_samples), desc=f"{db_id}"):
            batch_inputs = []
            batch_retrieved = []
            batch_sample_ids = []
            if with_bos:
                if is_test:
                    for sample in batch:
                        question = sample.final.question
                        doc_ids = retrieved_by_db_id[db_id][sample.sample_id]
                        hint = create_hint(id2bo, doc_ids)
                        input_data = {'schema': schema_str, 'input_query': question, 'hint': hint}
                        batch_inputs.append(input_data)
                        batch_retrieved.append(doc_ids)
                        batch_sample_ids.append(sample.sample_id)
                else:
                    for bo_id, sample in product(retrieved[db_id], batch):
                        question = sample.final.question
                        doc_ids = [bo_id]
                        hint = create_hint(id2bo, doc_ids)
                        input_data = {'schema': schema_str, 'input_query': question, 'hint': hint}
                        batch_inputs.append(input_data)
                        batch_retrieved.append(doc_ids)
                        batch_sample_ids.append(sample.sample_id)
            else:
                for sample in batch:
                    question = sample.final.question
                    input_data = {'schema': schema_str, 'input_query': question}
                    batch_inputs.append(input_data)
                    batch_retrieved.append([])
                    batch_sample_ids.append(sample.sample_id)
            
            with get_openai_callback() as cb:
                batch_outputs: list[GenTemplateResponse] = chain.batch(inputs=batch_inputs)

            for sample_id, output, doc_ids in zip(batch_sample_ids, batch_outputs, batch_retrieved):
                res = {}
                res['sample_id'] = sample_id
                res['rationale'] = output.rationale
                res['sql_template'] = output.sql
                if with_bos:
                    res['hint_used'] = output.hint_used
                res['token_usage'] = {'tokens': cb.total_tokens/len(batch), 'cost': cb.total_cost/len(batch)}
                res['doc_ids'] = doc_ids
                results.append(res)

        with open(prediction_path / f'{prefix}{db_id}.json', 'w') as f:
            json.dump(results, f, indent=4)
    
    file_p1 = 'no_bos' if not with_bos else 'with_bos'
    file_p2 = 'test' if is_test else 'dev'
    file_name = f'{file_p1}-{file_p2}.json'
    
    with open(prediction_path / file_name, 'w') as file:
        all_results = []
        for p in prediction_path.glob(f'{prefix}*.json'):
            with p.open() as f:
                temp = json.load(f)
            all_results.extend(temp)
        json.dump(all_results, file, indent=4)

    # remove temp_files
    for p in prediction_path.glob(f'{prefix}*.json'):
        p.unlink()

    for p in Path(f"./cache").glob(f"{prefix}{prediction_path.stem}_*.db"):
        p.unlink()

def task_keyword_extraction(
        input_samples: list[dict[str, str|int]],
        template_sample_id2doc_ids: dict[int, set[int]],  # mapping back by combining `sample_id` and `doc_ids`
        tables: dict[str, DatabaseModel],
        chain: RunnableSequence,
        prediction_path: Path,
        with_bos: bool = False,
        prefix: str = 'x-',
        is_test: bool = False
    ):
    n_batch = 32
    # filename pattern [prefix][db_id].json
    processed_db_ids = [p.stem.split('-')[-1] for p in prediction_path.glob(f'{prefix}*.json')]
    # group by db_ids
    samples_by_db_id = defaultdict(list)
    for sample in input_samples:
        # sample: {sample_id, db_id, question, evidence, sql_template}
        db_id = sample['db_id']
        if db_id not in processed_db_ids:
            x = {
                'sample_id': sample['sample_id'],
                'question': sample['question'],
                'sql_template': sample['sql_template'],
            }
            if sample.get('evidence'):
                x['evidence'] = sample['evidence']
            samples_by_db_id[db_id].append(x)

    for db_id, samples in samples_by_db_id.items():
        set_llm_cache(SQLiteCache(database_path=f"./cache/{prefix}{prediction_path.stem}_{db_id}.db"))
        results = []
        schema_str = get_schema_str(schema=tables[db_id].db_schema)

        batched_samples: list[dict] = list(batched(samples, n_batch))
        for batch in tqdm(batched_samples, total=len(batched_samples), desc=f"{db_id}"):
            batch_inputs = []
            batch_iidx2oidx: dict[int, int] = {}  # input_idx --> output_idx for batch_outputs
            
            for iidx, sample in enumerate(batch):
                # STRING_TYPE = '[PLACEHOLDER-TYPE:STRING]'.lower()
                # NUMERIC_TYPE = '[PLACEHOLDER-TYPE:NUMERIC]'.lower()
                # only need to inference on the queries that has placeholders
                if (STRING_TYPE.lower() in sample['sql_template'].lower()) or (NUMERIC_TYPE.lower() in sample['sql_template'].lower()):
                    input_data = {
                        'schema': schema_str,
                        'input_query': sample['question'],
                        'sql_template': sample['sql_template'] 
                    }
                    if sample.get('evidence'):
                        input_data['evidence'] = sample['evidence']
                    batch_inputs.append(input_data)
                    batch_iidx2oidx[iidx] = len(batch_inputs) - 1
            
            if batch_inputs:
                with get_openai_callback() as cb:
                    batch_outputs: list[KeywordExtractionResponse] = chain.batch(inputs=batch_inputs)
            else:
                batch_outputs = []

            for iidx, sample in enumerate(batch):
                sample_id = sample['sample_id']
                sql_template = sample['sql_template']
                res = {'sample_id': sample_id}
                
                oidx = batch_iidx2oidx.get(iidx)
                if oidx is not None:
                    res['keywords'] = batch_outputs[oidx].extraction
                else:
                    res['keywords'] = {}
                res['token_usage'] = {'tokens': cb.total_tokens/n_batch, 'cost': cb.total_cost/n_batch}

                key = (sql_template, sample_id)
                doc_ids = template_sample_id2doc_ids.get(key)
                if doc_ids:
                    for doc_id in doc_ids:
                        new_res = deepcopy(res)
                        new_res['doc_ids'] = [doc_id]
                        results.append(new_res)
                else:
                    res['doc_ids'] = []
                    results.append(res)

        with open(prediction_path / f'{prefix}{db_id}.json', 'w') as f:
            json.dump(results, f, indent=4)

    file_p1 = 'no_bos' if not with_bos else 'with_bos'
    file_p2 = 'test' if is_test else 'dev'
    file_name = f'{file_p1}-{file_p2}.json'

    with open(prediction_path / file_name, 'w') as file:
        all_results = []
        for p in prediction_path.glob(f'{prefix}*.json'):
            with p.open() as f:
                temp = json.load(f)
            all_results.extend(temp)
        json.dump(all_results, file, indent=4)

    # remove temp_files
    for p in prediction_path.glob(f'{prefix}*.json'):
        p.unlink()

    for p in Path(f"./cache").glob(f"{prefix}{prediction_path.stem}_*.db"):
        p.unlink()

def task_search_value(
        input_samples: list[dict[str, str|int]],
        keyword_sample_id2doc_ids: dict[tuple[str, int], set[int]],
        prediction_path: Path,
        with_bos: bool = False,
        prefix: str = 'x-',
        is_test: bool = False
    ):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    # filename pattern [prefix][db_id].json
    processed_db_ids = [p.stem.split('-')[-1] for p in prediction_path.glob(f'{prefix}*.json')]
    # group by db_ids
    samples_by_db_id = defaultdict(list)
    for sample in input_samples:
        # sample: {sample_id, db_id, db_file, search_keywords}
        db_id = sample['db_id']
        db_file = sample['db_file']
        if db_id not in processed_db_ids:
            samples_by_db_id[(db_id, str(db_file))].append({
                'sample_id': sample['sample_id'],
                'db_file': db_file,
                'search_keywords': sample['search_keywords']
            })

    for (db_id, db_file), samples in samples_by_db_id.items():
        results = []
        # initialize db
        db = SqliteDatabase(db_file=db_file)
        iterator = tqdm(samples, total=len(samples))
        for sample in iterator:
            iterator.set_description(f"{db_id} - {sample['sample_id']}")
            keywords: dict[str, list[str]] = sample['search_keywords']
            all_values: dict[str, dict[str, set[str]]] = {}
            
            if keywords:
                # only search for values if there are keywords
                for target_column, kws in keywords.items():    
                    for kw in kws:
                        iterator.set_postfix_str(f"Keyword: {kw}")
                        values = db.search_possible_similar_values(
                            keyword=kw,
                            target_column=target_column,
                            top_k=1,
                            embedding_type='huggingface'
                        )
                        for table_name, column_values in values.items():
                            for column_name, vs in column_values.items():
                                if not vs:
                                    continue

                                if table_name not in all_values:
                                    all_values[table_name] = {}
                                if column_name not in all_values[table_name]:
                                    all_values[table_name][column_name] = set()
                            
                                all_values[table_name][column_name].update(vs)
            
            # make set as list
            for table_name, column_values in all_values.items():
                for column_name, vs in column_values.items():
                    all_values[table_name][column_name] = list(vs)

            res = {
                'sample_id': sample['sample_id'],
                'values': all_values
            }
            key = (json.dumps(keywords), sample['sample_id'])
            doc_ids = keyword_sample_id2doc_ids[key]
            if doc_ids:
                for doc_id in doc_ids:
                    new_res = deepcopy(res)
                    new_res['doc_ids'] = [doc_id]
                    results.append(new_res)
            else:
                res['doc_ids'] = []
                results.append(res)

        with open(prediction_path / f'{prefix}{db_id}.json', 'w') as f:
            json.dump(results, f, indent=4)

    file_p1 = 'no_bos' if not with_bos else 'with_bos'
    file_p2 = 'test' if is_test else 'dev'
    file_name = f'{file_p1}-{file_p2}.json'
    with open(prediction_path / file_name, 'w') as file:
        all_results = []
        for p in prediction_path.glob(f'{prefix}*.json'):
            with p.open() as f:
                temp = json.load(f)
            all_results.extend(temp)
        json.dump(all_results, file, indent=4)

    # remove temp_files
    for p in prediction_path.glob(f'{prefix}*.json'):
        p.unlink()

def task_fill_in(
        input_samples: list[dict[str, str|int]],
        template_values_sample_id2doc_ids: dict[tuple[str, int], set[int]],
        tables: dict[str, DatabaseModel],
        chain: RunnableSequence,
        prediction_path: Path,
        with_bos: bool = False,
        prefix: str = 'x-',
        is_test: bool = False
    ):
    n_batch = 32
    # filename pattern [prefix][db_id].json
    processed_db_ids = [p.stem.split('-')[-1] for p in prediction_path.glob(f'{prefix}*.json')]
    # group by db_ids
    samples_by_db_id = defaultdict(list)
    for sample in input_samples:
        # sample: {sample_id, question, db_id, values, question, sql_template}
        # `doc_ids`: if not `with_bos` then `doc_ids` is None
        db_id = sample['db_id']
        if db_id not in processed_db_ids:
            samples_by_db_id[db_id].append({
                'sample_id': sample['sample_id'],
                'question': sample['question'],
                'values': sample['values'],
                'sql_template': sample['sql_template'],
            })

    for db_id, samples in samples_by_db_id.items():
        set_llm_cache(SQLiteCache(database_path=f"./cache/{prefix}{prediction_path.stem}_{db_id}.db"))
        results = []
        schema_str = get_schema_str(
            schema=tables[db_id].db_schema, 
            col_explanation=tables[db_id].col_explanation
        )

        batched_samples: list[dict] = list(batched(samples, n_batch))
        for batch in tqdm(batched_samples, total=len(batched_samples), desc=f"{db_id}"):
            batch_inputs = []
            batch_iidx2oidx: dict[int, int] = {}  # input_idx --> output_idx for batch_outputs
            
            for iidx, sample in enumerate(batch):
                # STRING_TYPE = '[PLACEHOLDER-TYPE:STRING]'.lower()
                # NUMERIC_TYPE = '[PLACEHOLDER-TYPE:NUMERIC]'.lower()
                # only need to inference on the queries that has placeholders
                if (STRING_TYPE.lower() in sample['sql_template'].lower()) or (NUMERIC_TYPE.lower() in sample['sql_template'].lower()):
                    input_data = {
                        'schema': schema_str,
                        'input_query': sample['question'],
                        'hint': json.dumps(sample['values'], indent=4),
                        'sql_template': sample['sql_template']
                    }
                    batch_inputs.append(input_data)
                    batch_iidx2oidx[iidx] = len(batch_inputs) - 1
            
            if batch_inputs:
                with get_openai_callback() as cb:
                    batch_outputs: list[SQLResponse] = chain.batch(inputs=batch_inputs)
            else:
                batch_outputs = []

            for iidx, sample in enumerate(batch):
                sample_id = sample['sample_id']
                sql_template = sample['sql_template']
                values = sample['values']
                res = {'sample_id': sample_id}

                oidx = batch_iidx2oidx.get(iidx)
                if oidx is not None:
                    res['rationale'] = batch_outputs[oidx].rationale
                    res['sql'] = batch_outputs[oidx].sql
                else:
                    res['rationale'] = ''
                    res['sql'] = sample['sql_template']
                res['token_usage'] = {'tokens': cb.total_tokens/n_batch, 'cost': cb.total_cost/n_batch}

                key = (sql_template, json.dumps(values), sample_id)
                doc_ids = template_values_sample_id2doc_ids.get(key)
                if doc_ids:
                    for doc_id in doc_ids:
                        new_res = deepcopy(res)
                        new_res['doc_ids'] = [doc_id]
                        results.append(new_res)
                else:
                    res['doc_ids'] = []
                    results.append(res)

        with open(prediction_path / f'{prefix}{db_id}.json', 'w') as f:
            json.dump(results, f, indent=4)

    file_p1 = 'no_bos' if not with_bos else 'with_bos'
    file_p2 = 'test' if is_test else 'dev'
    file_name = f'{file_p1}-{file_p2}.json'
    with open(prediction_path / file_name, 'w') as file:
        all_results = []
        for p in prediction_path.glob(f'{prefix}*.json'):
            with p.open() as f:
                temp = json.load(f)
            all_results.extend(temp)
        json.dump(all_results, file, indent=4)

    # remove temp_files
    for p in prediction_path.glob(f'{prefix}*.json'):
        p.unlink()

    for p in Path(f"./cache").glob(f"{prefix}{prediction_path.stem}_*.db"):
        p.unlink()

def evaluate_exec(
        eval_data: dict,
        eval_data2doc_ids: dict[str, set[int]],  # key(sample_id+pred_sql): {doc_id}
        eval_file: Path, 
        num_cpus: int = 4, 
        meta_time_out: float = 30.0,
        prefix: str = 'x-'
    ): 
    eval_path = eval_file.parent
    n_batch = 2000
    n_samples = len(eval_data['sample_ids'])
    batches = list(batched(range(n_samples), n_batch))
    for batch_i, idxes in enumerate(batches):
        logging.info(f"Processing execution - batch {batch_i+1}/{len(batches)}")
        if (eval_path / f'{prefix}temp_exec-{batch_i}.json').exists():
            continue
        batch_results = []
        batch_eval_data = {k: [v[i] for i in idxes] for k, v in eval_data.items()}
        batch_preds = [x for x in batch_eval_data['pred_queries']]
        batch_sample_ids = [x for x in batch_eval_data['sample_ids']]

        # if num_cpus == 1:
        #     batch_exec_result = run_sqls(batch_eval_data, meta_time_out=meta_time_out)
        # else:
        #     batch_exec_result = run_sqls_parallel(batch_eval_data, num_cpus=num_cpus, meta_time_out=meta_time_out)
        batch_exec_result = run_sqls_parallel(batch_eval_data, num_cpus=num_cpus, meta_time_out=meta_time_out)
        # assert len(batch_exec_result) == len(batch_sample_ids), f"Length of exec_result({len(batch_exec_result)}) and eval_data({len(batch_sample_ids)}) should be the same"
        logging.info(f"Batch {batch_i+1} - Done execution")
        for j, (sample_id, pred_sql) in enumerate(zip(batch_sample_ids, batch_preds)):
            key = hashlib.sha256(f'{sample_id}-{pred_sql}'.encode()).hexdigest()
            doc_ids = eval_data2doc_ids.get(key)
            result = batch_exec_result[j]
            if doc_ids:
                # if result is None:
                #     # skipped db_ids
                #     result = {'sample_id': sample_id, 'res': None, 'target_error': None}
                for doc_id in doc_ids:
                    new_res = deepcopy(result)
                    new_res['doc_ids'] = [doc_id]
                    batch_results.append(new_res)
            else:
                result['doc_ids'] = []
                batch_results.append(result)

        with open(eval_path / f'{prefix}temp_exec-{batch_i}.json', 'w') as f:
            json.dump(batch_results, f, indent=4)
        

    final_results = []
    for i in range(len(batches)):
        with open(eval_path / f'{prefix}temp_exec-{i}.json') as f:
            temp = json.load(f)
        final_results.extend(temp)

    with open(eval_file, 'w') as f:
        json.dump(final_results, f, indent=4)

    # remove all temp files
    for i in range(len(batches)):
        (eval_path / f'{prefix}temp_exec-{i}.json').unlink()

    # assert len(exec_result) == len(eval_data['sample_ids']), f"Length of exec_result({len(exec_result)}) and eval_data({len(eval_data['sample_ids'])}) should be the same"
    # final_results = []
    # for i in range(len(eval_data['sample_ids'])):
    #     sample_id = eval_data['sample_ids'][i]
    #     pred_sql = eval_data['pred_queries'][i]
    #     key = hashlib.sha256(f'{sample_id}-{pred_sql}'.encode()).hexdigest()
    #     doc_ids = eval_data2doc_ids.get(key)
    #     result = exec_result[i]  # result: sample_id, res, target_error
    #     if doc_ids: 
    #         for doc_id in doc_ids:
    #             new_res = deepcopy(result)
    #             new_res['doc_ids'] = [doc_id]
    #             final_results.append(new_res)
    #     else:
    #         result['doc_ids'] = []
    #         final_results.append(result)

    # with open(eval_file, 'w') as f:
    #     json.dump(final_results, f, indent=4)

def evaluate_merit(
        eval_data: dict,
        eval_data2doc_ids: dict[str, set[int]],  # key(sample_id+pred_sql): {doc_id}
        parsed_queries: dict[str, dict[str, Any]],  # {pred/target: {pred_sql/target_sql: parsed_query}}
        eval_file: Path, 
        prefix: str = 'x-'
    ):
    eval_path = eval_file.parent
    n_batch = 2000
    n_samples = len(eval_data['sample_ids'])
    batches = list(batched(range(n_samples), n_batch))
    for batch_i, idxes in enumerate(batches):
        logging.info(f"Processing merit - batch {batch_i+1}/{len(batches)}")
        if (eval_path / f'{prefix}temp_merit-{batch_i}.json').exists():
            continue
        batch_keys = []
        batch_target_parsed = []
        batch_pred_parsed = []
        batch_target_complexities = []
        batch_eval_data = {k: [v[i] for i in idxes] for k, v in eval_data.items()}
        for i in range(len(batch_eval_data['sample_ids'])):
            sample_id = batch_eval_data['sample_ids'][i]
            pred_sql = batch_eval_data['pred_queries'][i]
            target_sql = batch_eval_data['target_queries'][i]
            batch_pred_parsed.append(parsed_queries['pred'][pred_sql])
            batch_target_parsed.append(parsed_queries['target'][target_sql])
            batch_target_complexities.append(get_complexity(parsed_queries['target'][target_sql]))
            key = hashlib.sha256(f'{sample_id}-{pred_sql}'.encode()).hexdigest()
            batch_keys.append(key)

        structural_scores = get_all_structural_score(batch_pred_parsed, batch_target_parsed)
        semantic_scores = get_all_semantic_score(batch_pred_parsed, batch_target_parsed)
        epsilon = 1e-9
        structural_scores: np.ndarray = np.array(structural_scores)
        semantic_scores: np.ndarray = np.array(semantic_scores)
        f1_scores: np.ndarray = 2 * (structural_scores * semantic_scores) / (structural_scores + semantic_scores + epsilon)

        results = []
        for i in range(len(batch_eval_data['sample_ids'])):
            key = batch_keys[i]
            doc_ids = eval_data2doc_ids.get(key)
            if doc_ids:
                for doc_id in doc_ids:
                    results.append({
                        'sample_id': batch_eval_data['sample_ids'][i],
                        'structural_score': structural_scores[i],
                        'semantic_score': semantic_scores[i],
                        'f1_score': f1_scores[i],
                        'target_complexity': batch_target_complexities[i],
                        'doc_ids': [doc_id]
                    })
            else:
                results.append({
                    'sample_id': batch_eval_data['sample_ids'][i],
                    'structural_score': structural_scores[i],
                    'semantic_score': semantic_scores[i],
                    'f1_score': f1_scores[i],
                    'target_complexity': batch_target_complexities[i],
                    'doc_ids': []
                })

        with open(eval_path / f'{prefix}temp_merit-{batch_i}.json', 'w') as f:
            json.dump(results, f, indent=4)

    final_results = []
    for i in range(len(batches)):
        with open(eval_path / f'{prefix}temp_merit-{i}.json') as f:
            temp = json.load(f)
        final_results.extend(temp)

    with open(eval_file, 'w') as f:
        json.dump(final_results, f, indent=4)

    # remove all temp files
    for i in range(len(batches)):
        (eval_path / f'{prefix}temp_merit-{i}.json').unlink()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot SQL generation with OpenAI')
    parser.add_argument('--task', type=str, default='retrieve', 
        help='''
        `create_bo`, `retrieve`,
        `gen_template`, `keyword_extraction`, `search_value`, `fill_in`, 
        `evaluate`, `aggregate`'''
    )
    parser.add_argument('--ds', type=str, default='spider', help='Dataset to use for training. spider or bird') 
    parser.add_argument('--type', type=str, default='train', help='Type of data to use for .')
    parser.add_argument('--exp_name', type=str, default='experiments', help='Folder to store the experiments')
    parser.add_argument('--eval_target', type=str, default='direct', 
                        help='Task to evaluate: `direct`, `fill_in`')
    parser.add_argument('--with_bos', action='store_true', help='Whether to use BOs or not')
    parser.add_argument('--k_retrieval', type=int, default=5, help='Number of retrievals for reranker to consider')
    parser.add_argument('--n_retrieval', type=int, default=1, help='Number of retrievals to consider')
    parser.add_argument('--use_reranker', action='store_true', help='Whether to use reranker or not')
    parser.add_argument('--embedding_model', type=str, default='openai', 
                        help='`openai`, `custom`(`msmarco-MiniLM-L6-cos-v5-q_ba`)')
    parser.add_argument('--reranker_model', type=str, default='cross-encoder/ms-marco-MiniLM-L-6-v2',
                        help='Reranker model to use: `msmarco`(`cross-encoder/ms-marco-MiniLM-L-6-v2`), `custom`(ms-marco-MiniLM-L-6-v2-q_ba-rerank)')
    parser.add_argument('--is_question_query', action='store_true', help='Whether to use question query or not')
    parser.add_argument('--prefix', type=str, default='x-', help='Prefix for the prediction files')
    parser.add_argument('--num_cpus', type=int, default=3, help='Number of CPUs to evaluate execution results')
    parser.add_argument('--n_bos_sample', type=int, default=30, help='Number of BOs to sample')
    parser.add_argument('--n_bos_select', type=int, default=25, help='Number of BOs to select')
    args = parser.parse_args()

    proj_path = Path('.').resolve()
    assert proj_path.name == 'BusinessObjects', f'Expected project path to be BusinessObjects, but got {proj_path.name}'
    
    experiment_folder = proj_path / args.exp_name / args.ds
    
    if args.task in ['evaluate', 'aggregate']:
        eval_path = experiment_folder / 'evals'
        if not eval_path.exists():
            eval_path.mkdir(parents=True)
    else:
        prediction_path = experiment_folder / 'predictions' / args.task
        if not prediction_path.exists():
            prediction_path.mkdir(parents=True)

    # must load components
    description_file = 'bird_description.json' if args.ds == 'bird' else 'description.json'
    tables, *_ = load_raw_data(proj_path / 'data' / args.ds, load_test=False)
    with (proj_path / 'data' / description_file).open() as f:
        all_descriptions = json.load(f)
    tables = process_all_tables(tables, descriptions=all_descriptions)

    if args.task == 'create_bo':
        samples = load_samples_spider_bird(proj_path / 'data' / f'{args.ds}_{args.type}.json')
        # drop samples that are in SKIP_DB_IDS
        samples = [s for s in samples if s.db_id not in SKIP_DB_IDS]

        prompt = PromptTemplate(
            template=Prompts.bo_description,
            input_variables=['schema', 'virtual_table']
        )
        model_openai = ChatOpenAI(
            model='gpt-4o-mini',
            temperature=0.3,
            frequency_penalty=0.1,
        )
        model = model_openai.with_structured_output(BODescription, method='json_mode')
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

    elif args.task == 'retrieve':
        samples = load_samples_spider_bird(proj_path / 'data' / f'{args.ds}_{args.type}.json')
        # drop samples that are in SKIP_DB_IDS
        samples = [s for s in samples if s.db_id not in SKIP_DB_IDS]

        bo_type = 'train' if args.type == 'dev' else 'test'
        bo_path = experiment_folder / 'predictions' / 'create_bo' / f'final_{args.ds}_{bo_type}_bo.json'
        if not bo_path.exists():
            raise NameError(f'Run with the `task=create_bo, type={bo_type}` first. {bo_path}')
        with bo_path.open() as f:
            bos = json.load(f)
        bos = remove_duplicate_bos(bos)

        task_retrieve(
            samples=samples,
            bos=bos,
            prediction_path=prediction_path,
            embedding_model=args.embedding_model,
            reranker_model=args.reranker_model,
            prefix=args.prefix,
            k_retrieval=args.k_retrieval,
            n_retrieval=args.n_retrieval,
            use_reranker=args.use_reranker,
            is_question_query=args.is_question_query,
            is_test=False if args.type == 'dev' else True,
            custom_retriever_model_name=f'./models/msmarco-MiniLM-L6-cos-v5-{args.ds}-q_ba',
            custom_reranker_model_name=f'./models/ms-marco-MiniLM-L-6-v2-{args.ds}-q_ba-rerank'
        )

        # sample bos to evaluate
        if args.type == 'dev':
            retrieval_path: Path = experiment_folder / 'predictions' / 'retrieve' / f'with_bos-{args.type}.json'
            with retrieval_path.open() as f:
                retrieved = json.load(f)
            retrieved_by_db_ids = defaultdict(set)
            
            for r in retrieved:
                db_id = r['db_id']
                sample_id = r['sample_id']
                ret = r['retrieved']
                for s, bo in product([sample_id], ret):
                    retrieved_by_db_ids[db_id].update(ret)
            sampled_bos = defaultdict(list)
            for db_id, bos in retrieved_by_db_ids.items():
                if len(bos) > args.n_bos_sample:
                    n_sample = args.n_bos_sample
                else:
                    # sample half of the bos
                    n_sample = len(bos) // 2
                sampled = np.random.choice(list(bos), n_sample, replace=False).tolist()
                sampled_bos[db_id].extend(sampled)

            with (experiment_folder / 'predictions' / 'retrieve' / f'with_bos-sampled.json').open('w') as f:
                json.dump(sampled_bos, f, indent=4)

    elif args.task == 'gen_template':
        # final save file
        # file_p1 = 'no_bos' if not with_bos else 'with_bos'
        # file_p2 = 'test' if is_test else 'dev'
        # file_name = f'{file_p1}-{file_p2}.json'
        samples = load_samples_spider_bird(proj_path / 'data' / f'{args.ds}_{args.type}.json')
        # drop samples that are in SKIP_DB_IDS
        samples = [s for s in samples if s.db_id not in SKIP_DB_IDS]
        
        if args.with_bos:
            bo_type = 'train' if args.type == 'dev' else 'test'
            bo_path = experiment_folder / 'predictions' / 'create_bo' / f'final_{args.ds}_{bo_type}_bo.json'
            if not bo_path.exists():
                raise NameError(f'Run with the `task=create_bo, type={bo_type}` first. {bo_path}')
            with bo_path.open() as f:
                bos = json.load(f)
            bos = remove_duplicate_bos(bos)

            file_name = 'with_bos-sampled.json' if args.type == 'dev' else f'with_bos-{args.type}.json' 
            retrieval_path: Path = experiment_folder / 'predictions' / 'retrieve' / file_name
            assert retrieval_path.exists(), f'Run with the `task=retrieve, type={args.task}` first'
            with retrieval_path.open() as f:
                retrieved = json.load(f)
            # test: [{db_id, sample_id, retrieved, }]
            # dev: {db_id: [bo_id1, ...]}

        else:
            bos = {}
            retrieved = []

        if args.with_bos:
            prompt_template = Prompts.gen_template_with_bos
            input_variables = ['schema', 'input_query', 'hint']
            structured_output = GenTemplateResponse
        else:
            prompt_template = Prompts.gen_template_no_bos
            input_variables = ['schema', 'input_query']
            structured_output = SQLResponse
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=input_variables,
        )
        model_openai = ChatOpenAI(
            model='gpt-4o-mini',
            temperature=0.5,
            frequency_penalty=0.1,
        )
        model = model_openai.with_structured_output(structured_output, method='json_mode')
        chain = (prompt | model)
        
        task_gen_template(
            samples=samples,
            tables=tables,
            bos=bos,
            retrieved=retrieved,
            chain=chain,
            prediction_path=prediction_path,
            with_bos=args.with_bos,
            prefix=args.prefix,
            is_test=False if args.type == 'dev' else True
        )

    elif args.task == 'keyword_extraction':
        # final save file
        # file_p1 = 'no_bos' if not with_bos else 'with_bos'
        # file_p2 = 'test' if is_test else 'dev'
        # file_name = f'{file_p1}-{file_p2}.json'
        samples = load_samples_spider_bird(proj_path / 'data' / f'{args.ds}_{args.type}.json')
        # drop samples that are in SKIP_DB_IDS
        samples = [s for s in samples if s.db_id not in SKIP_DB_IDS]

        file_name = f'{"with_bos" if args.with_bos else "no_bos"}-{args.type}'
        template_path = experiment_folder / 'predictions' / 'gen_template' / f'{file_name}.json'
        assert template_path.exists(), f'Run with the `task=gen_template, type={args.type}` first'
        with template_path.open() as f:
            templates = json.load(f)
            # dev set could have duplicate sample_id that referes to the retrieval process
        
        # prepare inputs
        samples_by_id = {s.sample_id: s for s in samples}
        # same template will have one keyword extraction inference
        input_samples = []
        template_sample_id2doc_ids = defaultdict(set)  # template, sample_id: {doc_id}
        for t in templates:
            sample_id: int = t['sample_id']
            doc_ids: list = t['doc_ids']   # [] for no_bos
            sql_template: str = t['sql_template']
            key = (sql_template, sample_id)
            if key not in template_sample_id2doc_ids:
                sample = samples_by_id[sample_id]
                x = {
                    'sample_id': sample_id,
                    'question': sample.final.question,
                    'db_id': sample.db_id,
                    'sql_template': sql_template,
                }
                if args.ds == 'bird':
                    x['evidence'] = sample.evidence
                input_samples.append(x)
                template_sample_id2doc_ids[key].update(doc_ids)

            for doc_id in doc_ids:
                template_sample_id2doc_ids[key].add(doc_id)

        if not (args.with_bos and args.type == 'dev'):
            # dev(with_bos): sample_id + doc_ids (need to evaluate for doc_ids)
            # dev(no_bos): sample_id
            # test(both cases): sample_id (under test retrieval number = 1)
            assert len(template_sample_id2doc_ids) == len(templates), f"{len(template_sample_id2doc_ids)} != {len(templates)}"
        
        del samples_by_id, templates # free memory

        if args.ds == 'bird':
            prompt = PromptTemplate(
                template=Prompts.keyword_extraction_bird,
                input_variables=['schema', 'input_query', 'evidence', 'sql_template'],
            )
        else:
            prompt = PromptTemplate(
                template=Prompts.keyword_extraction_spider,
                input_variables=['schema', 'input_query', 'sql_template'],
            )
        model_openai = ChatOpenAI(
            model='gpt-4o-mini',
            temperature=0.5,
            frequency_penalty=0.1,
        )
        model = model_openai.with_structured_output(KeywordExtractionResponse, method='json_mode')
        chain = (prompt | model)

        task_keyword_extraction(
            input_samples=input_samples,
            template_sample_id2doc_ids=template_sample_id2doc_ids,
            tables=tables,
            chain=chain,
            prediction_path=prediction_path,
            with_bos=args.with_bos,
            prefix=args.prefix,
            is_test=False if args.type == 'dev' else True
        )

    elif args.task == 'search_value':
        # final save file
        # file_p1 = 'no_bos' if not with_bos else 'with_bos'
        # file_p2 = 'test' if is_test else 'dev'
        # file_name = f'{file_p1}-{file_p2}.json'

        def _format_column_value(string: str):
            if not isinstance(string, str):
                string = str(string)
            operations = [
                '>', '<', '=', '>=', '<=', '!=', '<>'
            ]
            found = False
            for op in operations:
                if op in string.lower():
                    found = True
                    break
            
            if found:
                # split the string by the operator
                string = string.split(op, 1)[-1].strip()
            
            # remove ' or " from the string
            string = string.replace("'", '').replace('"', '')
            return string
        
        samples = load_samples_spider_bird(proj_path / 'data' / f'{args.ds}_{args.type}.json')
        # drop samples that are in SKIP_DB_IDS
        samples = [s for s in samples if s.db_id not in SKIP_DB_IDS]

        file_name = f'{"with_bos" if args.with_bos else "no_bos"}-{args.type}'
        keyword_path = experiment_folder / 'predictions' / 'keyword_extraction' / f'{file_name}.json'
        assert keyword_path.exists(), f'Run with the `task=keyword_extraction, type={args.type}` first'
        with keyword_path.open() as f:
            keywords = json.load(f)

        # prepare inputs
        samples_by_id = {s.sample_id: s for s in samples}

        input_samples = []
        keyword_sample_id2doc_ids = defaultdict(set) # (all_kws, sample_id): {doc_id}
        for k in keywords:
            sample_id: int = k['sample_id']
            doc_ids: list = k['doc_ids']
            # process keywords
            output_kws: dict[str, list[str]] = k['keywords']
            # lower column and sort and lower keywords first
            output_kws = {k: sorted([x for x in vs if x]) for k, vs in output_kws.items()}
            
            all_kws: dict[str, list[str]] = {}
            for output_column_name, kws in output_kws.items():
                if kws:  # only store non-empty keywords
                    if ('.' in output_column_name): # rewrite the column_name
                        _, column_name = output_column_name.split('.')
                    else:
                        column_name = output_column_name
                    all_kws[column_name] = [_format_column_value(kw) for kw in kws if kw]

            key = (json.dumps(all_kws), sample_id)
            if key not in keyword_sample_id2doc_ids:
                sample = samples_by_id[sample_id]
                x = {
                    'sample_id': sample_id,
                    'db_id': sample.db_id,
                    'db_file': get_db_file(proj_path, args.ds, sample.db_id),
                    'search_keywords': all_kws
                }
                input_samples.append(x)
                keyword_sample_id2doc_ids[key].update(doc_ids)

            for doc_id in doc_ids:
                keyword_sample_id2doc_ids[key].add(doc_id)

        if not (args.with_bos and args.type == 'dev'):
            # dev(with_bos): sample_id + doc_ids (need to evaluate for doc_ids)
            # dev(no_bos): sample_id
            # test(both cases): sample_id (under test retrieval number = 1)
            assert len(keyword_sample_id2doc_ids) == len(keywords), f"{len(keyword_sample_id2doc_ids)} != {len(keywords)}"
        del samples_by_id, keywords # free memory

        task_search_value(
            input_samples=input_samples,
            keyword_sample_id2doc_ids=keyword_sample_id2doc_ids,
            prediction_path=prediction_path,
            with_bos=args.with_bos,
            prefix=args.prefix,
            is_test=False if args.type == 'dev' else True
        )
        
    elif args.task == 'fill_in':
        # final save file
        # file_p1 = 'no_bos' if not with_bos else 'with_bos'
        # file_p2 = 'test' if is_test else 'dev'
        # file_name = f'{file_p1}-{file_p2}.json'
        samples = load_samples_spider_bird(proj_path / 'data' / f'{args.ds}_{args.type}.json')
        # drop samples that are in SKIP_DB_IDS
        samples = [s for s in samples if s.db_id not in SKIP_DB_IDS]

        file_name = f'{"with_bos" if args.with_bos else "no_bos"}-{args.type}'
        values_path = experiment_folder / 'predictions' / 'search_value' / f'{file_name}.json'
        assert values_path.exists(), f'Run with the `task=search_value, type={args.type}` first'
        with values_path.open() as f:
            searched_values = json.load(f)

        template_path = experiment_folder / 'predictions' / 'gen_template' / f'{file_name}.json'
        assert template_path.exists(), f'Run with the `task=gen_template, type={args.type}` first'
        with template_path.open() as f:
            templates = json.load(f)

        # prepare inputs
        samples_by_id = {s.sample_id: s for s in samples}
        values_by_id = {v['sample_id']: v for v in searched_values}
        # same template will have one keyword extraction inference
        input_samples = []
        template_values_sample_id2doc_ids = defaultdict(set)  # template, sample_id: {doc_id}
        for t in templates:
            sample_id: int = t['sample_id']
            doc_ids: list = t['doc_ids']   # [] for no_bos
            sql_template: str = t['sql_template']
            values = values_by_id[sample_id]['values'] if values_by_id.get(sample_id) else {}
            # sort values
            if values:
                values = {
                    table_name: {column_name: sorted(vs) for column_name, vs in column_values.items()}
                        for table_name, column_values in values.items()}
            key = (sql_template, json.dumps(values), sample_id)
            if key not in template_values_sample_id2doc_ids:
                sample = samples_by_id[sample_id]
                x = {
                    'sample_id': sample_id,
                    'db_id': sample.db_id,
                    'question': sample.final.question,
                    'values': values,
                    'sql_template': sql_template,
                }
                input_samples.append(x)
                template_values_sample_id2doc_ids[key].update(doc_ids)

            for doc_id in doc_ids:
                template_values_sample_id2doc_ids[key].add(doc_id)

        print(f'Number of input samples: {len(template_values_sample_id2doc_ids)}')
        # keys = list(template_values_sample_id2doc_ids.keys())
        # selected = np.random.choice(list(range(len(keys))), 5, replace=False)
        # for i in selected:
        #     key = keys[i]
        #     print(f'# of doc_ids: {len(template_values_sample_id2doc_ids[key])}')

        if not (args.with_bos and args.type == 'dev'):
            # dev(with_bos): sample_id + doc_ids (need to evaluate for doc_ids)
            # dev(no_bos): sample_id
            # test(both cases): sample_id (under test retrieval number = 1)
            assert len(template_values_sample_id2doc_ids) == len(templates), f"{len(template_values_sample_id2doc_ids)} != {len(templates)}"

        del samples_by_id, values_by_id, templates, searched_values # free memory

        prompt = PromptTemplate(
            template=Prompts.fill_in,
            input_variables=['schema', 'input_query', 'hint', 'sql_template'],
        )
        model_openai = ChatOpenAI(
            model='gpt-4o-mini',
            temperature=0.3,
            frequency_penalty=0.1,
        )
        model = model_openai.with_structured_output(SQLResponse, method='json_mode')
        chain = (prompt | model)

        task_fill_in(
            input_samples=input_samples,
            template_values_sample_id2doc_ids=template_values_sample_id2doc_ids,
            tables=tables,
            chain=chain,
            prediction_path=prediction_path,
            with_bos=args.with_bos,
            prefix=args.prefix,
            is_test=False if args.type == 'dev' else True
        )
        
    elif args.task == 'evaluate':
        
        def parse_sql_to_output(sql: str, schema: Schema):
            try:
                ei = extract_all(sql, schema)
                assert len(ei['sel']) > 0, f'No selection found'
            except Exception as e:
                return None
            return ei
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
        samples = load_samples_spider_bird(proj_path / 'data' / f'{args.ds}_{args.type}.json')
        # drop samples that are in SKIP_DB_IDS
        samples = [s for s in samples if s.db_id not in SKIP_DB_IDS]

        file_name = f'{"with_bos" if args.with_bos else "no_bos"}-{args.type}'
        eval_target_path = experiment_folder / 'predictions' / args.eval_target / f'{file_name}.json'
        assert eval_target_path.exists(), f'Run with the `task={args.eval_target}, type={args.type}` first'
        with eval_target_path.open() as f:
            eval_targets = json.load(f)

        parsed_queries_path: Path = eval_path / f'parsed_queries-{file_name}.pkl'
        parsed_queries_exist = parsed_queries_path.exists()
        if parsed_queries_exist:
            with parsed_queries_path.open('rb') as f:
                parsed_queries = pickle.load(f)
        else:
            parsed_queries = defaultdict(dict)
            parsed_queries['unable'] = set()

        samples_by_id = {s.sample_id: s for s in samples}
        eval_data: dict[str, list] = defaultdict(list)
        eval_data2doc_ids = defaultdict(set)

        # from flatten to nested by doc_ids
        for pred in tqdm(eval_targets, total=len(eval_targets), desc='preparing eval data'):
            sample_id = pred['sample_id']
            pred_sql = pred['sql']
            doc_ids = pred['doc_ids']
            target_sample = samples_by_id[sample_id]
            target_sql = target_sample.final.sql
            db_id = target_sample.db_id
            key = hashlib.sha256(f'{sample_id}-{pred_sql}'.encode()).hexdigest()

            if not parsed_queries_exist:
                schema = Schema(tables[db_id].db_schema)
                if parsed_queries['target'].get(target_sql) is None:
                    target_output = parse_sql_to_output(target_sql, schema)
                    parsed_queries['target'][target_sql] = target_output
                else:
                    target_output = parsed_queries['target'][target_sql]
                if parsed_queries['pred'].get(pred_sql) is None:
                    pred_output = parse_sql_to_output(pred_sql, schema)
                    parsed_queries['pred'][pred_sql] = pred_output
                else:
                    pred_output = parsed_queries['pred'][pred_sql]
            
                if (not pred_output) or (not target_output):
                    if key not in parsed_queries['unable']:
                        parsed_queries['unable'].add(key)
                    continue
            else:
                if key in parsed_queries['unable']:
                    continue
            
            if key not in eval_data2doc_ids:
                eval_data['sample_ids'].append(sample_id)
                eval_data['target_queries'].append(target_sql)
                eval_data['db_paths'].append(get_db_file(proj_path, args.ds, db_id))
                eval_data['pred_queries'].append(pred_sql)
                eval_data2doc_ids[key].update(doc_ids)

            for doc_id in doc_ids:
                eval_data2doc_ids[key].add(doc_id)

        if not parsed_queries_exist:
            with parsed_queries_path.open('wb') as f:
                pickle.dump(parsed_queries, f)

        del samples_by_id, eval_targets # free memory

        # execution evaluation
        logging.info(f'Evaluation data loaded: {len(eval_data["sample_ids"])} samples')
        execution_result_path = eval_path / f'execution_result-{file_name}.json'
        if execution_result_path.exists():
            logging.info('Execution result exists, skip evaluation')
        else:
            evaluate_exec(
                eval_data,
                eval_data2doc_ids,
                execution_result_path,
                num_cpus=args.num_cpus,
                meta_time_out=30.0,
                prefix=args.prefix
            )

        # semantic and structural evaluation
        merit_result_path = eval_path / f'merit_result-{file_name}.json'
        if merit_result_path.exists():
            logging.info('Merits result exists, skip evaluation')
        else:
            evaluate_merit(
                eval_data, 
                eval_data2doc_ids,
                parsed_queries,
                merit_result_path,
                prefix=args.prefix
            )
    
    elif args.task == 'aggregate':
        samples = load_samples_spider_bird(proj_path / 'data' / f'{args.ds}_{args.type}.json')
        # drop samples that are in SKIP_DB_IDS
        samples = [s for s in samples if s.db_id not in SKIP_DB_IDS]
        
        file_name = f'{"with_bos" if args.with_bos else "no_bos"}-{args.type}'
        samples_by_id = {s.sample_id: s for s in samples}
        
        if args.eval_target == 'fill_in':  # pipeline method
            # gen_template
            with open(experiment_folder / 'predictions' / 'gen_template' / f'{file_name}.json', 'r') as f:
                sql_templates = json.load(f)

            # keyword_extraction
            with open(experiment_folder / 'predictions' / 'keyword_extraction' / f'{file_name}.json', 'r') as f:
                keyword_extraction = json.load(f)

            # search_value
            with open(experiment_folder / 'predictions' / 'search_value' / f'{file_name}.json', 'r') as f:
                search_value = json.load(f)
        else:
            sql_templates = []
            keyword_extraction = []
            search_value = []

        with open(experiment_folder / 'evals' / f'execution_result-{file_name}.json', 'r') as f:
            exec_results = json.load(f)

        with open(experiment_folder / 'evals' / f'merit_result-{file_name}.json', 'r') as f:
            merit_results = json.load(f)

        all_results = defaultdict(list)
        # column: sample_id, db_id, retrieved, exec_res, structural_score, semantic_score, f1_score, target_complexity
        # if fill_in: bo_used, keywords, values
        for i in range(len(exec_results)):
            sample_id = exec_results[i]['sample_id']
            doc_ids = exec_results[i]['doc_ids']
            exec_res = exec_results[i]['res']
            structural_score = merit_results[i]['structural_score']
            semantic_score = merit_results[i]['semantic_score']
            f1_score = merit_results[i]['f1_score']
            target_complexity = merit_results[i]['target_complexity']
            sample = samples_by_id[sample_id]
            db_id = sample.db_id

            all_results['sample_id'].append(sample_id)
            all_results['db_id'].append(db_id)
            all_results['retrieved'].append(','.join(map(str, doc_ids)))
            all_results['exec_res'].append(exec_res)
            all_results['structural_score'].append(structural_score)
            all_results['semantic_score'].append(semantic_score)
            all_results['f1_score'].append(f1_score)
            all_results['target_complexity'].append(target_complexity)
            

            if args.eval_target == 'fill_in':
                if args.with_bos:
                    bo_used = int(sql_templates[i]['hint_used'])
                    all_results['bo_used'].append(bo_used)
                keywords = json.dumps({column_name: kws for column_name, kws in keyword_extraction[i]['keywords'].items() if kws})
                values = json.dumps(search_value[i]['values'])
                all_results['keywords'].append(keywords)
                all_results['values'].append(values)

        df = pd.DataFrame(all_results)
        df.to_csv(eval_path / f'result-{file_name}.csv', index=False)
        print("Aggregated results saved to:", eval_path / f'result-{file_name}.csv')

    elif args.task == 'direct':
        samples = load_samples_spider_bird(proj_path / 'data' / f'{args.ds}_{args.type}.json')
        # drop samples that are in SKIP_DB_IDS
        samples = [s for s in samples if s.db_id not in SKIP_DB_IDS]
        file_name = f'{"with_bos" if args.with_bos else "no_bos"}-{args.type}'
        if args.with_bos:
            bo_type = 'train' if args.type == 'dev' else 'test'
            bo_path: Path = experiment_folder / 'predictions' / 'create_bo' / f'final_{args.ds}_{bo_type}_bo.json'
            if not bo_path.exists():
                raise NameError(f'Run with the `task=create_bo, type={bo_type}` first. {bo_path}')
            with bo_path.open() as f:
                bos = json.load(f)
            bos = remove_duplicate_bos(bos)

            file_name = 'with_bos-sampled.json' if args.type == 'dev' else f'with_bos-{args.type}.json' 
            retrieval_path: Path = experiment_folder / 'predictions' / 'retrieve' / file_name
            assert retrieval_path.exists(), f'Run with the `task=retrieve, type={args.task}` first'
            with retrieval_path.open() as f:
                retrieved = json.load(f)
            # test: [{db_id, sample_id, retrieved, }]
            # dev: {db_id: [bo_id1, ...]}

        else:
            bos = {}
            retrieved = []

        if args.with_bos:
            prompt_template = Prompts.zero_shot_hints_inference
            input_variables = ['schema', 'input_query', 'hint']
            structured_output = DirectSQLResponse
        else:
            prompt_template = Prompts.zero_shot_inference
            input_variables = ['schema', 'input_query']
            structured_output = SQLResponse
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=input_variables,
        )
        model_openai = ChatOpenAI(
            model='gpt-4o-mini',
            temperature=0.5,
            frequency_penalty=0.1,
        )
        model = model_openai.with_structured_output(structured_output, method='json_mode')
        chain = (prompt | model)
        
        task_gen_template(
            samples=samples,
            tables=tables,
            bos=bos,
            retrieved=retrieved,
            chain=chain,
            prediction_path=prediction_path,
            with_bos=args.with_bos,
            prefix=args.prefix,
            is_test=False if args.type == 'dev' else True
        )

        # change sql_template key to sql
        file_p1 = 'no_bos' if not args.with_bos else 'with_bos'
        file_p2 = args.type
        file_name = f'{file_p1}-{file_p2}.json'
        
        with open(prediction_path / file_name, 'r') as file:
            all_results = json.load(file)
            
        for r in all_results:
            r['sql'] = r.pop('sql_template')

        with open(prediction_path / file_name, 'w') as file:
            json.dump(all_results, file, indent=4)

    elif args.task == 'valid_bo':
        eval_path = experiment_folder / 'evals'
        no_bos_path = eval_path / f'result-no_bos-{args.type}.csv'
        with_bos_path = eval_path / f'result-with_bos-{args.type}.csv'
        assert no_bos_path.exists(), f'Run with the `task=evaluate, type={args.type}` first'
        assert with_bos_path.exists(), f'Run with the `task=evaluate, type={args.type}` first'

        df_no_bos = pd.read_csv(no_bos_path)
        df_no_bos = df_no_bos[~df_no_bos['db_id'].isin(SKIP_DB_IDS)]
        df_no_bos.reset_index(drop=True, inplace=True)
        df_no_bos.drop(columns=['retrieved'], inplace=True)
        df_with_bos = pd.read_csv(with_bos_path)
        df_cates = df_no_bos.groupby('db_id')['target_complexity'].apply(_get_categories).rename('category').apply(_format_interval)
        df_no_bos = pd.merge(df_no_bos, df_cates.reset_index('db_id', drop=True), left_index=True, right_index=True)
        df = pd.merge(
            left=df_with_bos,
            right=df_no_bos,
            how='inner',
            on=['db_id'],
            suffixes=('_bo', '')
        )

        group_column = ['db_id', 'retrieved']
        # count the exec result with/without BOs
        # > 0 means improved with BOs
        # = 0 means execution is same with/witout BOs
        # < 0 means worse with BOs
        execution_improvement = df.groupby(group_column)[['exec_res', 'exec_res_bo']].sum().diff(axis=1)['exec_res_bo'].rename('execution_improvement')

        # no_bos: tsed between (source=pred_sql, target=target_sql)
        # with_bos: tsed between (source=pred_sql, target=target_sql)
        # merit = tsed(pred_sql_bo, target_sql) - tsed(pred_sql, target_sql)
        # merit > 0 means similarity to the targer_sql improved with BOs
        # merit = 0 means similarity to the target_sql is same with/without BOs
        # merit < 0 means similarity to the target_sql getting worse with BOs
        merit_structural = df.groupby(group_column)[['structural_score', 'structural_score_bo']].mean().diff(axis=1)['structural_score_bo'].rename('merit_structural')
        merit_semantic = df.groupby(group_column)[['semantic_score', 'semantic_score_bo']].mean().diff(axis=1)['semantic_score_bo'].rename('merit_semantic')
        merit = df.groupby(group_column)[['f1_score', 'f1_score_bo']].mean().diff(axis=1)['f1_score_bo'].rename('merit')

        ranks = merit.reset_index().groupby(['db_id'])['merit'].rank(method='first', ascending=False).rename('rank').astype(np.int64)
        merit = pd.concat([merit.reset_index(), ranks], axis=1)
        merit_by_rank = merit.sort_values(by=['db_id', 'rank'], ascending=True)
        merit_by_rank.to_csv(experiment_folder / 'evals' / f'merits.csv', index=False)

        bo_path = experiment_folder / 'predictions' / 'create_bo' / f'final_{args.ds}_train_bo.json'
        with bo_path.open() as f:
            bos = json.load(f)

        test_bos = defaultdict(list)
        for x in merit_by_rank.loc[:, ['db_id', 'retrieved']].to_dict(orient='records'):
            if len(test_bos[x['db_id']]) >= args.n_bos_select:
                continue
            bo_id = x['retrieved']
            db_id = x['db_id']
            bo = list(filter(lambda x: x['sample_id'] == bo_id, bos[db_id]))[0]
            test_bos[db_id].append(bo)

        with (experiment_folder / 'predictions' / 'create_bo' / f'final_{args.ds}_test_bo.json').open('w') as f:
            json.dump(test_bos, f, indent=4)

        print('BOs selected for test set saved to:', experiment_folder / 'predictions' / 'create_bo' / f'final_{args.ds}_test_bo.json')
    # elif args.task == 'valid_bo_prepare_batch_run':
    #     samples = load_samples_spider_bird(proj_path / 'data' / f'{args.ds}_{args.type}.json')
    #     df = pd.read_csv(experiment_folder / 'evals' / 'zero_shot' / f'bird_dev.csv')
    #     df_score = df.loc[:, ['sample_id', 'db_id', 'exec_result']]
    #     df_error = df_score.loc[df_score['exec_result'] == 0, ['db_id', 'sample_id']]
    #     error_ids = df_error['sample_id'].tolist()
    #     samples = list(filter(lambda x: x.sample_id in error_ids, samples))
        
    #     with open(experiment_folder / f'partial_{args.ds}_db_ids.json') as f:
    #         partial_db_ids = json.load(f)
        
    #     bo_path = experiment_folder / 'predictions' / 'create_bo' / f'final_{args.ds}_train_bo.json'
    #     assert bo_path.exists(), 'Run with the `task=create_bo, type=train` first'
    #     with bo_path.open() as f:
    #         bos = json.load(f)

    #     bos = remove_duplicate_bos(bos)

    #     with open(experiment_folder / f'partial_{args.ds}_db_ids.json', 'w') as f:
    #         json.dump(partial_db_ids, f, indent=4)
        
    #     sampler = Sampler(bos)
        
    #     sampled_bos = {}
    #     for db_id_group in partial_db_ids:
    #         sampled_bos[str(db_id_group)] = defaultdict()
    #         for db_id in partial_db_ids[str(db_id_group)]:
    #             x_samples = list(filter(lambda x: x.db_id == db_id, samples))
    #             for idx_bos, train_bos in enumerate(sampler.sample(db_id, args.n_sample, args.n_stop, rt_idx=False)):
    #                 # print(f'{db_id}-{idx_bos} :', f'{len(train_bos)}', f'{len(list(product(train_bos, x_samples)))}')
    #                 sampled_bos[str(db_id_group)][f'{db_id}-{idx_bos}'] = {
    #                     'train_bos': train_bos,
    #                     'n_iter': len(list(product(train_bos, x_samples))), 
    #                     'total_bos_in_batch': len(train_bos),
    #                     'total_samples_in_batch': len(x_samples)
    #                 }

    #     with (experiment_folder / f'partial_{args.ds}_batch.json').open('w') as f:
    #         json.dump(sampled_bos, f, indent=4)

    # elif args.task == 'valid_bo':
    #     # use error sample to validate
    #     samples = load_samples_spider_bird(proj_path / 'data' / f'{args.ds}_{args.type}.json')
    #     df = pd.read_csv(experiment_folder / 'evals' / 'zero_shot' / f'{args.ds}_dev.csv')
    #     df_error = df.loc[df['exec_result'] == 0]
    #     error_ids = df_error['sample_id'].tolist()
    #     samples = list(filter(lambda x: x.sample_id in error_ids, samples))

    #     bo_path = experiment_folder / 'predictions' / 'create_bo' / f'final_{args.ds}_train_bo.json'
    #     assert bo_path.exists(), 'Run with the `task=create_bo, type=train` first'
    #     with bo_path.open() as f:
    #         bos = json.load(f)
    #     bos = remove_duplicate_bos(bos)

    #     batch_file_path = experiment_folder / f'partial_{args.ds}_batch.json'
    #     # assert batch_file_path.exists(), 'Run with the `task=create_bo, type=train` first'
    #     if (args.db_id_group >= 0) and batch_file_path.exists():
    #         with batch_file_path.open() as f:
    #             sampled_bos = json.load(f)[str(args.db_id_group)]  
    #         # dict["db_id-idx_bo", dict["train_bos", "n_iter", "total_bos_in_batch", "total_samples_in_batch"]]
    #     else:
    #         raise KeyError('Run with the `task=valid_bo_prepare_batch_run` first')
        
    #     # filter samples with db_ids gorup
    #     with open(experiment_folder / f'partial_{args.ds}_db_ids.json') as f:
    #         partial_db_ids = json.load(f)
    #     samples = list(filter(lambda x: x.db_id in partial_db_ids[str(args.db_id_group)], samples))
    #     print(f'{args.ds}-{args.type} samples loaded: {len(samples)}')
        
    #     prompt = PromptTemplate(
    #         template=Prompts.zero_shot_hints_inference,
    #         input_variables=['schema', 'input_query', 'hint'],
    #     )

    #     model = model_openai.with_structured_output(SQLResponse, method='json_mode')
    #     chain = (prompt | model)

    #     valid_bo(
    #         samples=samples, 
    #         tables=tables, 
    #         bos=sampled_bos, 
    #         chain=chain,
    #         prediction_path=prediction_path, 
    #         file_name=f'{args.ds}_{args.type}', 
    #         split_k=2,
    #     )

    # elif args.task == 'zero_shot_hint':
    #     bo_path = experiment_folder / 'predictions' / 'create_bo' / f'final_{args.ds}_train_bo.json'
    #     with bo_path.open() as f:
    #         all_bos = json.load(f)

    #     # test_scenarios
    #     with (experiment_folder / 'test_scenarios.json').open('r') as f:
    #             test_scenarios = json.load(f)
            
    #     sce = {0: "10", 1: "15", 2: "25", 3: "25"}[args.scenario]
    #     test_bo_ids = test_scenarios[sce]
    #     test_bos = defaultdict(list)
    #     for db_id, bos in all_bos.items():
    #         if db_id in test_bo_ids:
    #             bo_ids = test_bo_ids[db_id]
    #             test_bos[db_id].extend(list(filter(lambda x: x['sample_id'] in bo_ids, bos)))
        
    #     # (bo-query)
    #     if args.scenario in (0, 1, 2):
    #         is_question_query = False
    #         print('bo-query scenario')
    #     # (question-query)
    #     elif args.scenario in (3,):
    #         is_question_query = True
    #         print('question-query scenario')
    #     else:
    #         print(f'Invalid scenario: {args.scenario}')
    #         raise ValueError('Invalid scenario')
        
    #     # args.type == test
    #     samples = load_samples_spider_bird(proj_path / 'data' / f'{args.ds}_{args.type}.json')
    #     samples = [x for x in samples if x.db_id in test_bo_ids]
    #     print(f'{args.ds}-{args.type} samples loaded: {len(samples)}')
        
    #     prompt = PromptTemplate(
    #         template=Prompts.zero_shot_hints_inference,
    #         input_variables=['schema', 'input_query', 'hint'],
    #     )

    #     model = model_openai.with_structured_output(SQLResponse, method='json_mode')
    #     chain = (prompt | model)

    #     predict_sql_bo(
    #         samples=samples, 
    #         tables=tables, 
    #         test_bos=test_bos, 
    #         chain=chain,
    #         prediction_path=prediction_path, 
    #         file_name=f'{args.ds}_{args.type}_{args.scenario}', 
    #         split_k=3,
    #         k_retrieval=args.k_retrieval,
    #         n_retrieval=args.n_retrieval,
    #         score_threshold=args.score_threshold,
    #         use_reranker=args.use_reranker,
    #         is_question_query=is_question_query
    #     )