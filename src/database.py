__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import re
import json
import pickle
import duckdb
import sqlite3
import difflib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from threading import Lock
from langchain_chroma import Chroma
from langchain.schema.document import Document

import pickle
from datasketch import MinHash, MinHashLSH
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, Tuple, List, Any, Callable
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import multiprocessing as mp

EMBEDDING_FUNCTION = OpenAIEmbeddings(model="text-embedding-3-large")
HUGGING_FACE_EMBEDDING_FUNCTION = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def _get_semantic_similarity_with_huggingface(target_string: str, list_of_similar_words: list[str]) -> list[float]:
    """
    Computes semantic similarity between a target string and a list of similar words using OpenAI embeddings.

    Args:
        target_string (str): The target string to compare.
        list_of_similar_words (list[str]): The list of similar words to compare against.

    Returns:
        list[float]: A list of similarity scores.
    """
    target_string_embedding = HUGGING_FACE_EMBEDDING_FUNCTION.embed_query(target_string)
    all_embeddings = HUGGING_FACE_EMBEDDING_FUNCTION.embed_documents(list_of_similar_words)
    similarities = [np.dot(target_string_embedding, embedding) for embedding in all_embeddings]
    return similarities

def _get_semantic_similarity_with_openai(target_string: str, list_of_similar_words: list[str]) -> list[float]:
    """
    Computes semantic similarity between a target string and a list of similar words using OpenAI embeddings.

    Args:
        target_string (str): The target string to compare.
        list_of_similar_words (list[str]): The list of similar words to compare against.

    Returns:
        list[float]: A list of similarity scores.
    """
    target_string_embedding = EMBEDDING_FUNCTION.embed_query(target_string)
    all_embeddings = EMBEDDING_FUNCTION.embed_documents(list_of_similar_words)
    similarities = [np.dot(target_string_embedding, embedding) for embedding in all_embeddings]
    return similarities

def _create_minhash(signature_size: int, string: str, n_gram: int) -> MinHash:
    """
    Creates a MinHash object for a given string.

    Args:
        signature_size (int): The size of the MinHash signature.
        string (str): The input string to create the MinHash for.
        n_gram (int): The n-gram size for the MinHash.

    Returns:
        MinHash: The MinHash object for the input string.
    """
    m = MinHash(num_perm=signature_size)
    for d in [string[i:i + n_gram] for i in range(len(string) - n_gram + 1)]:
        m.update(d.encode('utf8'))
    return m

def _jaccard_similarity(m1: MinHash, m2: MinHash) -> float:
    """
    Computes the Jaccard similarity between two MinHash objects.

    Args:
        m1 (MinHash): The first MinHash object.
        m2 (MinHash): The second MinHash object.

    Returns:
        float: The Jaccard similarity between the two MinHash objects.
    """
    return m1.jaccard(m2)

def make_lsh(unique_values: Dict[str, Dict[str, List[str]]], signature_size: int, n_gram: int, threshold: float, verbose: bool = True) -> Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]:
    """
    Creates a MinHash LSH from unique values.

    Args:
        unique_values (Dict[str, Dict[str, List[str]]]): The dictionary of unique values.
        signature_size (int): The size of the MinHash signature.
        n_gram (int): The n-gram size for the MinHash.
        threshold (float): The threshold for the MinHash LSH.
        verbose (bool): Whether to display progress information.

    Returns:
        Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]: The MinHash LSH object and the dictionary of MinHashes.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=signature_size)
    minhashes: Dict[str, Tuple[MinHash, str, str, str]] = {}
    try:
        total_unique_values = sum(len(column_values) for table_values in unique_values.values() for column_values in table_values.values())
        logging.info(f"Total unique values: {total_unique_values}")
        
        progress_bar = tqdm(total=total_unique_values, desc="Creating LSH") if verbose else None
        
        for table_name, table_values in unique_values.items():
            for column_name, column_values in table_values.items():
                logging.info(f"Processing {table_name} - {column_name} - {len(column_values)}")
                
                for id, value in enumerate(column_values):
                    minhash = _create_minhash(signature_size, value, n_gram)
                    minhash_key = f"{table_name}_{column_name}_{id}"
                    minhashes[minhash_key] = (minhash, table_name, column_name, value)
                    lsh.insert(minhash_key, minhash)
                    
                    if verbose:
                        progress_bar.update(1)
        
        if verbose:
            progress_bar.close()
    except Exception as e:
        logging.error(f"Error creating LSH: {e}")
    
    return lsh, minhashes

def load_db_lsh(db_directory_path: str) -> Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]:
    """
    Loads the LSH and MinHashes from the preprocessed files in the specified directory.

    Args:
        db_directory_path (str): The path to the database directory.

    Returns:
        Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]: The LSH object and the dictionary of MinHashes.

    Raises:
        Exception: If there is an error loading the LSH or MinHashes.
    """
    db_id = Path(db_directory_path).name
    try:
        with open(Path(db_directory_path) / "preprocessed" / f"{db_id}_lsh.pkl", "rb") as file:
            lsh = pickle.load(file)
        with open(Path(db_directory_path) / "preprocessed" / f"{db_id}_minhashes.pkl", "rb") as file:
            minhashes = pickle.load(file)
        return lsh, minhashes
    except Exception as e:
        #logging.error(f"Error loading LSH for {db_id}: {e}")
        raise e

def query_lsh(lsh: MinHashLSH, minhashes: Dict[str, Tuple[MinHash, str, str, str]], keyword: str, 
              signature_size: int = 20, n_gram: int = 3, top_n: int = 10) -> Dict[str, Dict[str, List[str]]]:
    """
    Queries the LSH for similar values to the given keyword and returns the top results.

    Args:
        lsh (MinHashLSH): The LSH object.
        minhashes (Dict[str, Tuple[MinHash, str, str, str]]): The dictionary of MinHashes.
        keyword (str): The keyword to search for.
        signature_size (int, optional): The size of the MinHash signature.
        n_gram (int, optional): The n-gram size for the MinHash.
        top_n (int, optional): The number of top results to return.

    Returns:
        Dict[str, Dict[str, List[str]]]: A dictionary containing the top similar values.
    """
    query_minhash = _create_minhash(signature_size, keyword, n_gram)
    results = lsh.query(query_minhash)

    similarities = [(result, _jaccard_similarity(query_minhash, minhashes[result][0])) for result in results]
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

    similar_values_trimmed: Dict[str, Dict[str, List[str]]] = {}
    for result, similarity in similarities:
        table_name, column_name, value = minhashes[result][1:]
        if table_name not in similar_values_trimmed:
            similar_values_trimmed[table_name] = {}
        if column_name not in similar_values_trimmed[table_name]:
            similar_values_trimmed[table_name][column_name] = []
        similar_values_trimmed[table_name][column_name].append(value)

    return similar_values_trimmed

def query_vector_db(vector_db: Chroma, query: str, top_k: int) -> Dict[str, Dict[str, dict]]:
    """
    Queries the vector database for the most relevant documents based on the query.

    Args:
        vector_db (Chroma): The vector database to query.
        query (str): The query string to search for.
        top_k (int): The number of top results to return.

    Returns:
        Dict[str, Dict[str, dict]]: A dictionary containing table descriptions with their column details and scores.
    """
    table_description = {}
    
    try:
        relevant_docs_score = vector_db.similarity_search_with_score(query, k=top_k)
        # logging.info(f"Query executed successfully: {query}")
    except Exception as e:
        # logging.error(f"Error executing query: {query}, Error: {e}")
        raise e
    
    for doc, score in relevant_docs_score:
        metadata = doc.metadata
        table_name = metadata["table_name"].lower()
        column_name = metadata["column_name"].strip().lower()
        description = metadata["description"].strip()
        
        if table_name not in table_description:
            table_description[table_name] = {}
        
        if column_name not in table_description[table_name]:
            table_description[table_name][column_name] = {
                "column_name": column_name,
                "description": description,
                "score": score
            }
    
    # logging.info(f"Query results processed for query: {query}")
    return table_description

class Database:
    _instance = None
    _lock = Lock()

    def __init__(self, db_file: str|Path):
        self.db_file = str(db_file)
        self.db_id = self.db_file.split('/')[-1].split('.')[0]
        self.db_directory_path = Path(db_file).parent
        self.dbtype = self.db_file.split('.')[-1]
        self.lsh = None
        self.minhashes = None
        self.vector_db = None
        self.column_types = {
            'Null': ['NULL'],
            'Boolean': ['BOOLEAN'],
            'Integer': ['INTEGER', 'INT4', 'INT', 'SIGNED', 'BIGINT', 'INT8', 'LONG'],
            'Real': ['REAL', 'FLOAT', 'FLOAT4', 'DOUBLE', 'DECIMAL'],  # use str.contains('DECIMAL') to detect decimal
            'Text': ['VARCHAR', 'CHAR', 'BPCHAR', 'TEXT', 'STRING'],
            'Time': ['DATE', 'DATETIME', 'TIMESTAMP', 'INTERVAL', 'TIMESTAMP WITH TIME ZONE', 'TIMESTAMPZ'],
        }
        
    def start(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def execute(self, query: str, rt_pandas: bool = True):
        raise NotImplementedError
    
    def _get_table_columns(self) -> dict[str, list[str]]:
        raise NotImplementedError
    
    def _get_primary_keys(self) -> list[str]:
        raise NotImplementedError
    
    def create_index(self):
        raise NotImplementedError
    
    def set_lsh(self) -> str:
        """Sets the LSH and minhashes attributes by loading from pickle files."""
        with self._lock:
            if self.lsh is None:
                try:
                    with (self.db_directory_path / "preprocessed" / f"{self.db_id}_lsh.pkl").open("rb") as file:
                        self.lsh = pickle.load(file)
                    with (self.db_directory_path / "preprocessed" / f"{self.db_id}_minhashes.pkl").open("rb") as file:
                        self.minhashes = pickle.load(file)
                    return "success"
                except Exception as e:
                    self.lsh = "error"
                    self.minhashes = "error"
                    print(f"Error loading LSH for {self.db_id}: {e}")
                    return "error"
            elif self.lsh == "error":
                return "error"
            else:
                return "success"

    def set_vector_db(self, embedding_type: str='openai') -> str:
        """Sets the vector_db attribute by loading from the context vector database."""
        if self.vector_db is None:
            try:
                if embedding_type == 'openai':
                    embedding_func = EMBEDDING_FUNCTION
                else:
                    embedding_func = HUGGING_FACE_EMBEDDING_FUNCTION
                vector_db_path = self.db_directory_path / f"context_vector_db_{embedding_type}"
                self.vector_db = Chroma(persist_directory=str(vector_db_path), 
                                        embedding_function=embedding_func)
                return "success"
            except Exception as e:
                self.vector_db = "error"
                print(f"Error loading Vector DB for {self.db_id}: {e}")
                return "error"
        elif self.vector_db == "error":
            return "error"
        else:
            return "success"

    def query_lsh(self, keyword: str, signature_size: int = 100, n_gram: int = 3, top_n: int = 10) -> Dict[str, List[str]]:
        """
        Queries the LSH for similar values to the given keyword.

        Args:
            keyword (str): The keyword to search for.
            signature_size (int, optional): The size of the MinHash signature. Defaults to 20.
            n_gram (int, optional): The n-gram size for the MinHash. Defaults to 3.
            top_n (int, optional): The number of top results to return. Defaults to 10.

        Returns:
            Dict[str, List[str]]: The dictionary of similar values.
        """
        lsh_status = self.set_lsh()
        if lsh_status == "success":
            return query_lsh(self.lsh, self.minhashes, keyword, signature_size, n_gram, top_n)
        else:
            raise Exception(f"Error loading LSH for {self.db_id}")

    def query_vector_db(
            self, keyword: str, top_k: int, embedding_type: str='huggingface'
        ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Queries the vector database for similar values to the given keyword.

        Args:
            keyword (str): The keyword to search for.
            top_k (int): The number of top results to return.

        Returns:
            Dict[str, Dict[str, Dict[str, Any]]]: The dictionary of similar values.
        """
        vector_db_status = self.set_vector_db(embedding_type)
        if vector_db_status == "success":
            return query_vector_db(self.vector_db, keyword, top_k)
        else:
            raise Exception(f"Error loading Vector DB for {self.db_id}")

    def _get_unique_values(self) -> dict[str, dict[str, list[str]]]:
        table_schema = self._get_table_columns()
        primary_keys = self._get_primary_keys()
        unique_values: dict[str, dict[str, list[str]]] = {}
        iterator = tqdm(table_schema.items(), desc="Fetching unique values")
        for table_name, columns in iterator:
            iterator.set_description_str(f'[{table_name}]')
            table_values: dict[str, list[str]] = {}
            columns = [col[1] for col in self.execute(f"PRAGMA table_info('{table_name}')", rt_pandas=False) 
                if ("TEXT" in col[2] and col[1].lower() not in [c.lower() for c in primary_keys])]
            for column in columns:
                if any(keyword in column.lower() for keyword in ["_id", " id", "url", "email", "web", "time", "phone", "date", "address"]) or column.endswith("Id"):
                    continue
                iterator.set_postfix_str(f'Column: {column}')
                q = f"""
                    SELECT SUM(LENGTH(unique_values)), COUNT(unique_values)
                    FROM (
                        SELECT DISTINCT `{column}` AS unique_values
                        FROM `{table_name}`
                        WHERE `{column}` IS NOT NULL
                    ) AS subquery
                """
                result = self.execute(q, rt_pandas=False)
                sum_of_lengths, count_distinct = result[0]
                if sum_of_lengths is None or count_distinct == 0:
                    continue

                average_length = sum_of_lengths / count_distinct
                # print(f"Column: {column}, sum_of_lengths: {sum_of_lengths}, count_distinct: {count_distinct}, average_length: {average_length}")
                
                if ("name" in column.lower() and sum_of_lengths < 5000000) or (sum_of_lengths < 2000000 and average_length < 25):
                    # print(f"Fetching distinct values for {column}")
                    values = [str(value[0]) for value in self.execute(
                        f"SELECT DISTINCT `{column}` FROM `{table_name}` WHERE `{column}` IS NOT NULL", rt_pandas=False)]
                    
                    # print(f"Number of different values: {len(values)}")
                    table_values[column] = values
            unique_values[table_name] = table_values
        
        return unique_values
    
    def make_db_lsh(self, **kwargs: Any):
        db_id = self.db_id
        preprocessed_path = (Path(self.db_directory_path) / "preprocessed").resolve()
        # if preprocessed_path.exists():
        #     os.system(f"rm -r {preprocessed_path}")
        preprocessed_path.mkdir(exist_ok=True)
        if (preprocessed_path / f"{db_id}_unique_values.pkl").exists():
            logging.info(f"Unique values already exist for {db_id}")
            with open(preprocessed_path / f"{db_id}_unique_values.pkl", "rb") as file:
                unique_values = pickle.load(file)
        else:    
            unique_values = self._get_unique_values()
            with open(preprocessed_path / f"{db_id}_unique_values.pkl", "wb") as file:
                pickle.dump(unique_values, file)

        lsh, minhashes = make_lsh(unique_values, **kwargs)
        with open(preprocessed_path / f"{db_id}_lsh.pkl", "wb") as file:
            pickle.dump(lsh, file)
        with open(preprocessed_path / f"{db_id}_minhashes.pkl", "wb") as file:
            pickle.dump(minhashes, file)

    def make_db_context_vec_db(
            self, description: dict[str, dict[str]],
            embedding_type: str = "openai",
        ):
        db_id = self.db_id
        vector_db_path = (Path(self.db_directory_path) / f"context_vector_db_{embedding_type}").resolve()
        # if vector_db_path.exists():
        #     os.system(f"rm -r {vector_db_path}")
        vector_db_path.mkdir(exist_ok=True)
        docs = []
        for table_name, columns in description.items():
            for column_name, desc in columns.items():
                metadata = {
                    'db_id': db_id,
                    'table_name': table_name,
                    'column_name': column_name,
                    'description': desc
                }
                docs.append(Document(page_content=desc, metadata=metadata))
        
        if embedding_type == 'openai':
            embedding_func = EMBEDDING_FUNCTION
            logging.info(f"Using OpenAI embeddings for {db_id}")
        else:
            embedding_func = HUGGING_FACE_EMBEDDING_FUNCTION
            logging.info(f"Using HuggingFace embeddings for {db_id}")
        Chroma.from_documents(docs, embedding_func, persist_directory=str(vector_db_path))

    def is_column_exists(self, column: str):
        table_columns = self._get_table_columns()
        table_columns = {k.lower(): [v.lower() for v in vs] for k, vs in table_columns.items()}
        found = False

        for table_name, columns in table_columns.items():
            if column in columns:
                found = True
                break
        if found:
            return found, table_name
        return found, ''
    
    def _search_similar_values(
            self, keyword: str, values: list[str],
            sim_func: Callable,
            edit_distance_threshold:float=0.3,
            top_k_edit_distance: int=5, 
            embedding_similarity_threshold: float=0.6,
            top_k_embedding: int=1
        ):
        edit_distance_similar_values = [
            (value, 
                difflib.SequenceMatcher(None, value.lower(), keyword.lower()).ratio())
            for value in values
            if difflib.SequenceMatcher(None, value.lower(), keyword.lower()).ratio() >= edit_distance_threshold
        ]
        edit_distance_similar_values.sort(key=lambda x: x[1], reverse=True)
        edit_distance_similar_values = edit_distance_similar_values[:top_k_edit_distance]
        similarities = sim_func(keyword, [value for value, _ in edit_distance_similar_values])
        
        embedding_similar_values = [
            (keyword, edit_distance_similar_values[i][0], edit_distance_similar_values[i][1], similarities[i])
            for i in range(len(edit_distance_similar_values))
            if similarities[i] >= embedding_similarity_threshold
        ]
        embedding_similar_values.sort(key=lambda x: x[2], reverse=True)

        similar_values = embedding_similar_values[:top_k_embedding]
        return similar_values

    def search_similar_values_for_target_column(
            self, target_column: str, keyword: str, 
            edit_distance_threshold:float=0.3,
            top_k_edit_distance: int=5, 
            embedding_similarity_threshold: float=0.6,
            top_k_embedding: int=1,
        ):
        # find distinct column directly
        found, table_name = self.is_column_exists(target_column)
        
        if found:
            # exact match
            res: pd.DataFrame = self.execute(f'SELECT DISTINCT CAST("{target_column}" AS TEXT) FROM "{table_name}" WHERE "{target_column}" = "{keyword}";')
            if len(res) > 0:
                # found
                return [keyword], [(keyword, keyword, 1.0, 1.0)]
            
            # like match
            res: pd.DataFrame = self.execute(f'SELECT DISTINCT CAST("{target_column}" AS TEXT) FROM "{table_name}" WHERE "{target_column}" LIKE "%{keyword}%";')
            if len(res) > 0:
                values = res.values.flatten().astype(str).tolist()
                similar_values = self._search_similar_values(
                    keyword, values, 
                    sim_func=_get_semantic_similarity_with_huggingface,
                    edit_distance_threshold=edit_distance_threshold,
                    top_k_edit_distance=top_k_edit_distance,
                    embedding_similarity_threshold=embedding_similarity_threshold,
                    top_k_embedding=top_k_embedding
                )
                final_res = [value for _, value, *_ in similar_values]
                return final_res, similar_values            
            
            # search all distinct values
            # res: pd.DataFrame = self.execute(
            #     f'SELECT DISTINCT {target_column} FROM {table_name}')
            # values = res.values.flatten().astype(str).tolist()
            # if keyword in values:
            #     return [keyword], [(keyword, keyword, 1.0, 1.0)]

            # similar_values = self._search_similar_values(
            #     keyword, values, 
            #     sim_func=_get_semantic_similarity_with_huggingface,
            #     edit_distance_threshold=edit_distance_threshold,
            #     top_k_edit_distance=top_k_edit_distance,
            #     embedding_similarity_threshold=embedding_similarity_threshold,
            #     top_k_embedding=top_k_embedding
            # )
            # final_res = [value for _, value, *_ in similar_values]
        return [], []
            
    def search_possible_similar_values(
            self, 
            keyword: str, 
            target_column: str|None=None, 
            top_n: int=10,
            top_k: int=1,
            edit_distance_threshold:float=0.3,
            top_k_edit_distance: int=5,
            embedding_similarity_threshold: float=0.6,
            top_k_embedding: int=1,
            embedding_type: str='huggingface'
        ):
        all_possible_values: dict[str, dict[str, set]] = {}
        res_vector_db: dict[str, dict[str, dict[str, Any]]] = self.query_vector_db(
            keyword, top_k=top_k, embedding_type=embedding_type)
        candidates = [
            (table.lower(), column.lower()) 
            for table, columns in res_vector_db.items() 
            for column in columns.keys()
        ]
        
        # include target column search if provided
        if target_column is not None:
            found, table_name = self.is_column_exists(target_column)
            if found:
                candidates = [(table_name.lower(), target_column.lower())] + candidates
        
        search_lsh = True
        # search similar values for each column candidate
        for table_name, column_name in candidates:
            # use original if the column is a primary key or contains certain keywords (since they are usually unique)
            # primary_keys = self._get_primary_keys()
            # is_primary_keys = any(pk in target_column.lower() for pk in primary_keys)
            # stop_words = ["_id", " id", "url", "email", "web", "time", "phone", "date", "address"]
            # is_stopwords = any(stop_word in target_column.lower() for stop_word in stop_words) or target_column.endswith("Id")
            # TODO: temporary fix the performance issue by creating column index for all columns
            values, _ = self.search_similar_values_for_target_column(
                target_column=column_name,
                keyword=keyword,
                edit_distance_threshold=edit_distance_threshold,
                top_k_edit_distance=top_k_edit_distance,
                embedding_similarity_threshold=embedding_similarity_threshold,
                top_k_embedding=top_k_embedding
            )
            if values:
                search_lsh = False
                if table_name not in all_possible_values:
                    all_possible_values[table_name] = {}
                if column_name not in all_possible_values[table_name]:
                    all_possible_values[table_name][column_name] = set()
                all_possible_values[table_name][column_name].update(values)

        if search_lsh:
            # LSH search
            res_lsh: dict[str, dict[str, Any]] = self.query_lsh(keyword, top_n=top_n)
            # lower case the keyword
            res_lsh_lower = {
                table_name.lower(): {col.lower(): vals for col, vals in column_values.items()} \
                for table_name, column_values in res_lsh.items()
            }
            for table_name, columns in res_lsh_lower.items():
                for column_name, column_values in columns.items():
                    # filter similar values
                    similar_values = self._search_similar_values(
                        keyword, column_values, 
                        sim_func=_get_semantic_similarity_with_huggingface,
                        edit_distance_threshold=edit_distance_threshold,
                        top_k_edit_distance=top_k_edit_distance,
                        embedding_similarity_threshold=embedding_similarity_threshold,
                        top_k_embedding=top_k_embedding
                    )
                    values = [value for _, value, *_ in similar_values]
                    if values:
                        if table_name not in all_possible_values:
                            all_possible_values[table_name] = {}
                        if column_name not in all_possible_values[table_name]:
                            all_possible_values[table_name][column_name] = set()
                        
                        all_possible_values[table_name][column_name].update(values)
        
        # change values into list
        for table_name, columns in all_possible_values.items():
            for column_name, values in columns.items():
                all_possible_values[table_name][column_name] = list(values)

        return all_possible_values

class SqliteDatabase(Database):
    def __init__(self, db_file: str|Path, foreign_keys: Optional[dict[str, str]|list[str]]=None):
        super().__init__(db_file)
        self.dbtype = 'sqlite'
        self.table_cols = self._get_table_columns()
        if isinstance(foreign_keys, list):
            # the format is list of ['table_name.col_name = table_name.col_name']
            assert all(['=' in fk for fk in foreign_keys]), 'if `foreign_keys` is a list, must be in the format of "table_name.col_name = table_name.col_name"'
            self.foreign_keys = [{'fkey': fk.split('=')[0].strip(), 'pkey': fk.split('=')[1].strip()} for fk in foreign_keys]
        else:
            # {'fkey': 'table_name.col_name', 'pkey': 'table_name.col_name'}
            self.foreign_keys = foreign_keys   
    
    def _get_table_columns(self) -> dict[str, list[str]]:
        query = 'SELECT name FROM sqlite_master WHERE type="table";'
        tables = self.execute(query)['name'].values.tolist()
        table_cols: dict[str, list[str]] = {}
        for table in tables:
            query = f'PRAGMA table_info(`{table}`);'
            df = self.execute(query)
            table_cols[table] = df['name'].values.flatten().tolist()
        return table_cols
    
    def _get_primary_keys(self) -> list[str]:
        primary_keys = []
        table_names = self._get_table_columns().keys()
        for table_name in table_names:
            columns = self.execute(f"PRAGMA table_info('{table_name}')" , rt_pandas=False)
            for column in columns:
                if column[5] > 0:  # Check if it's a primary key
                    column_name = column[1]
                    if column_name.lower() not in [c.lower() for c in primary_keys]:
                        primary_keys.append(column_name)
        return primary_keys

    def start(self):
        self.con = sqlite3.connect(self.db_file)

    def close(self):
        self.con.close()

    def execute(self, query: str, rt_pandas: bool = True):
        self.start()
        
        if rt_pandas:
            output = pd.read_sql_query(query, self.con)
        
        else:
            c = self.con.cursor()
            output = c.execute(query).fetchall()
            c.close()

        self.close()
        return output
    
    def create_index(self):
        self.start()
        c = self.con.cursor()
        for table_name, columns in self.table_cols.items():
            if table_name == 'sqlite_sequence':
                continue
            for column in columns:
                c.execute(f"""
                CREATE INDEX IF NOT EXISTS 'idx_{table_name}_{column}' ON '{table_name}'('{column}');
                """)
        self.con.commit()
        c.close()

class DuckDBDatabase(Database):
    def __init__(self, db_file: str|Path):
        super().__init__(db_file)
        self.dbtype = 'duckdb'

        self.table_cols = self._get_table_columns()
        self.foreign_keys = self._get_foreign_keys()

    def _get_table_columns(self):
        query = 'SHOW ALL TABLES;'
        df = self.execute(query)
        table_names = df['name'].values.flatten().tolist()
        column_names = df['column_names'].values.flatten().tolist()
        return dict(zip(table_names, column_names))

    def _get_foreign_keys(self):
        query = 'SELECT * FROM information_schema.referential_constraints;'
        df = self.execute(query)
        if df.empty:
            return []
        df_keys = df.loc[:, ['constraint_name', 'unique_constraint_name']]

        ftbls, fcols = list(zip(*df_keys['constraint_name'].apply(lambda x: x.rstrip('_fkey').split('_', 1)).values.tolist()))
        ptbls, pcols = list(zip(*df_keys['unique_constraint_name'].apply(lambda x: x.rstrip('_pkey').split('_', 1)).values.tolist()))

        foregin_keys = []
        for ft, fc, pt, pc in zip(ftbls, fcols, ptbls, pcols):  # parent key
            foregin_keys.append({'fkey': f'{ft}.{fc}', 'pkey': f'{pt}.{pc}'})

        return foregin_keys

    def get_table_summaries(self, 
                            categorical_threshold: Optional[float]=0.05,
                            skip_keys: Optional[list[str]]=[]
        ):
        assert 0.0 < categorical_threshold <= 1.0, 'categorical_threshold must be in [0, 1]'
        table_summary = {}
        for table_name in self.table_cols.keys():
            table_summary[table_name] = self._summarize_table(
                table_name, categorical_threshold, skip_keys)
        return table_summary

    def _summarize_table(
            self, 
            table_name: str, 
            categorical_threshold: Optional[float]=0.05, 
            skip_keys: Optional[list[str]]=[],
        ):
        query = f'SUMMARIZE {table_name};'
        df = self.execute(query)
        df['logical_type'] = df.apply(
            self._check_logical_type, 
            column_types=self.column_types, 
            categorical_threshold=categorical_threshold,
            skip_keys=skip_keys,
            axis=1
        )
        df = df.loc[:, ['column_name', 'column_type', 'logical_type', 'approx_unique', 
                        'count', 'null_percentage', 'min', 'max',  'avg', 'std', 'q25', 'q50', 'q75']]

        return df

    def _check_logical_type(self, x, 
                            column_types: dict[str, list[str]], 
                            categorical_threshold: Optional[float]=0.05, 
                            skip_keys: Optional[list[str]]=[]
        ):
        """must apply with axis=1 for the whole table"""
        def re_exists(pattern, s):
            return bool(re.search(pattern, s))

        for logical_type, physical_types in column_types.items():
            if re_exists(r'DECIMAL', x['column_type']):
                return 'Real'
            if x['column_type'] in physical_types:
                cond1 = any([not re_exists(r, x['column_name']) for r in skip_keys])
                cond2 = logical_type in ['Integer', 'Text']
                cond3 = self._is_categorical(
                    x['approx_unique'], 
                    int(x['count'] * (1-x['null_percentage'])), 
                    threshold=categorical_threshold)
                if cond1 and cond2 and cond3:
                    return 'Categorical'
                return logical_type
        return 'Null'
    
    def _is_categorical(self, n_unique, n_total, threshold=0.05):
        if (n_unique / n_total) < threshold:
            return True
        return False
    
    def start(self):
        self.con = duckdb.connect(self.db_file)

    def close(self):
        self.con.close()

    def execute(self, query: str, rt_pandas: bool = True):
        self.start()
        if rt_pandas:
            output = self.con.execute(query).df()
        else:
            output = self.con.execute(query)
        self.close()

        return output
    
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', type=str, default='spider', help='Dataset to use for training. spider or bird') 
    parser.add_argument('--signature_size', type=int, default=100, help="Size of the MinHash signature")
    parser.add_argument('--n_gram', type=int, default=3, help="N-gram size for the MinHash")
    parser.add_argument('--threshold', type=float, default=0.01, help="Threshold for the MinHash LSH")
    parser.add_argument('--num_workers', type=int, default=1, help="Number of workers to use")
    parser.add_argument('--task', type=str, default='make_index', 
                        help='Task to perform. make_index or lsh_context')
    parser.add_argument('--embedding', type=str, default='openai', 
                        help='Embedding function to use. openai or huggingface')
    parser.add_argument('--only_context', action='store_true', help='Only create context vectors')
    args = parser.parse_args()

    proj_path = Path('.').resolve()
    assert proj_path.name == 'BusinessObjects', f'Expected project path to be BusinessObjects, but got {proj_path.name}'
    description_file = 'bird_description.json' if args.ds == 'bird' else 'description.json'
    with (proj_path / 'data' / description_file).open() as f:
        all_descriptions = json.load(f)
    
    database_path = proj_path / 'data' / args.ds
    db_paths: dict[str, Path] = {}
    if args.ds == 'bird':
        for split in ['train', 'dev']:
            for p in (database_path / split / f'{split}_databases').glob('*/'):
                db_paths[p.stem] = p / f'{p.stem}.sqlite'
    else:
        for p in (database_path / 'database').glob('*'):
            db_paths[p.stem] = p / f'{p.stem}.sqlite'

    if args.task == 'make_index':
        # uv run python ./src/database.py --task make_index --ds bird 
        for db_id, db_file in tqdm(db_paths.items(), total=len(db_paths), desc="Creating index"):
            db = SqliteDatabase(db_file)
            db.create_index()
    elif args.task == 'lsh_context':
        """
        uv run python ./src/database.py --task lsh_context --ds bird \
            --signature_size 100 --n_gram 3 \
            --threshold 0.01 --num_workers 1 \
            --embedding huggingface --only_context
        """ 
        def generate_db_extra_info(
                args: argparse.Namespace, 
                db_file: str, 
                description: dict[str, dict[str, str]],
            ):
            db = SqliteDatabase(db_file)
            if args.only_context:
                logging.info(f"Creating context vectors {db.db_id}")
                db.make_db_context_vec_db(description, embedding_type=args.embedding)
            else:
                logging.info(f"Creating LSH {db.db_id}")
                db.make_db_lsh(signature_size=args.signature_size, n_gram=args.n_gram, threshold=args.threshold)
                logging.info(f"Creating context vectors {db.db_id}")
                db.make_db_context_vec_db(description, embedding_type=args.embedding)
            logging.info(f"Finished processing {db.db_id}")
        # skip already processed databases
        skip = set()
        for db_id in db_paths:
            cond1, cond2 = False, False
            db_folder = db_paths[db_id].parent
            for x in list(db_folder.glob('*/')):
                if x.stem in [f'context_vector_db_{args.embedding}']:
                    # check two folders should be in the directory
                    cond1 = len(list(x.glob('*'))) == 2
                if x.stem in ['preprocessed']:
                    # check three files should be in the directory
                    cond2 = len(list(x.glob('*'))) == 3
                
            if cond1 and cond2:
                skip.add(db_id)
            # else:
            #     # remove all files in the directory and re-do the process
            #     os.system(f"rm -r {db_folder / 'context_vector_db'}")
            #     os.system(f"rm -r {db_folder / 'preprocessed'}")

        current_completed = len(skip)
        total_completed = len(db_paths)

        if args.num_workers == 1:
            for db_id, db_file in db_paths.items():
                if db_id in skip:
                    logging.info(f"Skipping {db_id}")
                    continue
                generate_db_extra_info(
                    args, db_file, all_descriptions[db_id]
                )
                current_completed += 1
                logging.info(f"Progress: {current_completed}/{total_completed}")
        else:
            with mp.Pool(args.num_workers) as pool: 
                for db_id, db_file in db_paths.items():
                    if db_id in skip:
                        logging.info(f"Skipping {db_id}")
                        continue
                    pool.apply_async(
                        generate_db_extra_info, 
                        args=(
                            args, db_file, all_descriptions[db_id]
                        ), 
                        error_callback=lambda e: logging.error(e))
                pool.close()
                pool.join()
