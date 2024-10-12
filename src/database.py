import re
import sqlite3
import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional

class Database:
    def __init__(self, db_file: str|Path):
        self.db_file = str(db_file)
        self.dbtype = db_file.split('.')[-1]
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
    
    def _get_table_columns(self):
        query = 'SELECT name FROM sqlite_master WHERE type="table";'
        tables = self.execute(query)['name'].values.tolist()
        table_cols = {}
        for table in tables:
            query = f'PRAGMA table_info({table});'
            df = self.execute(query)
            table_cols[table] = df['name'].values.flatten().tolist()
        return table_cols

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