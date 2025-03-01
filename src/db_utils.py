from typing import Optional
from pathlib import Path

def get_db_file(proj_path: Path, ds: str, db_id: str):
    if ds == 'spider':
        db_file = str(proj_path / 'data' / ds / 'database' / db_id / f'{db_id}.sqlite')
    elif ds == 'bird':
        bird_train_db_ids = [p.stem for p in (proj_path / 'data' / ds / 'train' / 'train_databases').glob('*')]
        bird_dev_db_ids = [p.stem for p in (proj_path / 'data' / ds / 'dev' / 'dev_databases').glob('*')]
    
        if db_id in bird_train_db_ids:
            db_file = str(proj_path / 'data' / ds / 'train' / 'train_databases' / db_id / f'{db_id}.sqlite')
        elif db_id in bird_dev_db_ids:
            db_file = str(proj_path / 'data' / ds / 'dev' / 'dev_databases' / db_id  / f'{db_id}.sqlite')
        else:
            raise ValueError(f'Unknown db_id {db_id}')
    return db_file

def is_lack_semantic_meanings(col: str, col_exp: str) -> bool:
    col = col.lower()
    col_exp = col_exp.lower()
    cond1 = col != col_exp
    cond2 = col_exp.replace('_', '') != col
    cond3 = col_exp.replace(' ', '') != col
    return cond1 and cond2 and cond3

def get_data_dict(database: dict) -> dict[str, dict|list]:
    data_dict = {
        'schema': {},
        'foreign_keys': [],
        'primary_keys': [],
        'col_explanation': {}, 
    }
    
    for (i, col), (_, col_exp) in zip(database['column_names_original'], database['column_names']):
        if i == -1:
            continue
        
        table_name = database['table_names_original'][i].lower()
        table_name_exp = database['table_names'][i]
        col_type = database['column_types'][i]
        col_name = col.lower()
        
        if data_dict['schema'].get(table_name) is None:
            data_dict['schema'][table_name] = {}
        if col_name not in data_dict['schema'][table_name]:
            data_dict['schema'][table_name][col_name] = col_type

        if data_dict['col_explanation'].get(table_name) is None:
            data_dict['col_explanation'][table_name] = {}
        if data_dict['col_explanation'][table_name].get(col_name) is None:
            data_dict['col_explanation'][table_name][col_name] = col_exp

    for i, j in database['foreign_keys']:
        table_index1, col_name1 = database['column_names_original'][i]
        table_index2, col_name2 = database['column_names_original'][j]
        table_name1 = database['table_names_original'][table_index1]
        table_name2 = database['table_names_original'][table_index2]
        fk = f'{table_name1}.{col_name1} = {table_name2}.{col_name2}'.lower()
        data_dict['foreign_keys'].append(fk)

    for i in database['primary_keys']:
        if isinstance(i, int):
            table_index, col_name = database['column_names_original'][i]
            table_name = database['table_names_original'][table_index]
            pk = f'{table_name}.{col_name}'.lower()
            data_dict['primary_keys'].append(pk)
        else:
            for j in i:
                table_index, col_name = database['column_names_original'][j]
                table_name = database['table_names_original'][table_index]
                pk = f'{table_name}.{col_name}'.lower()
            data_dict['primary_keys'].append(pk)
    return data_dict

def get_schema_str(
        schema: dict[str, dict[str, str]], 
        foreign_keys: list[str]=None, 
        primary_keys: list[str]=None, 
        col_explanation: Optional[dict[str, dict[str, str]]]=None,   # table_name: {col_name: col_desc}
        col_fmt: str="'",
        skip_type: bool=False,
        remove_meta: bool=False
    ) -> str:
    """
    col_explanation: overrides the column explanation in the schema
    """
    def format_column(
            col_name: str, 
            col_type: str, 
            col_fmt: str="'", 
            col_desc: Optional[str]=None,
            skip_type: Optional[bool]=False
        ) -> str:
        if skip_type:
            s = f'{col_fmt}{col_name}{col_fmt}'
        else:
            s = f'{col_fmt}{col_name}{col_fmt}({col_type})'
        
        if col_desc is not None:
            return f'{s}: {col_desc}'
        
        return s
    
    def format_line_of_columns(
            cols: dict[str, str], 
            col_fmt: str, 
            col_descs: Optional[dict[str, str]]=None, 
            skip_type: Optional[bool]=False
        ) -> str:
        s = ''
        for col_name, col_type in cols.items():
            s += '  - ' + format_column(col_name, col_type, col_fmt, col_descs.get(col_name), skip_type)
            s += '\n'

        return s

    def format_list_of_columns(
            cols: dict[str, str], 
            col_fmt: str, 
            skip_type: Optional[bool]=False
        ) -> str:
        return ', '.join([
            format_column(col_name, col_type, col_fmt, None, skip_type) for col_name, col_type in cols.items()
        ])

    col_explanation = _lower_keys(col_explanation) if col_explanation is not None else None

    schema_str = ''
    schema_str += '[Table and Columns]\n' if not remove_meta else ''
    for table_name, cols in schema.items():
        if col_explanation is not None:
            schema_str += f'Table Name: {table_name}\n'
            col_descs = col_explanation.get(table_name)
            schema_str += format_line_of_columns(cols, col_fmt, col_descs, skip_type)
        else:
            schema_str += f'{table_name}: {format_list_of_columns(cols, col_fmt, skip_type)}\n'
    schema_str = schema_str.strip()
    if foreign_keys is not None:
        schema_str += '\n\n[Foreign Keys]\n' if not remove_meta else ''
        schema_str += '\n'.join(foreign_keys)
    if primary_keys is not None:
        schema_str += '\n\n[Primary Keys]\n' if not remove_meta else ''
        schema_str += '\n'.join(primary_keys)
    return schema_str

def get_schema_str_with_tables(
        schema: dict[str, dict[str, str]], 
        table_list: list[str],
        foreign_keys: list[str]=None, 
        primary_keys: list[str]=None, 
        col_explanation: Optional[dict[str, dict[str, str]]]=None,   # table_name: {col_name: col_desc}
        col_fmt: str="'"
    ) -> str:
    """
    col_explanation: overrides the column explanation in the schema
    """
    def format_column(col_name: str, col_type: str, col_fmt: str="'", col_desc: Optional[str]=None) -> str:
        if col_desc is not None:
            return f'{col_fmt}{col_name}{col_fmt}({col_type}): {col_desc}'
        
        return f'{col_fmt}{col_name}{col_fmt}({col_type})'
    
    def format_line_of_columns(cols: dict[str, str], col_fmt: str, col_descs: Optional[dict[str, str]]=None) -> str:
        s = ''
        for col_name, col_type in cols.items():
            s += '  - ' + format_column(col_name, col_type, col_fmt, col_descs.get(col_name))
            s += '\n'

        return s

    def format_list_of_columns(cols: dict[str, str], col_fmt: str) -> str:
        return ', '.join([
            format_column(col_name, col_type, col_fmt) for col_name, col_type in cols.items()
        ])

    schema_str = ''
    schema_str += '[Table and Columns]\n'
    for table_name, cols in schema.items():
        if table_name not in table_list:
            continue
        if col_explanation is not None:
            schema_str += f'Table Name: {table_name}\n'
            col_descs = col_explanation.get(table_name)
            schema_str += format_line_of_columns(cols, col_fmt, col_descs)
        else:
            schema_str += f'{table_name}: {format_list_of_columns(cols, col_fmt)}\n'
    schema_str = schema_str.strip()
    if foreign_keys is not None:
        schema_str += '\n\n[Foreign Keys]\n'
        schema_str += '\n'.join(foreign_keys)
    if primary_keys is not None:
        schema_str += '\n\n[Primary Keys]\n'
        schema_str += '\n'.join(primary_keys)
    return schema_str

def _lower_keys(d: dict[str, dict[str, str]]) -> dict:
    new_d = {}
    for k, v in d.items():
        new_d[k.lower()] = {k.lower(): v for k, v in v.items()}
    return new_d

def format_raw_sql(raw_sql: str) -> str:
    # replace '`' to '"'
    raw_sql = raw_sql.replace('`', '"')
    return raw_sql

# check LLM cache
# from pprint import pprint
# def get_content(x: dict):
#     return json.loads(x['kwargs']['message']['kwargs']['content'])
# task = 'direct'  
# # task = 'keyword_extraction'
# db_id = 'movie_3'
# prefix = 'x-'
# file_name = f'{task}_{db_id}'
# db = SqliteDatabase(f"./cache/{prefix}{file_name}.db")

# # df = db.execute("SELECT * FROM full_llm_cache WHERE ROWID = (SELECT MAX(ROWID) FROM full_llm_cache);")
# df = db.execute("SELECT ROWID, response FROM full_llm_cache;")
# df.shape
# file_name = f'{task}_{db_id}'
# db = SqliteDatabase(f"./cache/{prefix}{file_name}.db")
# db.start()
# c = db.con.cursor()
# c.execute('BEGIN TRANSACTION')

# # remove the last row record
# c.execute(f"""
# DELETE FROM full_llm_cache
# WHERE ROWID IN ({idxes});
# """)
# db.con.commit()
# db.close()