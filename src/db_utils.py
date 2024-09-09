from typing import Optional

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
        'tables': {}, 
        'columns': {}
    }
    
    for (i, col), (_, col_exp) in zip(database['column_names_original'], database['column_names']):
        if i == -1:
            continue
        
        table_name = database['table_names_original'][i]
        table_name_exp = database['table_names'][i]
        col_type = database['column_types'][i]
        col_name = f'{table_name}.{col}'
        
        if data_dict['schema'].get(table_name) is None:
            data_dict['schema'][table_name] = {}
        if col not in data_dict['schema'][table_name]:
            data_dict['schema'][table_name][col] = col_type

        if table_name not in data_dict['tables']:
            data_dict['tables'][table_name] = table_name_exp
        if col_name not in data_dict['columns']:
            data_dict['columns'][col_name] = col_exp
        if is_lack_semantic_meanings(col, col_exp):
            data_dict['col_explanation'][col_name] = col_exp

    for i, j in database['foreign_keys']:
        table_index1, col_name1 = database['column_names_original'][i]
        table_index2, col_name2 = database['column_names_original'][j]
        table_name1 = database['table_names_original'][table_index1]
        table_name2 = database['table_names_original'][table_index2]
        fk = f'{table_name1}.{col_name1} = {table_name2}.{col_name2}'
        data_dict['foreign_keys'].append(fk)

    for i in database['primary_keys']:
        table_index, col_name = database['column_names_original'][i]
        table_name = database['table_names_original'][table_index]
        pk = f'{table_name}.{col_name}'
        data_dict['primary_keys'].append(pk)
    return data_dict

def get_schema_str(
        schema: dict[str, dict[str, str]], 
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