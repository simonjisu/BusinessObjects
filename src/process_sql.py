# Source: https://github.com/taoyds/spider/blob/master/process_sql.py
################################
# Assumptions:
#   1. sql is correct
#   2. only table name has alias
#   3. only one intersect/union/except
#
# val: number(float)/string(str)/sql(dict)
# col_unit: (agg_id, col_id, isDistinct(bool))
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, op_id, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
#   'having': condition
#   'limit': None/limit value
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

import json
import sqlite3
from nltk import word_tokenize

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', '<>', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')



class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """
    def __init__(self, schema):
        self._schema = schema
        self._idMap = self._map(self._schema)

    @property
    def schema(self) -> dict:
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema):
        idMap = {'*': "__all__"}
        id = 1
        for key, vals in schema.items():
            for val in vals:
                idMap[key.lower() + "." + val.lower()] = "__" + key.lower() + "." + val.lower() + "__"
                id += 1

        for key in schema:
            idMap[key.lower()] = "__" + key.lower() + "__"
            id += 1

        return idMap

    # def get_table_name(self, column: str, tables: list[str]=[]):
    #     if tables:
    #         schema = dict(list(filter(lambda x: x[0] in tables, self.schema.items())))
    #     else:
    #         schema = self.schema
    #     for table, cols in schema.items():
    #         if column.lower() in cols:
    #             return table
        
    #     return None
    
    def get_table_name(self, column: str, tables: list[str]=[]):
        if tables:
            subset_schema = {k: v for k,v in self.schema.items() if k in tables}
        else:
            subset_schema = self.schema

        for table, cols in subset_schema.items():
            if column.lower() in cols:
                return table
        return None

def get_schema(db):
    """
    Get database's schema, which is a dict with table name as key
    and list of column names as value
    :param db: database path
    :return: schema dict
    """

    schema = {}
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # fetch table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [str(table[0].lower()) for table in cursor.fetchall()]

    # fetch table info
    for table in tables:
        cursor.execute("PRAGMA table_info({})".format(table))
        schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]

    return schema


def get_schema_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)

    schema = {}
    for entry in data:
        table = str(entry['table'].lower())
        cols = [str(col['column_name'].lower()) for col in entry['col_data']]
        schema[table] = cols

    return schema


def tokenize(string):
    string = str(string)
    string = string.replace("\'", "\"")  # ensures all string values wrapped by "" problem??
    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
    assert len(quote_idxs) % 2 == 0, "Unexpected quote"

    # keep string value as token
    vals = {}
    for i in range(len(quote_idxs)-1, -1, -2):
        qidx1 = quote_idxs[i-1]
        qidx2 = quote_idxs[i]
        val = string[qidx1: qidx2+1]
        key = "__val_{}_{}__".format(qidx1, qidx2)
        string = string[:qidx1] + key + string[qidx2+1:]
        vals[key] = val

    toks = [word.lower() for word in word_tokenize(string)]
    # replace with string value token
    for i in range(len(toks)):
        if toks[i] in vals:
            toks[i] = vals[toks[i]]

    # find if there exists !=, >=, <=
    eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
    eq_idxs.reverse()
    prefix = ('!', '>', '<')
    for eq_idx in eq_idxs:
        pre_tok = toks[eq_idx-1]
        if pre_tok in prefix:
            toks = toks[:eq_idx-1] + [pre_tok + "="] + toks[eq_idx+1: ]

    # find if there exists <>
    neq_idxs = [idx for idx, tok in enumerate(toks) if tok == "<"]
    neq_idxs.reverse()
    for neq_idx in neq_idxs:
        if toks[neq_idx+1] == ">":
            toks = toks[:neq_idx] + ["<>"] + toks[neq_idx+2:]
            
    return toks


def scan_alias(toks):
    """Scan the index of 'as' and build the map for all alias"""
    as_idxs = [idx for idx, tok in enumerate(toks) if tok == 'as']
    alias = {}
    for idx in as_idxs:
        alias[toks[idx+1]] = toks[idx-1]
    return alias


def get_tables_with_alias(schema, toks):
    tables = scan_alias(toks)
    for key in schema:
        assert key not in tables, "Alias {} has the same name in table".format(key)
        tables[key] = key
    return tables


def parse_col(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, column id
    """
    tok = toks[start_idx]
    if tok == "*":
        return start_idx + 1, schema.idMap[tok]
    
    # check if the token is number
    if tok.replace('.', '').isnumeric():
        return start_idx+1, tok
    
    # check if the token is a unit operation
    if tok in UNIT_OPS:
        return start_idx+1, tok
    
    if '.' in tok:  # if token is a composite
        alias, col = tok.split('.')
        key = tables_with_alias[alias] + "." + col
        return start_idx+1, schema.idMap[key]

    assert default_tables is not None and len(default_tables) > 0, "Default tables should not be None or empty"

    for alias in default_tables:
        table = tables_with_alias[alias]
        if tok in schema.schema[table]:
            key = table + "." + tok
            return start_idx+1, schema.idMap[key]

    assert False, "Error col: {}".format(tok)


def parse_col_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, (agg_op id, col_id)
    """
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    isDistinct = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] in AGG_OPS:
        agg_id = AGG_OPS.index(toks[idx])
        idx += 1
        assert idx < len_ and toks[idx] == '('
        idx += 1
        if toks[idx] == "distinct":
            idx += 1
            isDistinct = True
        idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)
        assert idx < len_ and toks[idx] == ')'
        idx += 1
        return idx, (agg_id, col_id, isDistinct)

    if toks[idx] == "distinct":
        idx += 1
        isDistinct = True
    agg_id = AGG_OPS.index("none")
    idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, (agg_id, col_id, isDistinct)


def parse_val_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    col_unit1 = None
    col_unit2 = None
    unit_op = UNIT_OPS.index('none')

    idx, col_unit1 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
    if idx < len_ and toks[idx] in UNIT_OPS:
        unit_op = UNIT_OPS.index(toks[idx])
        idx += 1
        print(idx, toks)
        idx, col_unit2 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, (unit_op, col_unit1, col_unit2)


def parse_table_unit(toks, start_idx, tables_with_alias, schema):
    """
        :returns next idx, table id, table name
    """
    idx = start_idx
    len_ = len(toks)
    key = tables_with_alias[toks[idx]]

    if idx + 1 < len_ and toks[idx+1] == "as":
        idx += 3
    else:
        idx += 1

    return idx, schema.idMap[key], key


def parse_value(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] == 'select':
        idx, val = parse_sql(toks, idx, tables_with_alias, schema)
    elif "\"" in toks[idx]:  # token is a string value
        val = toks[idx]
        idx += 1
    else:
        try:
            val = float(toks[idx])
            idx += 1
        except:
            end_idx = idx
            while end_idx < len_ and toks[end_idx] != ',' and toks[end_idx] != ')'\
                and toks[end_idx] != 'and' and toks[end_idx] not in CLAUSE_KEYWORDS and toks[end_idx] not in JOIN_KEYWORDS:
                    end_idx += 1

            idx, val = parse_col_unit(toks[start_idx: end_idx], 0, tables_with_alias, schema, default_tables)
            idx = end_idx

    if isBlock:
        assert toks[idx] == ')'
        idx += 1

    return idx, val


def parse_condition(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    conds = []
    
    while idx < len_:
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        not_op = False
        if toks[idx] == 'not':
            not_op = True
            idx += 1

        assert idx < len_ and toks[idx] in WHERE_OPS, "Error condition: idx: {}, tok: {}".format(idx, toks[idx])
        op_id = WHERE_OPS.index(toks[idx])
        idx += 1
        val1 = val2 = None
        
        if op_id == WHERE_OPS.index('between'):  # between..and... special case: dual values
            idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
            assert toks[idx] == 'and'
            idx += 1
            idx, val2 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
        
        else:  # normal case: single value
            idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
            val2 = None

        conds.append((not_op, op_id, val_unit, val1, val2))

        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";") or toks[idx] in JOIN_KEYWORDS):
            break

        if idx < len_ and toks[idx] in COND_OPS:
            conds.append(toks[idx])
            idx += 1  # skip and/or

    return idx, conds


def parse_select(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    assert toks[idx] == 'select', "'select' not found"
    idx += 1
    isDistinct = False
    if idx < len_ and toks[idx] == 'distinct':
        idx += 1
        isDistinct = True
    val_units = []

    while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
        agg_id = AGG_OPS.index("none")
        if toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        val_units.append((agg_id, val_unit))
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','

    return idx, (isDistinct, val_units)


def parse_from(toks, start_idx, tables_with_alias, schema):
    """
    Assume in the from clause, all table units are combined with join
    """
    assert 'from' in toks[start_idx:], "'from' not found"

    len_ = len(toks)
    idx = toks.index('from', start_idx) + 1
    default_tables = []
    table_units = []
    conds = []

    while idx < len_:
        isBlock = False
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        if toks[idx] == 'select':
            idx, sql = parse_sql(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['sql'], sql))
        else:
            if idx < len_ and toks[idx] == 'join':
                idx += 1  # skip join
            idx, table_unit, table_name = parse_table_unit(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['table_unit'],table_unit))
            default_tables.append(table_name)
        if idx < len_ and toks[idx] == "on":
            idx += 1  # skip on
            idx, this_conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
            if len(conds) > 0:
                conds.append('and')
            conds.extend(this_conds)

        if isBlock:
            assert toks[idx] == ')'
            idx += 1
        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            break

    return idx, table_units, conds, default_tables


def parse_where(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'where':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_group_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    col_units = []

    if idx >= len_ or toks[idx] != 'group':
        return idx, col_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, col_unit = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
        col_units.append(col_unit)
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, col_units


def parse_order_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    val_units = []
    order_type = 'asc' # default type is 'asc'

    if idx >= len_ or toks[idx] != 'order':
        return idx, val_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        val_units.append(val_unit)
        if idx < len_ and toks[idx] in ORDER_OPS:
            order_type = toks[idx]
            idx += 1
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, (order_type, val_units)


def parse_having(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'having':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_limit(toks, start_idx):
    idx = start_idx
    len_ = len(toks)

    if idx < len_ and toks[idx] == 'limit':
        idx += 2
        return idx, int(toks[idx-1])

    return idx, None


def parse_sql(toks, start_idx, tables_with_alias, schema):
    isBlock = False # indicate whether this is a block of sql/sub-sql
    len_ = len(toks)
    idx = start_idx

    sql = {}
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    # parse from clause in order to get default tables
    from_end_idx, table_units, conds, default_tables = parse_from(toks, start_idx, tables_with_alias, schema)
    sql['from'] = {'table_units': table_units, 'conds': conds}
    # select clause
    _, select_col_units = parse_select(toks, idx, tables_with_alias, schema, default_tables)
    idx = from_end_idx
    sql['select'] = select_col_units
    # where clause
    idx, where_conds = parse_where(toks, idx, tables_with_alias, schema, default_tables)
    sql['where'] = where_conds
    # group by clause
    idx, group_col_units = parse_group_by(toks, idx, tables_with_alias, schema, default_tables)
    sql['groupBy'] = group_col_units
    # having clause
    idx, having_conds = parse_having(toks, idx, tables_with_alias, schema, default_tables)
    sql['having'] = having_conds
    # order by clause
    idx, order_col_units = parse_order_by(toks, idx, tables_with_alias, schema, default_tables)
    sql['orderBy'] = order_col_units
    # limit clause
    idx, limit_val = parse_limit(toks, idx)
    sql['limit'] = limit_val

    idx = skip_semicolon(toks, idx)
    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'
    idx = skip_semicolon(toks, idx)

    # intersect/union/except clause
    for op in SQL_OPS:  # initialize IUE
        sql[op] = None
    if idx < len_ and toks[idx] in SQL_OPS:
        sql_op = toks[idx]
        idx += 1
        idx, IUE_sql = parse_sql(toks, idx, tables_with_alias, schema)
        sql[sql_op] = IUE_sql
    return idx, sql


def load_data(fpath):
    with open(fpath) as f:
        data = json.load(f)
    return data


def get_sql(schema, query):
    toks = tokenize(query)
    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    _, sql = parse_sql(toks, 0, tables_with_alias, schema)

    return sql


def skip_semicolon(toks, start_idx):
    idx = start_idx
    while idx < len(toks) and toks[idx] == ";":
        idx += 1
    return idx




# ====
# from typing import Set, Tuple, Dict, Any, List
# from pydantic import BaseModel, Field
# import sqlglot
# from sqlglot import Expression, Select, Subquery, Table, Column, Alias, Union, Intersect, Except, Function, Identifier, Binary, Condition, expressions as exp

# def is_aggregate_function(func_name: str) -> bool:
#     AGG_OPS = ('MAX', 'MIN', 'COUNT', 'SUM', 'AVG', 'STDDEV', 'VARIANCE')
#     return func_name.upper() in AGG_OPS

# def extract_aliases(query: Select) -> Dict[str, Dict[str, str]]:
#     """
#     Extract table and column aliases from the given SELECT query.
#     Returns:
#         {
#           'table': {alias_or_table: table_name},
#           'column': {alias: actual_expression_string}
#         }
#     """
#     alias_mapping = {'table': {}, 'column': {}}

#     # Extract table aliases from FROM clause
#     from_clause = query.args.get("from")
#     if from_clause and isinstance(from_clause, exp.From):
#         _handle_from_clause(from_clause, alias_mapping['table'])

#     # Extract column aliases from SELECT clause
#     for select_exp in query.select(exclude_alias=True):
#         # If we have a Tuple (e.g., SELECT (col1 AS alias1, col2 AS alias2)),
#         # we need to dive into it to find Alias nodes
#         if isinstance(select_exp, exp.Tuple):
#             _extract_column_aliases_from_tuple(select_exp, alias_mapping['column'])
#         elif isinstance(select_exp, exp.Alias):
#             # Direct alias
#             alias = select_exp.alias
#             if alias:
#                 alias_mapping['column'][alias.lower()] = str(select_exp.this)
#         else:
#             # It's possible we have columns without AS aliases
#             # or other expressions. If you want to handle those, do it here.
#             pass

#     return alias_mapping

# def _handle_from_clause(from_clause: exp.From, table_alias_map: Dict[str, str]):
#     main_source = from_clause.this
#     _handle_table_or_subquery(main_source, table_alias_map)

#     joins = from_clause.args.get("joins")
#     if joins:
#         for join_expr in joins:
#             right_source = join_expr.args.get("expression")
#             if right_source:
#                 _handle_table_or_subquery(right_source, table_alias_map)

# def _handle_table_or_subquery(expr: Expression, table_alias_map: Dict[str, str]):
#     if isinstance(expr, exp.Table):
#         table_name = expr.name
#         alias = expr.alias
#         if alias:
#             table_alias_map[alias.lower()] = table_name.lower()
#         else:
#             table_alias_map[table_name.lower()] = table_name.lower()
#     elif isinstance(expr, exp.Subquery):
#         alias = expr.alias
#         if alias:
#             table_alias_map[alias.lower()] = f"({expr})"
#     elif isinstance(expr, exp.Select):
#         alias = expr.alias
#         if alias:
#             table_alias_map[alias.lower()] = f"({expr})"

# def _extract_column_aliases_from_tuple(tuple_expr: exp.Tuple, column_alias_map: Dict[str, str]):
#     # Tuple has multiple expressions inside: could be Alias, Column, or other things
#     for inner_expr in tuple_expr.expressions:
#         if isinstance(inner_expr, exp.Alias):
#             alias = inner_expr.alias
#             if alias:
#                 column_alias_map[alias.lower()] = str(inner_expr.this)
#         elif isinstance(inner_expr, exp.Tuple):
#             # Nested tuples (rare but possible)
#             _extract_column_aliases_from_tuple(inner_expr, column_alias_map)
#         # If needed, handle other node types here


# def extract_selection(query: Select, aliases: Dict[str, Dict[str, str]], schema: Schema) -> Tuple[Set[str], Set[Tuple[str, str]]]:
#     """
#     Extract columns and types (<s>, <c>, <a>) from SELECT clause.
#     unique_columns: set of columns encountered
#     selection_types: set of (selection_str, tag)
#     """
#     unique_columns = set()
#     selection_types = set()

#     for select_exp in query.select():
#         # select_exp could be Column, Function, Binary operation, etc.
#         columns_in_item = _extract_columns_from_expression(select_exp, aliases, schema)
#         unique_columns.update(columns_in_item)

#         tag = _classify_expression(select_exp)
#         selection_str = str(select_exp).lower().strip()
#         selection_types.add((selection_str, tag))

#     return unique_columns, selection_types

# def _get_full_column_name(col: Column, aliases: Dict[str, Dict[str, str]], schema: Schema) -> str:
#     """
#     Returns the fully qualified column name using the schema.
#     """
#     column_name = col.name.lower()
#     table_alias = col.table.lower() if col.table else None

#     # Determine the real table name either from aliases or the schema
#     if table_alias and table_alias in aliases['table']:
#         real_table_name = aliases['table'][table_alias]
#     else:
#         # If no alias or alias not found, attempt to resolve via schema
#         possible_tables = list(aliases['table'].values())
#         real_table_name = schema.get_table_name(column_name, possible_tables)

#     # Construct key for idMap lookup
#     if real_table_name:
#         key = f"{real_table_name.lower()}.{column_name}"
#     else:
#         # Fallback if no table found
#         key = column_name

#     # Return mapped identifier or a fallback
#     return schema.idMap.get(key, f"__{key}__")

# def _extract_columns_from_expression(expr: Expression, aliases: Dict[str, Dict[str, str]], schema: Schema) -> Set[str]:
#     """
#     Extracts column identifiers from an expression using the schema and aliases.
#     """
#     columns = set()
#     for col in expr.find_all(Column):
#         full_col_name = _get_full_column_name(col, aliases, schema)
#         columns.add(full_col_name)
#     return columns

# def _classify_expression(expr: Expression) -> str:
#     # classify into <s>, <c>, <a>
#     # <s> simple column
#     # <c> calculation or non-agg function
#     # <a> aggregate function
#     if isinstance(expr, Column):
#         return '<s>'
#     if isinstance(expr, exp.Function):
#         return '<a>' if is_aggregate_function(expr.name) else '<c>'
#     if isinstance(expr, (exp.Binary, exp.Condition)):
#         return '<c>'
#     return '<s>'

# def extract_condition(query: Select, aliases: Dict[str, Dict[str, str]], schema: Schema) -> Tuple[Set[str], Set[str]]:
#     """
#     Extract conditions from WHERE and HAVING.
#     Returns conditions and operator types
#     """
#     conditions = set()
#     operator_types = set()

#     for clause_name in ("where", "having"):
#         clause = query.args.get(clause_name)
#         if clause:
#             # clause is typically a Condition (AND/OR structure)
#             for cond in _extract_conditions(clause):
#                 # cond is a string representation of a condition
#                 conditions.add(cond)
#                 # extract ops
#                 ops = _extract_operation_types(cond)
#                 if ops:
#                     operator_types.add(ops)

#     return conditions, operator_types

# def _extract_conditions(expr: Expression) -> List[str]:
#     # Flatten a WHERE/HAVING condition (which may be a complex AND/OR tree) into a list of conditions
#     # Each leaf condition is often a Binary or Comparison.
#     conditions = []
#     if isinstance(expr, (exp.And, exp.Or)):
#         conditions.extend(_extract_conditions(expr.args['this']))
#         conditions.extend(_extract_conditions(expr.args['expression']))
#     else:
#         # This should be a leaf condition, convert to string
#         conditions.append(str(expr))
#     return conditions

# def _extract_operation_types(condition_str: str) -> str:
#     # Simple heuristic: look for operators like =, >, <, IN, LIKE, etc.
#     # In sqlglot, you could also directly inspect the expressions instead of using strings.
#     WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', '<>', 'in', 'like', 'is', 'exists')
#     ops_found = []
#     # Lowercase and split by non-alphabetic chars might help find ops
#     lower_cond = condition_str.lower()
#     for op in WHERE_OPS:
#         if op in lower_cond:
#             ops_found.append(op)
#     return ' '.join(ops_found)

# def extract_aggregation(query: Select, aliases: Dict[str, Dict[str, str]], schema: Schema) -> Tuple[Set[str], Set[Tuple[str, str]]]:
#     """
#     Extract columns in GROUP BY and classify them.
#     """
#     unique_columns = set()
#     aggregation_types = set()

#     group = query.args.get('group')
#     if group:
#         for g in group:
#             columns_in_item = _extract_columns_from_expression(g, aliases, schema)
#             unique_columns.update(columns_in_item)
#             tag = _classify_expression(g)
#             aggregation_types.add((str(g).lower().strip(), tag))

#     return unique_columns, aggregation_types

# def extract_nested_setoperation(expression: Expression) -> int:
#     """
#     Count set operations and nested queries.
#     """
#     count = 0
#     # If there is a UNION, INTERSECT, EXCEPT, they will appear as such nodes
#     if isinstance(expression, (Union, Intersect, Except)):
#         # count the two sides
#         count += 2  # or count += 1 depending on interpretation
#         # check nested
#         count += extract_nested_setoperation(expression.args['this'])
#         count += extract_nested_setoperation(expression.args['expression'])
#     elif isinstance(expression, Subquery):
#         # subquery inside
#         count += 1
#         if expression.args.get('this'):
#             count += extract_nested_setoperation(expression.args['this'])
#     elif isinstance(expression, Select):
#         # check for subqueries in FROM / WHERE / HAVING
#         for from_exp in expression.from_:
#             if isinstance(from_exp, Subquery):
#                 count += 1
#                 count += extract_nested_setoperation(from_exp.args['this'])

#         # WHERE/HAVING subqueries
#         for clause_name in ('where', 'having'):
#             clause = expression.args.get(clause_name)
#             if clause:
#                 for sq in clause.find_all(Subquery):
#                     count += 1
#                     count += extract_nested_setoperation(sq.args['this'])

#     return count

# def extract_others(query: Select, aliases: Dict[str, Dict[str, str]], schema: Schema) -> Dict[str, Any]:
#     """
#     Extract DISTINCT columns, ORDER BY columns, LIMIT usage.
#     """
#     others = {'distinct': set(), 'order by': set(), 'limit': False}

#     # DISTINCT
#     if query.distinct:
#         # If DISTINCT is used, we find columns in SELECT again
#         # Typically distinct applies to all selected columns
#         for col in query.select():
#             columns_in_item = _extract_columns_from_expression(col, aliases, schema)
#             others['distinct'].update(columns_in_item)

#     # ORDER BY
#     order = query.args.get('order')
#     if order:
#         # order is a list of Order expressions
#         for o in order:
#             # Extract columns from the order expression
#             cols = _extract_columns_from_expression(o, aliases, schema)
#             others['order by'].update(cols)

#     # LIMIT
#     if query.args.get('limit') is not None:
#         others['limit'] = True

#     return others

# # -------------------------------------------------------------------------------------------
# # Example usage
# if __name__ == '__main__':
#     sql = "SELECT t1.a, b AS bb, c FROM table1 t1 WHERE a > 10 AND b < 20"
#     query = sqlglot.parse_one(sql)
#     schema = Schema({'table1': ['a','b','c']})
#     aliases = extract_aliases(query)
#     print("Aliases:", aliases)
#     print("Selection:", extract_selection(query, aliases, schema))
#     print("Condition:", extract_condition(query, aliases, schema))
#     print("Aggregation:", extract_aggregation(query, aliases, schema))
#     print("Nested/SetOperations:", extract_nested_setoperation(query))
#     print("Others:", extract_others(query, aliases, schema))
