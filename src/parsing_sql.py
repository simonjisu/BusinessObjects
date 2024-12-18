# import sqlparse
# from sqlparse.sql import (
#     Token, TokenList, IdentifierList, Identifier, Function, Statement, Parenthesis, Operation,
#     Where, Comparison
# )
# import sqlparse.tokens as tks
# from typing import Callable
# from pydantic import BaseModel, Field
from .process_sql import Schema
from collections import defaultdict
import sqlglot
from sqlglot import expressions as exp
from typing import Set, Tuple, Dict, Any, List
from src.process_sql import Schema

OPERATOR_MAP = {
    exp.EQ: "=",
    exp.GT: ">",
    exp.GTE: ">=",
    exp.LT: "<",
    exp.LTE: "<=",
    exp.NEQ: "<>",    # for not equal
    exp.Is: "is",     # if needed
    exp.Like: "like", # if needed
    # Add more if needed
}

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
    
    def get_table_name(self, column: str, tables: list[str]=[]):
        if tables:
            subset_schema = {k: v for k,v in self.schema.items() if k in tables}
        else:
            subset_schema = self.schema

        for table, cols in subset_schema.items():
            if column.lower() in cols:
                return table
        return None

def get_subqueries(node: exp.Expression) -> List[exp.Select]:
    """
    Recursively extract all SELECT subqueries from the given expression.
    
    Handles:
    - Single SELECT statements
    - Set operations (UNION, INTERSECT, EXCEPT) by extracting SELECTs from both sides
    - Nested subqueries in FROM, WHERE, HAVING, and SELECT clauses
    
    Returns a list of exp.Select nodes representing all individual SELECT queries.
    """
    if isinstance(node, (exp.Union, exp.Intersect, exp.Except)):
        # Set operation: recursively get subqueries from both sides
        left = get_subqueries(node.args.get('this'))
        right = get_subqueries(node.args.get('expression'))
        return left + right

    if isinstance(node, exp.Subquery):
        # Subquery: extract subqueries from its inner SELECT (or whatever it holds)
        inner = node.args.get("this")
        return get_subqueries(inner) if inner else []

    if isinstance(node, exp.Select):
        # Start with this SELECT node
        results = [node]

        # Check FROM clause for subqueries
        from_clause = node.args.get("from")
        if from_clause and isinstance(from_clause, exp.From):
            # Main source in FROM
            main_source = from_clause.this
            if main_source:
                results += get_subqueries(main_source)

            # Joins in FROM
            for join_expr in from_clause.args.get("joins", []):
                right_source = join_expr.args.get("expression")
                if right_source:
                    results += get_subqueries(right_source)

        # Check WHERE and HAVING for nested subqueries
        for clause_name in ("where", "having"):
            clause = node.args.get(clause_name)
            if clause:
                # find_all Subqueries inside the clause
                for sub in clause.find_all(exp.Subquery):
                    results += get_subqueries(sub)

        # Check SELECT expressions for nested subqueries
        for e in node.select():
            for sub in e.find_all(exp.Subquery):
                results += get_subqueries(sub)

        return results

    # If it's none of the above (e.g., just a Table or Column), no subqueries here
    return []

def extract_aliases(node: exp.Expression) -> Dict[str, Dict[str, str]]:
    """
    Extract table and column aliases from the given query expression.
    This function now supports set operations by recursively extracting aliases
    from both sides of the set operation.
    
    Returns:
        {
          'table': {alias_or_table: table_name},
          'column': {alias: actual_expression_string}
        }
    """
    alias_mapping = {'table': {}, 'column': {}}

    def merge_aliases(target, source):
        for key, val in source.items():
            target[key].update(val)

    if isinstance(node, exp.Select):
        # Extract from a single SELECT
        merge_aliases(alias_mapping, _extract_aliases_from_select(node))
    elif isinstance(node, (exp.Union, exp.Intersect, exp.Except)):
        # Extract from both sides of the set operation
        left_aliases = extract_aliases(node.args.get('this'))
        right_aliases = extract_aliases(node.args.get('expression'))
        merge_aliases(alias_mapping, left_aliases)
        merge_aliases(alias_mapping, right_aliases)
    else:
        # If we encounter a node that is neither a Select nor a set operation,
        # we attempt to find Select or set operations inside it (e.g. Subquery).
        # This might be rare for top-level, but let's be safe.
        for sub in node.find_all(exp.Select):
            merge_aliases(alias_mapping, _extract_aliases_from_select(sub))
        for op_node in node.find_all(exp.Union):
            sub_aliases = extract_aliases(op_node)
            merge_aliases(alias_mapping, sub_aliases)
        for op_node in node.find_all(exp.Intersect):
            sub_aliases = extract_aliases(op_node)
            merge_aliases(alias_mapping, sub_aliases)
        for op_node in node.find_all(exp.Except):
            sub_aliases = extract_aliases(op_node)
            merge_aliases(alias_mapping, sub_aliases)

    return alias_mapping

def _extract_aliases_from_select(query: exp.Select) -> Dict[str, Dict[str, str]]:
    """
    Extract table and column aliases from a single SELECT query.
    """
    alias_mapping = {'table': {}, 'column': {}}

    # Extract table aliases from FROM clause
    from_clause = query.args.get("from")
    if from_clause and isinstance(from_clause, exp.From):
        _handle_from_clause(from_clause, alias_mapping['table'])

    # Extract column aliases from SELECT clause
    for select_exp in query.select(exclude_alias=True):
        if isinstance(select_exp, exp.Alias):
            alias = select_exp.alias
            if alias:
                alias_mapping['column'][alias.lower()] = str(select_exp.this)
        elif isinstance(select_exp, exp.Tuple):
            _extract_column_aliases_from_tuple(select_exp, alias_mapping['column'])

    return alias_mapping

def _handle_from_clause(from_clause: exp.From, table_alias_map: Dict[str, str]):
    main_source = from_clause.this
    _handle_table_or_subquery(main_source, table_alias_map)

    joins = from_clause.args.get("joins")
    if joins:
        for join_expr in joins:
            right_source = join_expr.args.get("expression")
            if right_source:
                _handle_table_or_subquery(right_source, table_alias_map)

def _handle_table_or_subquery(expr: exp.Expression, table_alias_map: Dict[str, str]):
    if isinstance(expr, exp.Table):
        table_name = expr.name
        alias = expr.alias
        if alias:
            table_alias_map[alias.lower()] = table_name.lower()
        else:
            table_alias_map[table_name.lower()] = table_name.lower()
    elif isinstance(expr, exp.Subquery):
        # For subqueries, we want just the inner SELECT (if present), not the parentheses or alias.
        alias = expr.alias
        if alias:
            inner_select = expr.args.get("this")
            if inner_select and isinstance(inner_select, exp.Select):
                # Use the string form of the select only
                subquery_str = '(' + str(inner_select) + ')'
            else:
                # Fallback if somehow not a SELECT
                subquery_str = str(expr)
            table_alias_map[alias.lower()] = subquery_str
    elif isinstance(expr, exp.Select):
        # A derived table: treat similarly to a subquery, but it might not be wrapped in parentheses.
        alias = expr.alias
        if alias:
            # Return just the SELECT's string
            table_alias_map[alias.lower()] = '(' + str(expr) + ')'

def _extract_column_aliases_from_tuple(tuple_expr: exp.Tuple, column_alias_map: Dict[str, str]):
    for inner_expr in tuple_expr.expressions:
        if isinstance(inner_expr, exp.Alias):
            alias = inner_expr.alias
            if alias:
                column_alias_map[alias.lower()] = str(inner_expr.this)
        elif isinstance(inner_expr, exp.Tuple):
            _extract_column_aliases_from_tuple(inner_expr, column_alias_map)

def extract_selection(query: exp.Select, aliases: Dict[str, Dict[str, str]], schema: Schema) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    """
    Extracts information about the columns and expressions used in the SELECT clause of a query.

    Parameters:
        query (exp.Select): The SQL SELECT query expression to analyze.
        aliases (Dict[str, Dict[str, str]]): A dictionary containing alias mappings.
            - 'table': Maps table aliases to their corresponding table names.
            - 'column': Maps column aliases to their corresponding full expressions.
        schema (Schema): An object that provides schema information, including table and column mappings.

    Returns:
        Tuple[Set[str], Set[Tuple[str, str]]]:
            - A set of unique column names, fully qualified (e.g., '__table.column__').
            - A set of tuples where each tuple contains:
                - A string representation of the expression or column used in the SELECT clause.
                - A tag indicating the type of the expression:
                    - '<s>': Simple column reference (e.g., `table.column` or `column`).
                    - '<f>': Function or calculation (e.g., `SUM(column)` or `column + 1`).

    Notes:
        - This function handles column aliases by replacing them with their original expressions.
        - Expressions that involve functions or calculations are tagged as '<f>'.
        - The result only includes columns and expressions explicitly referenced in the SELECT clause.
        - Any '*' in the SELECT clause is replaced with '__all__' based on the schema configuration.

    Example:
        Given the query:
            SELECT cli.first as f, cli.middle + 1 as m, COUNT(cli.name) as cnt 
            FROM client cli
        
        And the schema mapping:
            {'client': ['first', 'middle', 'name']}
        
        The function would return:
            (
                {'__client.first__', '__client.middle__', '__client.name__'},
                {('__client.first__', '<s>'), ('__client.middle__ + 1', '<f>'), ('COUNT(__client.name__)', '<f>')}
            )
    """
    unique_columns = set()
    selection_types = set()

    for select_exp in query.select():
        columns_in_item = _extract_columns_from_expression(select_exp, aliases, schema)
        unique_columns.update(columns_in_item)

        tag = _determine_tag(select_exp)
        expr_str = _format_expression(select_exp, aliases, schema, remove_alias=True)
        selection_types.add((expr_str, tag))

    return unique_columns, selection_types


def _determine_tag(expr: exp.Expression) -> str:
    # All functions or binary operations => <f>, else <s>
    if isinstance(expr, exp.Column):
        return '<s>'
    if isinstance(expr, (exp.Func, exp.Binary, exp.Condition)):
        return '<f>'
    # Check nested functions or binaries inside
    if any(isinstance(x, exp.Func) for x in expr.find_all(exp.Func)):
        return '<f>'
    if any(isinstance(x, exp.Binary) for x in expr.find_all(exp.Binary)):
        return '<f>'

    # If it's just columns, <s>, else default <s>
    if all(isinstance(x, exp.Column) for x in expr.find_all(exp.Column)):
        return '<s>'
    return '<s>'


def _format_expression(expr: exp.Expression, aliases: Dict[str, Dict[str, str]], schema: Schema, remove_alias: bool = False) -> str|Tuple[str, str]:
    if remove_alias and isinstance(expr, exp.Alias):
        return _format_expression(expr.this, aliases, schema, remove_alias=True)

    if isinstance(expr, exp.Column):
        return _get_full_column_name(expr, aliases, schema)

    if isinstance(expr, exp.Func):
        func_name = type(expr).__name__
        if func_name.lower() == 'substr':
            func_name = 'substring'
        else:
            func_name = func_name.lower()

        args_list = []
        main_arg = expr.args.get("this")
        if main_arg:
            args_list.append(_format_expression(main_arg, aliases, schema, remove_alias=remove_alias))

        for additional_arg in expr.args.get("expressions") or []:
            args_list.append(_format_expression(additional_arg, aliases, schema, remove_alias=remove_alias))

        return f"{func_name}({', '.join(args_list)})"

    if isinstance(expr, exp.Tuple):
        formatted = []
        for t in expr.expressions:
            formatted.append(_format_expression(t, aliases, schema, remove_alias=remove_alias))
        return f"({', '.join(formatted)})"

    if isinstance(expr, exp.Binary):
        # Binary operators like EQ, LT, GT, etc.
        left = _format_expression(expr.args.get('this'), aliases, schema, remove_alias=remove_alias)
        right = _format_expression(expr.args.get('expression'), aliases, schema, remove_alias=remove_alias)
        op = OPERATOR_MAP.get(type(expr), expr.key)  # fallback to expr.key if not in map
        return op, f"{left} {op} {right}"

    # if isinstance(expr, exp.Condition):
    #     left = _format_expression(expr.args.get('this'), aliases, schema, remove_alias=remove_alias)
    #     right = _format_expression(expr.args.get('expression'), aliases, schema, remove_alias=remove_alias)
    #     op = expr.key.lower()
    #     return f"{left} {op} {right}"

    if isinstance(expr, exp.Alias) and not remove_alias:
        base_str = _format_expression(expr.this, aliases, schema, remove_alias=remove_alias)
        return f"{base_str} as {expr.alias.lower()}"

    if isinstance(expr, exp.Identifier):
        return expr.name.lower()

    if isinstance(expr, exp.Subquery):
        subexpr = expr.args.get('this')
        if subexpr:
            return f"({_format_expression(subexpr, aliases, schema, remove_alias=remove_alias)})"
        return "(subquery)"

    if isinstance(expr, exp.Select):
        cols = [_format_expression(s, aliases, schema, remove_alias=remove_alias) for s in expr.select()]
        return "SELECT " + ", ".join(cols)

    if isinstance(expr, exp.Literal):
        val = expr.this
        if expr.is_string:
            return f"'{val}'"
        return str(val)

    return str(expr).lower()


def _get_full_column_name(col: exp.Column, aliases: Dict[str, Dict[str, str]], schema: Schema) -> str:
    column_name = col.name.lower()

    # If the column is *, map directly to __all__
    if column_name == '*':
        return schema.idMap['*']

    table_alias = col.table.lower() if col.table else None

    if table_alias and table_alias in aliases['table']:
        real_table_name = aliases['table'][table_alias]
    else:
        possible_tables = list(aliases['table'].values())
        real_table_name = schema.get_table_name(column_name, possible_tables)

    if real_table_name:
        key = f"{real_table_name.lower()}.{column_name}"
    else:
        key = column_name
    return schema.idMap.get(key, f"__{key}__")


def _extract_columns_from_expression(expr: exp.Expression, aliases: Dict[str, Dict[str, str]], schema: Schema) -> Set[str]:
    columns = set()
    for col in expr.find_all(exp.Column, exp.Star):
        full_col_name = _get_full_column_name(col, aliases, schema)
        columns.add(full_col_name)
    
    return columns

def extract_condition(query: exp.Select, aliases: Dict[str, Dict[str, str]], schema: Schema) -> Tuple[Set[str], Set[str]]:
    """
    Extracts information about conditions and operators used in the WHERE and HAVING clauses of a query.

    Parameters:
        query (exp.Select): The SQL SELECT query expression to analyze.
        aliases (Dict[str, Dict[str, str]]): A dictionary containing alias mappings.
            - 'table': Maps table aliases to their corresponding table names.
            - 'column': Maps column aliases to their corresponding full expressions.
        schema (Schema): An object that provides schema information, including table and column mappings.

    Returns:
        Tuple[Set[str], Set[str]]:
            - A set of unique conditions as strings, fully qualified with schema mappings
              (e.g., '__table.column__ = value').
            - A set of unique operator types used in the conditions (e.g., '=', '>', '<', 'IN').

    Notes:
        - This function handles both WHERE and HAVING clauses.
        - Nested subqueries within the conditions are processed recursively.
        - Conditions are extracted and formatted to replace aliases with fully qualified schema names.
        - Operators are extracted from each condition to provide insight into the logical structure.

    Example:
        Given the query:
            SELECT cli.first, COUNT(cli.name) 
            FROM client cli 
            WHERE cli.name = 'John' AND cli.age > 30
            HAVING COUNT(cli.name) > 5
        
        And the schema mapping:
            {'client': ['first', 'name', 'age']}
        
        The function would return:
            (
                {
                    "__client.name__ = 'John'", 
                    "__client.age__ > 30", 
                    "COUNT(__client.name__) > 5"
                },
                {'=', '>', 'COUNT'}
            )
    """
    conditions = set()
    operator_types = set()

    for clause_name in ("where", "having"):
        clause = query.args.get(clause_name)
        if clause:
            # clause is WHERE or HAVING. We only want the conditions inside.
            # clause.this is the actual condition expression (e.g., AND/OR tree).
            ops, conds = _extract_conditions(clause.this, aliases, schema)
            conditions.update(conds)
            operator_types.update(ops)
    return conditions, operator_types

def _extract_conditions(expr: exp.Expression, aliases: Dict[str, Dict[str, str]], schema: Schema) -> Tuple[List[str], List[str]]:
    """
    Flatten a WHERE/HAVING condition (which may be a complex AND/OR tree) into a list of individual conditions.
    Each leaf condition is returned as a separate string.
    """
    if isinstance(expr, (exp.And, exp.Or)):
        # Recurse into both sides
        operations = []
        conditions = []
        left = expr.args.get('this')
        right = expr.args.get('expression')
        if left:
            ops, conds = _extract_conditions(left, aliases, schema)
            operations.extend(ops)
            conditions.extend(conds)
        if right:
            ops, conds = _extract_conditions(right, aliases, schema)
            operations.extend(ops)
            conditions.extend(conds)
        return operations, conditions
    else:
        # This is a leaf condition (equality, comparison, etc.)
        # Format this condition to replace columns and remove aliases
        op, cond = _format_condition_expression(expr, aliases, schema)
        return [op], [cond]

def _format_condition_expression(expr: exp.Expression, aliases: Dict[str, Dict[str, str]], schema: Schema) -> Tuple[str, str]:
    """
    Format a leaf condition expression into a string, with columns replaced by schema-based names.
    We can reuse the _format_expression function defined previously to ensure consistency.
    """
    # We'll assume we have a _format_expression function from earlier steps.
    # It converts columns to __table.column__, literals to strings, etc.
    op, cond = _format_expression(expr, aliases, schema, remove_alias=True)
    return op, cond

def extract_aggregation(query: exp.Select, aliases: Dict[str, Dict[str, str]], schema: Schema) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    """
    Extracts information about columns and expressions used in the GROUP BY clause of a query.

    Parameters:
        query (exp.Select): The SQL SELECT query expression to analyze.
        aliases (Dict[str, Dict[str, str]]): A dictionary containing alias mappings.
            - 'table': Maps table aliases to their corresponding table names.
            - 'column': Maps column aliases to their corresponding full expressions.
        schema (Schema): An object that provides schema information, including table and column mappings.

    Returns:
        Tuple[Set[str], Set[Tuple[str, str]]]:
            - A set of unique column names, fully qualified (e.g., '__table.column__').
            - A set of tuples where each tuple contains:
                - A string representation of the expression or column used in the GROUP BY clause.
                - A tag indicating the type of the expression:
                    - '<s>': Simple column reference (e.g., `table.column` or `column`).
                    - '<f>': Function or calculation (e.g., `SUM(column)` or `column + 1`).

    Notes:
        - The function processes the GROUP BY clause to identify columns and expressions.
        - Column aliases are replaced with their original expressions where applicable.
        - Functions and calculations in the GROUP BY clause are tagged as '<f>'.
        - Ensures consistency with the schema mappings for fully qualified column names.

    Example:
        Given the query:
            SELECT cli.first, COUNT(cli.name) 
            FROM client cli 
            GROUP BY cli.first, cli.age
        
        And the schema mapping:
            {'client': ['first', 'name', 'age']}
        
        The function would return:
            (
                {'__client.first__', '__client.age__'},
                {('__client.first__', '<s>'), ('__client.age__', '<s>')}
            )
    """
    unique_columns = set()
    aggregation_types = set()

    group = query.args.get('group')
    if group:
        for g in group:
            columns_in_item = _extract_columns_from_expression(g, aliases, schema)
            unique_columns.update(columns_in_item)
            tag = _determine_tag(g)
            aggregation_types.add((str(g).lower().strip(), tag))

    return unique_columns, aggregation_types

def extract_others(query: exp.Select, aliases: Dict[str, Dict[str, str]], schema: Schema) -> Dict[str, Any]:
    """
    Extracts:
    1. distinct columns (if DISTINCT is present)
    2. order by columns
    3. whether LIMIT is used
    """
    others = {'distinct': set(), 'order by': set(), 'limit': False}

    # Check for DISTINCT
    distinct_node = query.args.get('distinct')
    if distinct_node and isinstance(distinct_node, exp.Distinct):
        # DISTINCT applies to all selected columns/expressions
        # Extract columns from all SELECT expressions
        for sel_expr in query.select():
            columns = _extract_columns_from_expression(sel_expr, aliases, schema)
            others['distinct'].update(columns)

    # Check for ORDER BY
    order_node = query.args.get('order')
    if order_node and isinstance(order_node, exp.Order):
        # order_node.expressions is a list of Ordered expressions
        for ordered_expr in order_node.expressions:
            # ordered_expr.this is the actual expression being ordered
            columns = _extract_columns_from_expression(ordered_expr.this, aliases, schema)
            others['order by'].update(columns)

    # Check for LIMIT
    limit_node = query.args.get('limit')
    if limit_node and isinstance(limit_node, exp.Limit):
        # presence of limit node means limit is used
        others['limit'] = True

    return others

def extract_all(parsed_query: exp.Expression, schema: Schema) -> Tuple[Set[str], Set[Tuple[str, str]], Set[str], Set[str], Set[Tuple[str, str]], Dict[str, Any]]:
    """
    Extract all components from a SELECT query.
    """
    
    aliases = extract_aliases(parsed_query)
    subqueries = get_subqueries(parsed_query)
    results = defaultdict(set)
    nested = len(subqueries)

    for query in subqueries:
        sel_cols, sel_types  = extract_selection(query, aliases, schema)
        conds, op_types = extract_condition(query, aliases, schema)
        agg_cols, agg_types  = extract_aggregation(query, aliases, schema)
        others = extract_others(query, aliases, schema)
        
        results['sel'].update(sel_cols)
        results['sel_types'].update(sel_types)
        results['cond'].update(conds)
        results['op_types'].update(op_types)
        results['agg'].update(agg_cols)
        results['agg_types'].update(agg_types)
        results['distinct'].update(others['distinct'])
        results['order by'].update(others['order by'])
        results['limit'] = others['limit']

    results['nested'] = nested

    return results

if __name__ == "__main__":
    sql = "SELECT * FROM table1 WHERE col1 > 10 AND col2 = 'abc' ORDER BY col3 LIMIT 10"
    query = sqlglot.parse_one(sql)
    aliases = extract_aliases(query)
    schema = Schema()
    selection = extract_selection(query, aliases, schema)
    print(selection)
    condition = extract_condition(query, aliases, schema)
    print(condition)
    aggregation = extract_aggregation(query, aliases, schema)
    print(aggregation)
    others = extract_others(query, aliases, schema)
    print(others)

# class ControlFlow(BaseModel):
#     select_seen: bool = Field(default=False)
#     from_seen: bool = Field(default=False)
#     join_seen: bool = Field(default=False)
#     where_seen: bool = Field(default=False)
#     groupby_seen: bool = Field(default=False)
#     having_seen: bool = Field(default=False)
#     orderby_seen: bool = Field(default=False)
#     limit_seen: bool = Field(default=False)

# def switch_control_flow(token, control_flow: ControlFlow):
#     if token.ttype is tks.DML and token.value.upper() == 'SELECT':
#         control_flow.select_seen = True

#     if token.ttype is tks.Keyword and token.value.upper() == 'FROM':
#         control_flow.select_seen = False
#         control_flow.from_seen = True

#     if token.ttype is tks.Keyword and token.value.upper() == 'JOIN':
#         control_flow.select_seen = False
#         control_flow.from_seen = False
#         control_flow.join_seen = True

#     if isinstance(token, Where) or (token.ttype is tks.Keyword and token.value.upper() == 'WHERE'):
#         control_flow.select_seen = False
#         control_flow.from_seen = False
#         control_flow.join_seen = False
#         control_flow.where_seen = True
        
#     if token.ttype is tks.Keyword and token.value.upper() == 'GROUP BY':
#         control_flow.from_seen = False
#         control_flow.join_seen = False
#         control_flow.where_seen = False
#         control_flow.groupby_seen = True
    
#     if token.ttype is tks.Keyword and token.value.upper() == 'HAVING':
#         control_flow.groupby_seen = False
#         control_flow.having_seen = True

#     if token.ttype is tks.Keyword and token.value.upper() == 'ORDER BY':
#         control_flow.from_seen = False
#         control_flow.join_seen = False
#         control_flow.where_seen = False
#         control_flow.having_seen = False
#         control_flow.groupby_seen = False
#         control_flow.having_seen = False
#         control_flow.orderby_seen = True

#     if token.ttype is tks.Keyword and token.value.upper() == 'LIMIT':
#         control_flow.from_seen = False
#         control_flow.join_seen = False
#         control_flow.where_seen = False
#         control_flow.having_seen = False
#         control_flow.groupby_seen = False
#         control_flow.having_seen = False
#         control_flow.orderby_seen = False
#         control_flow.limit_seen = True

# # def is_actual_keyword(token):
# #     KEYWORDS = (
# #         'SELECT', 'FROM', 'JOIN', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT',
# #         'UNION', 'INTERSECT', 'EXCEPT', 'AS', 'ON', 'AND', 'OR', 'NOT', 'IN', 'LIKE', 'BETWEEN', 'IS', 'NULL',
# #         'INNER', 'LEFT', 'RIGHT', 'FULL', 'OUTER', 'CROSS', 'NATURAL', 'ASC', 'DESC', 'DISTINCT', 'EXISTS',
# #         'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'ALL', 'ANY', 'WITH'
# #     )
# #     if token.value in KEYWORDS:
# #         return True
# #     return False

# def check_keyword_in_token(token):
#     KEYWORDS = (
#         'SELECT', 'FROM', 'JOIN', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT',
#         'UNION', 'INTERSECT', 'EXCEPT', 'AS', 'ON', 'AND', 'OR', 'NOT', 'IN', 'LIKE', 'BETWEEN', 'IS', 'NULL',
#         'INNER', 'LEFT', 'RIGHT', 'FULL', 'OUTER', 'CROSS', 'NATURAL', 'ASC', 'DESC', 'DISTINCT', 'EXISTS',
#         'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'ALL', 'ANY', 'WITH'
#     )
#     for key in KEYWORDS:
#         if key.lower() in str(token).lower():
#             return True
#     return False


# def is_pwn(token, only_wn=False):
#     if only_wn:
#         rule = (tks.Whitespace, tks.Newline)
#     else:
#         rule = (tks.Punctuation, tks.Whitespace, tks.Newline)
#     if token.ttype in rule:
#         return True
#     return False

# def is_set_operator(token):
#     if token.ttype is tks.Keyword and token.value.upper() in ('UNION', 'INTERSECT', 'EXCEPT'):
#         return True
#     return False

# def is_subquery(tokens: Parenthesis|Identifier):
#     for token in tokens:
#         if token.ttype is tks.DML and token.value.upper() == 'SELECT':
#             return True
#     return False

# def split_set_operation(statement: Statement) -> list[Statement]:
#     """
#     Checks if the statement contains a set operation.
#     """
#     is_set = False
#     statement1 = []
#     statement2 = []
#     tokens = statement.tokens
#     for i, token in enumerate(tokens):
#         if is_set_operator(token):
#             break
#         else:
#             if isinstance(token, Where):
#                 # only where clause has a subquery form
#                 for j, sub_token in enumerate(token.tokens):
#                     if is_set_operator(sub_token):
#                         is_set = True
#                         break
#                     else:
#                         statement1.append(sub_token)
#             else:
#                 statement1.append(token)

#     if is_set:
#         statement2 = token.tokens[j+1:]
#     else:
#         is_set_new = False
#         for i, token in enumerate(tokens):
#             if is_set_operator(token):
#                 is_set_new = True
#                 break
#         if is_set_new:
#             statement2 = tokens[i+1:]

#     # post process the statement
#     if is_pwn(statement1[-1]):
#         statement1 = statement1[:-1]
#     if statement2 and is_pwn(statement2[0]):
#         statement2 = statement2[1:]
    
#     if statement2:
#         return [Statement(statement1), Statement(statement2)]
#     return [Statement(statement1)]

# # --------------------------------------------------------------------------------------------
# # Extracting Aliases
# # --------------------------------------------------------------------------------------------

# def extract_aliases(statement: Statement) -> dict[str, dict[str, str]]:
#     """
#     Extracts a mapping of table & column aliases to their actual table names from the SQL statement.

#     Assumptions:
#     - Only table names have aliases.
#     - Subqueries can occur in WHERE and HAVING clauses.
#     - Only one set operation or subquery can occur.
#     """
#     alias_mapping = {'table': {}, 'column': {}}
#     # Handle set operations
#     statements = split_set_operation(statement)

#     for stmt in statements:
#         extract_alias(stmt, alias_mapping, ControlFlow())
#     return alias_mapping

# def process_token_alias(token, alias_mapping: dict[str, dict[str, str]], alias_str: str):
#     alias_str2func = {
#         'column': extract_alias_column_name,
#         'table': extract_alias_table_name
#     }
#     func = alias_str2func[alias_str]
#     if isinstance(token, IdentifierList):
#         for identifier in token.get_identifiers():
#             if isinstance(identifier, (Identifier, Function)):
#                 # If it's an Identifier or Function, we can check for subqueries
#                 # safely because these have a .tokens attribute.
#                 if (identifier.tokens and isinstance(identifier.tokens[0], Parenthesis)
#                         and is_subquery(identifier.tokens[0])):
#                     extract_alias(identifier.tokens[0], alias_mapping, control_flow=ControlFlow())
#                 func(identifier, alias_mapping[alias_str])
#             elif isinstance(identifier, Parenthesis):
#                 # Handle subqueries inside parentheses if any
#                 subqueries = extract_subqueries([identifier])
#                 if subqueries:
#                     for sub_statement in subqueries:
#                         extract_alias(sub_statement, alias_mapping, control_flow=ControlFlow())
#                 # If needed, process this as a table/column afterwards
#                 # func(identifier, alias_mapping[alias_str]) # only if makes sense
#             elif isinstance(identifier, Token):
#                 # It's just a single token column name
#                 # Convert it into an Identifier-like object for consistency
#                 # or handle directly.
#                 if identifier.ttype in (tks.Name, Identifier):
#                     # Wrap in Identifier for uniform handling
#                     single_identifier = Identifier([identifier])
#                     func(single_identifier, alias_mapping[alias_str])
#                 else:
#                     # If it's not a name or identifier type, handle accordingly
#                     pass
#     elif isinstance(token, (Identifier, Function)):
#         if isinstance(token[0], Parenthesis) and is_subquery(token[0]):
#             # Handle subquearies in with alias
#             extract_alias(token[0], alias_mapping, control_flow=ControlFlow())
#         func(token, alias_mapping[alias_str])
#     elif isinstance(token, Parenthesis):
#         # Handle subqueries with no alias
#         subqueries = extract_subqueries([token])
#         if subqueries:
#             for sub_statement in subqueries:
#                 extract_alias(sub_statement, alias_mapping, control_flow=ControlFlow())
#         # func(token, alias_mapping[alias_str])
#     elif isinstance(token, (Where, Comparison)):
#         if token.value.upper() != 'WHERE':
#             for sub_token in token.tokens:
#                 if isinstance(sub_token, Parenthesis) and is_subquery(sub_token):
#                     # Handle subqueries in where clause
#                     extract_alias(sub_token, alias_mapping, control_flow=ControlFlow())
#     else:
#         # Other tokens
#         pass

# def extract_alias(
#         stmt, 
#         alias_mapping: dict[str, dict[str, str]], 
#         control_flow: ControlFlow
#     ):
#     if isinstance(stmt, Statement):
#         iterator = stmt.tokens
#     else:
#         iterator = stmt

#     for token in iterator:
#         # Skip comments and whitespace
#         if token.is_whitespace or token.ttype in (sqlparse.tokens.Newline,):
#             continue
        
#         switch_control_flow(token, control_flow)

#         if control_flow.select_seen or control_flow.having_seen:
#             process_token_alias(token, alias_mapping, 'column')
#         elif control_flow.from_seen or control_flow.join_seen:
#             process_token_alias(token, alias_mapping, 'table')
#         elif control_flow.where_seen:
#             process_token_alias(token, alias_mapping, 'column')
#             if isinstance(token, (Where, Comparison)):
#                 process_token_alias(token, alias_mapping, 'table')
#         else:
#             pass
     
# def extract_alias_column_name(identifier, alias_column: dict):
#     """
#     Processes an identifier to extract column name and alias.
#     """
#     column_components = []
#     alias = identifier.get_alias()

#     if alias is None:
#         column_components.append(identifier)
#     else:
#         # col as alias, col alias
#         for x in identifier:
#             if x.ttype is tks.Keyword and x.value.upper() == 'AS':
#                 break
#             else:
#                 column_components.append(x)
#         if column_components[-1].value == alias:
#             column_components = column_components[:-2]

#     column = format_column(column_components)
#     alias = alias.lower() if alias is not None else None

#     key = column if alias is None else alias
#     if key == column:
#         return None
#     elif alias_column.get(key) is None:
#         alias_column[key] = column
#     elif alias is not None:
#         raise ValueError(f'Alias {alias} already exists in the column mapping.\n{alias_column}')

# def format_column(column_components: list[Token]) -> str:
#     tokens = []
#     for token in TokenList(column_components).flatten():
#         if token.ttype is tks.Literal.String.Single:
#             tokens.append(str(token))
#         else:
#             tokens.append(str(token).lower())
            
#     column = ''.join(tokens).strip()
#     return column

# def extract_alias_table_name(identifier, alias_table: dict):
#     """
#     Processes an identifier to extract table name and alias.
#     """
#     alias = identifier.get_alias()
#     if isinstance(identifier[0], Parenthesis):
#         table_name = format_column(identifier[0].tokens)
#     else:
#         table_name = identifier.get_real_name().lower()

#     if alias:
#         alias_table[alias.lower()] = table_name
#     else:
#         alias_table[table_name] = table_name

# def extract_subqueries(tokens):
#     subqueries = []
#     for token in tokens:
#         if isinstance(token, Parenthesis):
#             inner_tokens = token.tokens
#             # Check if the first meaningful token is a SELECT statement
#             for i, t in enumerate(inner_tokens):
#                 if t.ttype in (tks.Punctuation, tks.Whitespace, tks.Newline):
#                     continue
#                 if t.ttype is tks.DML and t.value.upper() == 'SELECT':
#                     # Found a subquery
#                     subquery = inner_tokens[i:]
#                     if subquery[-1].value == ')':
#                         subquery = subquery[:-1]
#                     subqueries.append(Statement(subquery))
#                     break
#     return subqueries

# def get_source_tables(aliases: dict[str, dict[str, str]]) -> list[str]:
#     source_tables = set()
#     for value in aliases['table'].values():
#         if '(' in value:
#             continue
#         source_tables.add(value)
#     return source_tables

# # --------------------------------------------------------------------------------------------
# # Extracting Selections
# # --------------------------------------------------------------------------------------------

# def extract_selection(statement: Statement, aliases: dict[str, dict[str, str]], schema: Schema) -> tuple[set, set]:
#     """
#     Extracts following information that used in the SELECT clause.
#     (1) Number of expressions used. 
#     (2) Number of logical calculations, functions used 
#         - simple column part '<s>', 
#         - calculation part '<c>'
#         - function part '<a>'

#     Assumptions:
#     - Only table names have aliases.
#     - Subqueries can occur in WHERE and HAVING clauses.
#     - Only one set operation or subquery can occur.
#     """
#     statements = split_set_operation(statement)
#     unique_columns = set()
#     selection_types = set()
#     for stmt in statements:
#         extract_select(stmt, aliases, schema, unique_columns, selection_types, ControlFlow())
#     return unique_columns, selection_types

# def extract_select(
#         stmt: Statement, 
#         aliases: dict[str, dict[str, str]], 
#         schema: Schema,
#         unique_columns: set, 
#         selection_types: set,
#         control_flow: ControlFlow
#     ):
#     """
#     Extracts columns and partial selections from the SELECT clause of the statement.
#     """
#     if isinstance(stmt, Statement):
#         iterator = stmt.tokens
#     else:
#         iterator = stmt

#     for token in iterator:
#         if token.is_whitespace or token.ttype in (tks.Newline, tks.Whitespace):
#             continue

#         switch_control_flow(token, control_flow)

#         if control_flow.select_seen:
#             if is_pwn(token):
#                 continue
            
#             if isinstance(token, IdentifierList):
#                 for identifier in token.get_identifiers():
#                     print(identifier, identifier.ttype, type(identifier))
#                     if isinstance(identifier, Parenthesis):
#                         if is_subquery([identifier]):
#                             # Handle subqueries if has alias
#                             extract_select([identifier], aliases, schema, unique_columns, selection_types, control_flow=ControlFlow())
#                         else:
#                             extract_select(identifier, aliases, schema, unique_columns, selection_types, control_flow=control_flow)
#                     else:
#                         process_item(identifier, aliases, schema, unique_columns, selection_types)
#             elif isinstance(token, (Identifier, Function)):
#                 if is_subquery([token]):
#                     # Handle subqueries if has alias
#                     extract_select(token[0], aliases, schema, unique_columns, selection_types, control_flow=ControlFlow())
#                 else:
#                     process_item(token, aliases, schema, unique_columns, selection_types)
#             elif isinstance(token, Parenthesis):
#                 subqueries = extract_subqueries([token])
#                 if subqueries:
#                     for sub_statement in subqueries:
#                         extract_select(sub_statement, aliases, schema, unique_columns, selection_types, control_flow=ControlFlow())
#                 else:
#                     for sub_token in token.tokens:
#                         print(sub_token, sub_token.ttype, type(sub_token))
#                         if is_pwn(sub_token):
#                             continue
#                         extract_select(sub_token, aliases, schema, unique_columns, selection_types, control_flow=control_flow)
#                 #     process_item(token, aliases, schema, unique_columns, selection_types)
#             elif not check_keyword_in_token(token):
#                 print(token, token.ttype, type(token))
#                 process_item(token, aliases, schema, unique_columns, selection_types)
#             else:
#                 # Token
#                 pass
                
# def process_item(token, aliases: dict[str, dict[str, str]], schema: Schema, unique_columns: set, types: set):
#     """
#     Processes an item and updates unique_columns and types.
#     Extracts following information.
#     (1) Number of expressions used. 
#     (2) Number of logical calculations, functions used 
#         - simple column part '<s>', 
#         - calculation part '<c>'
#         - aggregate function part '<a>'
#     """
#     tag = None
#     expression = []
#     if isinstance(token, Function):
#         # It's a function
#         columns_in_item = extract_columns_from_expression(token, aliases, schema, expression)
#         unique_columns.update(columns_in_item)
#         tag = '<a>' if is_aggregate_function(token) else '<c>'
#     elif isinstance(token, Identifier):
#         # Simple column or Fucntion with column
#         if isinstance(token[0], Function):  # alias
#             columns_in_item = extract_columns_from_expression(token[0], aliases, schema, expression)
#             unique_columns.update(columns_in_item)
#             tag = '<a>' if is_aggregate_function(token[0]) else '<c>'
#         elif token.value.lower() in aliases['column']:
#             new_token = sqlparse.parse(aliases['column'][token.value.lower()])[0].tokens[0]
#             process_item(new_token, aliases, schema, unique_columns, types)
#         else:
#             # column_name = get_full_column_name(token, aliases, schema)
#             # unique_columns.add(column_name)
#             # tag = '<s>'
#             columns_in_item = extract_columns_from_expression(token, aliases, schema, expression)
#             unique_columns.update(columns_in_item)
#             tag = '<s>'
#     elif isinstance(token, Operation):
#         # Operation
#         columns_in_item = extract_columns_from_expression(token, aliases, schema, expression)
#         unique_columns.update(columns_in_item)
#         tag = '<c>'
#     elif isinstance(token, Parenthesis):
#         # operation or subquery
#         subqueries = extract_subqueries([token])
#         if subqueries:
#             for sub_statement in subqueries:
#                 process_item(sub_statement, aliases, schema, unique_columns, types)
#         else:
#             for sub_token in token.tokens:
#                 if is_pwn(sub_token):
#                     continue
#                 process_item(sub_token, aliases, schema, unique_columns, types)
#     else:
#         # Other tokens
#         pass
#     # Get the token string
#     if tag:
#         selection_text = str(''.join(expression)).strip().lower()  # token
#         types.update([(selection_text, tag)])

# def get_full_column_name(token, aliases: dict[str, dict[str, str]], schema: Schema) -> str:
#     """
#     Returns the fully qualified column name (table.column) from a token.
#     """
#     if isinstance(token, Identifier):
#         column_name = token.get_real_name().lower()
#         if aliases['column'].get(column_name):
#             real_column_name = aliases['column'][column_name]
#         else:
#             real_column_name = column_name

#         table_alias = token.get_parent_name().lower() if token.get_parent_name() else token.get_parent_name()  # None for *
#         if table_alias:
#             real_table_name = aliases['table'].get(table_alias)
#         else:
#             real_table_name = schema.get_table_name(column_name, tables=list(aliases['table'].values()))
            
#         if real_table_name:
#             key = f"{real_table_name}.{real_column_name}"
#         else:
#             key = f"{real_column_name}"

#         try: 
#             return schema.idMap[key]
#         except:
#             return '__' + key + '__'
#     elif token.ttype is tks.Wildcard:
#         return schema.idMap['*']
#     else:
#         assert False, f"Unexpected token type: {type(token), token.ttype} - {token}"
    
# def extract_columns_from_expression(token, aliases: dict[str, str], schema: Schema, expression: list) -> set:
#     """
#     Recursively extracts column names from an expression.
#     """
#     columns = set()
#     print(token, token.ttype, type(token))
#     if isinstance(token, Identifier) or token.ttype is tks.Wildcard:
#         if 'DESC' in str(token):
#             # no column in order by desc
#             expression.append(str(token))
#         else:
#             column_name = get_full_column_name(token, aliases, schema)
#             columns.add(column_name)
#             expression.append(column_name)
#     elif isinstance(token, IdentifierList):
#         for identifier in token.get_identifiers():
#             columns.update(extract_columns_from_expression(identifier, aliases, schema, expression))
#     elif isinstance(token, Function):
#         expression.append(str(token[0]).lower())
#         for sub_token in token.tokens[1:]:
#             columns.update(extract_columns_from_expression(sub_token, aliases, schema, expression))
#     elif isinstance(token, (Operation, Parenthesis)):
#         for sub_token in token.tokens:
#             columns.update(extract_columns_from_expression(sub_token, aliases, schema, expression))
#     elif not check_keyword_in_token(token):
#         # check if it is a keyword, FIRST, LAST, etc.
#         columns.update(extract_columns_from_expression(sub_token, aliases, schema, expression))
#     else:
#         # Other tokens, possibly operators or literals
#         expression.append(str(token))
#     return columns

# def is_aggregate_function(token):
#     """
#     Determines if the function is an aggregate function.
#     """
#     AGG_OPS = ('max', 'min', 'count', 'sum', 'avg', 'stddev', 'variance')
#     if isinstance(token, (Function, Identifier)):
#         function_name = token.get_name()
#         if function_name:
#             return function_name.lower() in AGG_OPS
#     return False

# # --------------------------------------------------------------------------------------------
# # Extracting Conditions
# # --------------------------------------------------------------------------------------------

# def extract_condition(
#         statement: Statement, 
#         aliases: dict[str, dict[str, str]], 
#         schema: Schema
#     ) -> tuple[set, set]:
#     """
#     Extracts following information that used in the WHERE and HAVING clause.
#     (1) the number of conditions used
#     (2) the number of kinds of conditions used

#     WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', '<>', 'in', 'like', 'is', 'exists')

#     Assumptions:
#     - Only table names have aliases.
#     - Subqueries can occur in WHERE and HAVING clauses.
#     - Only one set operation or subquery can occur.
#     """
#     statements = split_set_operation(statement)
#     operator_types = set()
#     conditions = []
#     for stmt in statements:
#         extract_where_having(stmt, conditions, ControlFlow())
#         extract_operation_types(conditions, operator_types)
#     conditions = formating_conditions(conditions, aliases, schema)
#     conditions = [sqlparse.format(str(c), reindent=True).strip() for c in conditions]
#     conditions = set(sorted(conditions))
#     return conditions, operator_types

# def process_where_having(token, cond: list, conditions: list, between_seen: bool=False):
#     if isinstance(token, Comparison):
#         conditions.append(token)
#     elif isinstance(token, Parenthesis):
#         # two cases: subquery or expression  (a or/and b)
#         subqueries = extract_subqueries([token])
#         if subqueries:
#             for sub_statement in subqueries:
#                 # get WHERE clause
#                 sub_where = extract_where_having(sub_statement, conditions, control_flow=ControlFlow())
#                 if sub_where:
#                     conditions.append(sub_where)
#         else:
#             for sub_token in token.tokens:
#                 process_where_having(sub_token, cond, conditions, between_seen)
#     elif (token.ttype is tks.Keyword and token.value.upper() == 'WHERE'):
#         pass
#     elif isinstance(token, Where) and token.value != 'WHERE':
#         # means there are something under the WHERE clause
#         for sub_token in token.tokens:
#             process_where_having(sub_token, cond, conditions, between_seen)
#     else:
#         if not is_pwn(token):
#             if between_seen or (token.value not in ('AND', 'OR')):
#                 # between -> add AND or OR
#                 cond.extend([token, Token(tks.Whitespace, ' ')])
    
# def extract_where_having(stmt, conditions: list, control_flow: ControlFlow):
#     between_seen = False   
#     cond = []

#     if isinstance(stmt, Statement):
#         iterator = stmt.tokens
#     else:
#         iterator = stmt

#     for token in iterator:
#         if token.is_whitespace or token.ttype in (tks.Newline, tks.Whitespace):
#             continue
        
#         switch_control_flow(token, control_flow)

#         if control_flow.where_seen or control_flow.having_seen:
#             if token.ttype is tks.Keyword and token.value.upper() == 'BETWEEN':
#                 between_seen = True
#             # print(token, token.ttype, type(token), control_flow.where_seen, control_flow.having_seen)
#             process_where_having(token, cond, conditions, between_seen)
#             # print('-->', str(TokenList(cond)))
            
#             if (token.ttype is tks.Keyword) and (token.value.upper() in ('AND', 'OR')) and (not between_seen):
#                 if cond:
#                     conditions.append(format_cond(cond))
#                 cond = []
#                 between_seen = False
                
#                 continue
            
#     if cond:
#         # if there only one condition
#         conditions.append(format_cond(cond))

# def format_cond(cond: list) -> Comparison:
#     if len(cond) >= 2 and cond[-1].ttype is tks.Whitespace:
#         cond = cond[:-1]
#     return Comparison(cond)

# def get_operation_type(token):
#     """
#     Determines if the function is an aggregate function.
#     """
#     WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', '<>', 'in', 'like', 'is', 'exists')
#     if token.ttype in (tks.Comparison, tks.Keyword):
#         op_name = token.value.lower()
#         if op_name and op_name in WHERE_OPS:
#             return op_name
#     return None

# def extract_operation_types(conditions: list, operator_types: set):
#     for cs in conditions:
#         ops = []
#         for c in cs:
#             if is_pwn(c):
#                 continue
#             op = get_operation_type(c)
#             if op:
#                 ops.append(op)
#         operator_types.add(' '.join(ops))

# def formating_conditions(
#         conditions: list[Comparison], 
#         aliases: dict[str, dict[str, str]],
#         schema: Schema
#     ) -> list[str]:
#     conditions_str = []
#     for cs in conditions:
#         token_list = []
#         for token in cs:
#             if isinstance(token, Identifier):
#                 token_list.append(get_full_column_name(token, aliases, schema))
#             else:
#                 token_list.append(str(token))
#         conditions_str.append(''.join(token_list))
#     return conditions_str


# # --------------------------------------------------------------------------------------------
# # Extracting Aggregations
# # --------------------------------------------------------------------------------------------

# def extract_aggregation(statement: Statement, aliases: dict[str, str], schema: Schema):
#     """
#     Extracts following information that used in the GROUP BY clause.
#     (1) Number of expressions used. 
#     (2) Number of logical calculations, functions used
#         - simple column part '<s>', 
#         - calculation part '<c>'
#         - aggregate function part '<a>'

#     Assumptions:
#     - Only table names have aliases.
#     - Subqueries can occur in WHERE and HAVING clauses.
#     - Only one set operation or subquery can occur.
#     """
#     statements = split_set_operation(statement)
#     unique_columns = set()
#     aggregation_types = set()
#     for stmt in statements:
#         extract_group_by(stmt, aliases, schema, unique_columns, aggregation_types, ControlFlow())
#     return unique_columns, aggregation_types

# def extract_group_by(
#         stmt: Statement, 
#         aliases: dict[str, str], 
#         schema: Schema, 
#         unique_columns: set, 
#         aggregation_types: set,
#         control_flow: ControlFlow
#     ):
#     if isinstance(stmt, Statement):
#         iterator = stmt.tokens
#     else:
#         iterator = stmt

#     for token in iterator:
#         if token.is_whitespace or token.ttype in (tks.Newline, tks.Whitespace):
#             continue
        
#         switch_control_flow(token, control_flow)

#         if control_flow.groupby_seen:
#             if is_pwn(token):
#                 continue
#             # Process the GROUP BY items
#             if isinstance(token, IdentifierList):
#                 # Multiple items in GROUP BY
#                 for identifier in token.get_identifiers():
#                     process_item(identifier, aliases, schema, unique_columns, aggregation_types)
#             elif isinstance(token, (Identifier, Function, Parenthesis)):
#                 # Single item in GROUP BY
#                 process_item(token, aliases, schema, unique_columns, aggregation_types)
#             else:
#                 # Handle other cases if needed
#                 pass

# # --------------------------------------------------------------------------------------------
# # Extracting Nested Queries
# # --------------------------------------------------------------------------------------------

# def extract_nested_setoperation(statement: Statement) -> int:
#     """
#     Extracts number of nested queries in the SQL statement(include set operation).
#     number of nested queries + number of set operation queries
#     Assumptions:
#     - Only table names have aliases.
#     - Subqueries can occur in WHERE and HAVING clauses.
#     - Only one set operation or subquery can occur.
#     """
#     statements = split_set_operation(statement)
#     nested = 0
#     if len(statements) > 1:
#         nested += len(statements)
#     for stmt in statements:    
#         nested = extract_nested(stmt, nested, ControlFlow())
#     return nested

# def extract_nested(stmt, nested: int, control_flow: ControlFlow):

#     if isinstance(stmt, Statement):
#         iterator = stmt.tokens
#     else:
#         iterator = stmt

#     for token in iterator:
#         # Skip comments and whitespace
#         if token.is_whitespace or token.ttype in (sqlparse.tokens.Newline,):
#             continue

#         switch_control_flow(token, control_flow)
        
#         if control_flow.select_seen or control_flow.from_seen or control_flow.join_seen or control_flow.having_seen:
#             if isinstance(token, Parenthesis):
#                 subqueries = extract_subqueries([token])
#                 nested += len(subqueries)
#         elif control_flow.where_seen:
#             # print(token, token.ttype, type(token))
#             if isinstance(token, Where) and token.value.upper() != 'WHERE':
#                 for sub_token in token.tokens:
#                     if isinstance(sub_token, Parenthesis) and is_subquery(sub_token):
#                         subqueries = extract_subqueries([sub_token])
#                         nested += len(subqueries)
#                 # nested += len(subqueries)
#             elif isinstance(token, Comparison):
#                 for sub_token in token.tokens:
#                     if isinstance(sub_token, Parenthesis) and is_subquery(sub_token):
#                         subqueries = extract_subqueries([sub_token])
#                         nested += len(subqueries)
#             elif isinstance(token, Parenthesis) and is_subquery(token):
#                 subqueries = extract_subqueries([token])
#                 if subqueries:
#                     nested += len(subqueries)
#                 else:
#                     for sub_token in token.tokens:
#                         nested = extract_nested(sub_token, nested, control_flow)
#             elif (token.ttype is tks.Keyword and token.value.upper() == 'WHERE'):
#                 pass
#             else:
#                 pass
#     return nested

# # --------------------------------------------------------------------------------------------
# # Extracting Others
# # --------------------------------------------------------------------------------------------

# def extract_others(statement: Statement, aliases: dict[str, dict[str, str]], schema: Schema) -> dict[str, set|bool]:
#     """
#     Extracts following information that used in the WHERE and HAVING clause.
#     (1) the columns that are used with DISTINCT
#     (2) the columns that are used in ORDER BY
#     (3) whether the LIMIT used

#     WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', '<>', 'in', 'like', 'is', 'exists')

#     Assumptions:
#     - Only table names have aliases.
#     - Subqueries can occur in WHERE and HAVING clauses.
#     - Only one set operation or subquery can occur.
#     """
#     # control_flow = ControlFlow()
#     others = {'distinct': set(), 'order by': set(), 'limit': False}
#     statements = split_set_operation(statement)
#     for stmt in statements:
#         others = extract_distinct_orderby_limit(stmt, aliases, schema, others)
#     return others

# def extract_distinct_orderby_limit(
#         stmt: Statement, 
#         aliases: dict[str, dict[str, str]],
#         schema: Schema,
#         others: dict[str, set|bool],
#     ):
#     distinct_used = False
#     order_by_used = False
#     orderby_tokens = []
#     ord = []
#     for token in stmt.flatten():
#         if token.ttype is tks.Keyword:
#             if token.value.upper() == 'DISTINCT':
#                 distinct_used = True
#                 continue
#             if token.value.upper() == 'ORDER BY':
#                 ord = []
#                 order_by_used = True
#                 continue
#             if token.value.upper() == 'LIMIT':
#                 order_by_used = False
#                 others['limit'] = True
#                 continue
        
#         if distinct_used:
#             if is_pwn(token):
#                 continue
#             column_name = get_full_column_name(Identifier([token]), aliases, schema)
#             others['distinct'].add(column_name)
#             distinct_used = False

#         if order_by_used:            
#             if token.ttype is tks.Punctuation and token.value == ',':
#                 if sqlparse.parse(str(TokenList(ord))):
#                     orderby_tokens.append(get_orderby_expression(ord, aliases, schema))
#                 ord = []
#             else:
#                 ord.append(token)

#     if ord and sqlparse.parse(str(TokenList(ord))):
#         orderby_tokens.append(get_orderby_expression(ord, aliases, schema))
#     others['order by'].update(orderby_tokens)
#     return others

# def get_orderby_expression(ord: list, aliases: dict[str, dict[str, str]], schema: Schema) -> str:
#     expression = []
#     for tkn in sqlparse.parse(str(TokenList(ord)))[0].tokens:
#         _ = extract_columns_from_expression(tkn, aliases, schema, expression)
#     return str(''.join(expression)).strip().lower()

# if __name__ == '__main__':
#     sql = 'SELECT t1.a, b AS bb, c FROM table1 t1 WHERE a > 10 AND b < 20'
#     statement = sqlparse.parse(sql)[0]
#     aliases = extract_aliases(statement)
#     schema = Schema(
#         schema={
#             'table1': ['a', 'b', 'c']
#         }
#     )
#     print(aliases)
#     print(extract_selection(statement, aliases, Schema()))
#     print(extract_condition(statement, aliases, Schema()))
#     print(extract_aggregation(statement, aliases, Schema()))
#     print(extract_nested_setoperation(statement))
#     print(extract_others(statement, aliases, Schema()))
