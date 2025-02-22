
from collections import defaultdict
import sqlglot
from sqlglot import expressions as exp
from typing import Set, Tuple, Dict, Any, List
from src.db_utils import format_raw_sql
# OPERATOR_MAP = {
#     exp.EQ: "=",
#     exp.GT: ">",
#     exp.GTE: ">=",
#     exp.LT: "<",
#     exp.LTE: "<=",
#     exp.NEQ: "<>",
#     exp.Is: "is",
#     exp.Like: "like",
#     # Add more if needed
# }
from copy import deepcopy

STRING_TYPE = '[PLACEHOLDER-TYPE:STRING]'.lower()
NUMERIC_TYPE = '[PLACEHOLDER-TYPE:NUMERIC]'.lower()
SUBQUERY = '[PLACEHOLDER-SUBQUERY]'.lower()

def flatten_conds(xs):
    for x in xs:
        if isinstance(x[0], str) and isinstance(x[1], exp.Expression):
            yield x
        else:
            yield from flatten_conds(x)

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
    
    def check_column_exist(self, column: str, tables: list[str]=[]):
        if column == '*':
            return True
        
        if tables:
            subset_schema = {k: v for k,v in self.schema.items() if k in tables}
        else:
            subset_schema = self.schema

        for cols in subset_schema.values():
            if column.lower() in cols:
                return True
        return False
    
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

    if isinstance(node, exp.CTE):
        # Common Table Expression: extract subqueries from its inner SELECT
        inner = node.args.get("this")
        print(repr(inner))
        return inner if inner else []

    if isinstance(node, exp.Select):
        # Start with this SELECT node
        results = [node]

        # Check FROM clause for subqueries
        from_clause = node.args.get("from")
        if from_clause and isinstance(from_clause, exp.From):
            # Main source in FROM
            main_source = from_clause.this
            if main_source:
                subquery = get_subqueries(main_source)
                results += subquery

        join_clause = node.args.get("joins", [])
        # Joins in FROM
        for join_expr in join_clause:
            right_source = join_expr.args.get("this")
            if right_source:
                subquery = get_subqueries(right_source)
                results += subquery

        # Check WHERE and HAVING for nested subqueries
        for clause_name in ("where", "having"):
            clause = node.args.get(clause_name)
            if clause:
                # find_all Subqueries inside the clause
                for sub in clause.find_all(exp.Subquery):
                    subquery = get_subqueries(sub)
                    results += subquery

        # check CTEs
        ctes = node.args.get('with')
        if ctes:
            for cte in ctes:
                subquery = get_subqueries(cte.this)
                results += subquery

        # Check SELECT expressions for nested subqueries
        for e in node.select():
            for sub in e.find_all(exp.Subquery):
                subquery = get_subqueries(sub)
                results += subquery

        # Find subqueries in EXISTS
        for e in node.find_all(exp.Exists):
            subquery = [e.args.get('this')]
            results += subquery

        return results

    # If it's none of the above (e.g., just a Table or Column), no subqueries here
    return []

def extract_aliases(node: exp.Expression) -> Dict[str, Dict[str, str]]:
    """
    Extract table and column aliases from the given query expression.
    This function handles set operations by recursively extracting aliases
    from both sides of the set operation.
    """
    
    def merge_aliases(target: Dict[str, set], source: Dict[str, set]):
        for key, val in source.items():
            target[key].update(val)

    alias_mapping = {'table': {}, 'column': {}}
    subqueries = get_subqueries(node)
    for s in subqueries:
        x = _extract_aliases_from_select(s)
        merge_aliases(alias_mapping, x)

    return alias_mapping

def _extract_aliases_from_select(query: exp.Select) -> Dict[str, Dict[str, str]]:
    alias_mapping = {'table': {}, 'column': {}}

    # Extract table aliases from FROM clause
    from_clause = query.args.get('from')
    if from_clause and isinstance(from_clause, exp.From):
        _handle_table_or_subquery(from_clause.this, alias_mapping['table'])

    # Extract table aliases from JOINS
    joins = query.args.get('joins')
    if joins:
        for join_expr in joins:
            right_source = join_expr.args.get('this')  # 'this' is the table in the Join node
            if right_source:
                _handle_table_or_subquery(right_source, alias_mapping['table'])

    # Extract column aliases from SELECT clause and where does alias from (table)
    for alias in query.find_all(exp.Alias):
        alias_mapping['column'][alias.alias] = str(alias.this)
        # TODO: add where does alias from (table)
    return alias_mapping

def _handle_table_or_subquery(expr: exp.Expression, table_alias_map: Dict[str, str]):
    if isinstance(expr, exp.Table):
        table_name = expr.name
        alias_expr = expr.args.get("alias")
        alias = None
        if alias_expr and isinstance(alias_expr, exp.TableAlias):
            alias = alias_expr.this.name
        else:
            alias_id = expr.alias
            if alias_id:
                alias = alias_id

        if alias:
            table_alias_map[alias] = table_name.lower()
        else:
            table_alias_map[table_name] = table_name.lower()

    elif isinstance(expr, exp.Subquery):
        alias_expr = expr.args.get("alias")
        if alias_expr and isinstance(alias_expr, exp.TableAlias):
            sub_alias = alias_expr.this.name
            inner_select = expr.args.get("this")
            if inner_select and isinstance(inner_select, exp.Select):
                table_alias_map[sub_alias] = str(inner_select)
            else:
                table_alias_map[sub_alias] = str(expr)
        else:
            # need to add a custom name for the subquery
            n_sub = len([k for k in table_alias_map if 'subquery' in k])
            sub_alias = f"q{n_sub}"
            table_alias_map[sub_alias] = str(expr)

    elif isinstance(expr, exp.Select):
        alias_expr = expr.args.get("alias")
        if alias_expr and isinstance(alias_expr, exp.TableAlias):
            sub_alias = alias_expr.this.name
            table_alias_map[sub_alias] = str(expr)

def extract_from_join(
        query: exp.Select, 
        aliases: Dict[str, Dict[str, str]], 
        schema: Schema,
        anonymize_literal: bool = True) -> Set[str]:
    query = deepcopy(query)
    tables = set()
    from_clause = query.args.get('from')
    if from_clause and isinstance(from_clause, exp.From):
        name, expr = _format_expression(from_clause.this, aliases, schema, anonymize_literal)
        tables.add((name, exp.Table(this=expr), 'from'))

    join_clause = query.args.get('joins', [])
    if join_clause:
        for join_expr in join_clause:
            left_expr = join_expr.args.get('this')
            left_name, new_left_expr = _format_expression(left_expr, aliases, schema, anonymize_literal=False)
            join_expr.args['this'] = new_left_expr
            table = exp.Table(this=left_expr)
            tables.add((left_name, table, 'join'))

    return tables

def extract_selection(
        query: exp.Select, 
        aliases: Dict[str, Dict[str, str]], 
        schema: Schema,
        anonymize_literal: bool = True
    ) -> Tuple[Set[str], Set[Tuple[str, str]]]:

    query = deepcopy(query)
    unique_columns = set()
    selection_asts = set()

    for select_exp in query.select():
        expr_str, expr = _format_expression(select_exp, aliases, schema, anonymize_literal)
        tag = _determine_tag(expr)
        selection_asts.add((expr_str, expr, tag))
        columns = _extract_columns_from_expression(expr, aliases, schema)
        unique_columns.update(columns)

    return unique_columns, selection_asts


def _determine_tag(expr: exp.Expression) -> str:
    # All functions or binary operations => <func>, <select>, <subquery>
    if isinstance(expr, exp.Column):
        return '<select>'
    if isinstance(expr, (exp.Func, exp.Binary, exp.Condition)):
        return '<func>'
    if isinstance(expr, exp.Subquery):
        return '<subquery>'
    # Check nested functions or binaries inside
    if any(isinstance(x, exp.Func) for x in expr.find_all(exp.Func)):
        return '<func>'
    if any(isinstance(x, exp.Binary) for x in expr.find_all(exp.Binary)):
        return '<func>'
    # If it's just columns, <s>, else default <s>
    if all(isinstance(x, exp.Column) for x in expr.find_all(exp.Column)):
        return '<select>'
    return '<select>'

def _format_expression(
        expr: exp.Expression, 
        aliases: Dict[str, Dict[str, str]], 
        schema: Schema, 
        anonymize_literal: bool = False,
        remove_alias: bool = True,
        lower_case: bool = True  # used for literal case
    ) -> Tuple[str, exp.Expression]:
    
    if remove_alias and isinstance(expr, (exp.Alias, exp.TableAlias, exp.Ordered)):
        return _format_expression(
            expr.this, 
            aliases, 
            schema, 
            remove_alias=True,
            anonymize_literal=anonymize_literal
        )

    if isinstance(expr, (exp.Column, exp.Star)):
        if isinstance(expr.args.get('this'), exp.Subquery):
            name, new_expr = _format_expression(
                expr.args.get('this'), 
                aliases, 
                schema, 
                anonymize_literal
            )
        name = _get_full_column_name(expr, aliases, schema)
        # ----- table ------
        if expr.args.get('table') and expr.table in aliases['table']:
            # 필요없을지도?
            # 1st condition: check if there is table
            # 2nd condition: check alias table
            original_table = aliases['table'][expr.table]
            quoted = expr.args['table'].quoted
            if 'select' in original_table.lower():
                # subquery table
                if original_table.startswith('(') and original_table.endswith(')'):
                    subquery_table = original_table.lstrip('(').rstrip(')')
                else:
                    subquery_table = f"{original_table}"
                expr.args['table'] = exp.Subquery(this=sqlglot.parse_one(subquery_table))
            else:
                expr.args['table'] = exp.Identifier(this=original_table, quoted=quoted)
            
        elif not expr.args.get('table') and name != '__all__':
            if 'select' in name.lower():
                # subquery column
                subquery_col = name.lstrip('(').rstrip(')')
                expr.args['this'] = exp.Subquery(this=sqlglot.parse_one(subquery_col))
            else:
                if len(name.split('.')) == 1:
                    # only column name
                    table = 'None'
                else:
                    table = name.split('.')[0].lstrip('__')
                expr.args['table'] = exp.Identifier(this=table, quoted=False)

        # ----- column ------
        if expr.name.lower() in aliases['column'] and 'select' in aliases['column'][expr.name.lower()]:
            alias_expr_str = aliases['column'][expr.name.lower()]
            # # replace the table alias to avoid recursive alias
            # if expr.args.get('table'):
            #     original_table = aliases['table'][expr.args['table'].name.lower()]
            #     alias_expr_str = alias_expr_str.replace(expr.args['table'].name, original_table)
            #     # ** what if the table is subquery ?
            return _format_expression(
                sqlglot.parse_one(alias_expr_str), 
                aliases, 
                schema, 
                anonymize_literal
            )
        
        if name != '__all__' and not isinstance(expr.this, exp.Subquery) and lower_case:
            # change column name as lower case
            expr.args['this'] = exp.Identifier(
                this=expr.args['this'].name.lower(),
                quoted=expr.args['this'].quoted
            )

        return name, expr
        
    if isinstance(expr, exp.Func):
        func_name = type(expr).__name__
        # need to run recursive if have multiple functions
        if func_name.lower() == 'anonymous':
            # `this` is the function name: e.g., strftime
            func_name = expr.args.get('this').lower()
        elif func_name.lower() == 'substr':
            func_name = 'substring'
        else:
            func_name = func_name.lower()

        args = [k for k in expr.args.keys() if expr.args.get(k)]
        args_list = []
        for arg in args:
            arg_expr = expr.args.get(arg)
            if isinstance(arg_expr, exp.Expression):
                name, new_expr = _format_expression(
                    arg_expr, 
                    aliases, 
                    schema, 
                    anonymize_literal=anonymize_literal
                )
                expr.args[arg] = new_expr
                args_list.append(name)
            elif isinstance(arg_expr, list):
                # handle case: IF, Anonymous Function arguments(e.g., strftime)
                for sub_arg_expr in arg_expr:
                    name, new_expr = _format_expression(
                        sub_arg_expr, 
                        aliases, 
                        schema, 
                        anonymize_literal=anonymize_literal
                    )
                    sub_arg_expr = new_expr
                    args_list.append(name)

        return f"{func_name}({', '.join(args_list)})", expr

    if isinstance(expr, exp.Distinct):
        # Only one case that appears with Function e.g., COUNT(DISTINCT col)
        if expr.args.get('expression'):
            name, new_expr = _format_expression(
                expr.expressions[0], 
                aliases, 
                schema, 
                anonymize_literal
            )
            expr.expressions[0] = new_expr
        else:
            name = ''

        return 'DISTINCT '.lower() + name, expr
    
    if isinstance(expr, exp.Paren):
        name, new_expr = _format_expression(
            expr.args.get('this'), 
            aliases, 
            schema, 
            anonymize_literal
        )
        expr.args['this'] = new_expr
        return '(' + name + ')', expr
    
    if isinstance(expr, exp.In):
        return _format_exp_in(expr, aliases, schema, anonymize_literal)
    
    if isinstance(expr, exp.Between):
        return _format_exp_between(expr, aliases, schema, anonymize_literal)

    if isinstance(expr, exp.Not):
        return _format_exp_not(expr, aliases, schema, anonymize_literal)
    
    if isinstance(expr, exp.Binary):
        # Binary operators like EQ, LT, GT, Div, Mul etc.
        left_expr = expr.args.get('this')
        left_name, new_expr = _format_expression(left_expr, aliases, schema, anonymize_literal)
        expr.args['this'] = new_expr
        
        right_expr = expr.args.get('expression')
        right_name, new_expr = _format_expression(right_expr, aliases, schema, anonymize_literal)
        expr.args['expression'] = new_expr
        return f"{left_name} {expr.key.lower()} {right_name}", expr

    if isinstance(expr, exp.Exists):
        assert isinstance(expr.args.get('this'), exp.Select), f"Exists should have a subquery {expr.args.get('this')}"
        name, new_expr = _format_expression(
            expr.args.get('this'), 
            aliases, 
            schema, 
            anonymize_literal
        )
        expr.args['this'] = new_expr
        return f"EXISTS".lower() + f"({name})", expr

    if isinstance(expr, exp.Window):
        # this, partition_by(list), order(list), over
        window_name, new_expr = _format_expression(
            expr.args.get('this'), 
            aliases, 
            schema, 
            anonymize_literal
        )
        expr.args['this'] = new_expr
        # partition_by
        partitions = expr.args.get('partition_by')
        if partitions:
            partitions_str = []
            partitions_expr = []
            for partition in partitions:
                name, new_expr = _format_expression(
                    partition, 
                    aliases, 
                    schema, 
                    anonymize_literal
                )
                partitions_str.append(name)
                partitions_expr.append(new_expr)
            expr.args['partition_by'] = partitions_expr
            
            key = f'(PARTITION BY {", ".join(partitions_str)})'
        # order
        orders = expr.args.get('order')
        if orders:
            orders_str = []
            orders_expr = []
            for order in orders:
                name, new_expr = _format_expression(
                    order, 
                    aliases, 
                    schema, 
                    anonymize_literal
                )
                orders_str.append(name)
                orders_expr.append(new_expr)
            expr.args['order'] = exp.Order(expressions=orders_expr)
            key = f"(ORDER BY {', '.join(orders_str)})"

        return (f"{window_name} OVER " + key).lower(), expr

    if isinstance(expr, exp.Identifier):
        if lower_case:
            expr = exp.Identifier(this=expr.name.lower(), quoted=expr.quoted)
        return expr.name.lower(), expr

    if isinstance(expr, exp.Table):
        if lower_case:
            if expr.args.get('alias'):
                if remove_alias:
                    expr.args.pop('alias')
                else:
                    expr.args['alias'] = exp.TableAlias(
                            this=exp.Identifier(
                                this=expr.args['alias'].name.lower(), 
                                quoted=expr.args['alias'].this.quoted
                            )
                        )
            expr.args['this'] = exp.Identifier(
                this=expr.args['this'].name.lower(), 
                quoted=expr.args['this'].quoted   
            )
        return str(expr), expr # expr.name.lower(), expr

    if isinstance(expr, (exp.Subquery, exp.Select)):
        args = [(k, v) for k, v in expr.args.items() if v]
        
        # if isinstance(expr, exp.Subquery):
        #     s2a = {v.lower(): k for k, v in aliases['table'].items()}
        #     table_alias = s2a.get(str(expr.this).lower())
        #     if table_alias:
        #         return f"{table_alias}".lower(), expr
        # try: 
        expr_str = str(expr)
        # except:
        #     expr_str = _format_select(expr)
        for arg, sub_expr in args:
            if arg in ('from'):
                # FROM / alias clause
                name, new_expr = _format_expression(
                    sub_expr.this,
                    aliases,
                    schema,
                    anonymize_literal,
                )
                expr.args['from'] = exp.From(this=new_expr)
            elif isinstance(sub_expr, list):
                new_sub_expr = []
                for sub in sub_expr:
                    name, new_expr = _format_expression(
                        sub, 
                        aliases, 
                        schema, 
                        anonymize_literal
                    )
                    new_sub_expr.append(new_expr)
                expr.args[arg] = new_sub_expr
            else:
                # check the follow up arguments for sub_expr
                sub_args = [(k, v) for k, v in sub_expr.args.items() if v]
                for sub_sub_args, sub_sub_expr in sub_args:
                    if isinstance(sub_sub_expr, list):
                        new_sub_sub_expr = []
                        for sub_sub in sub_sub_expr:
                            name, new_expr = _format_expression(
                                sub_sub, 
                                aliases, 
                                schema, 
                                anonymize_literal
                            )
                            new_sub_sub_expr.append(new_expr)
                        sub_expr.args[sub_sub_args] = new_sub_sub_expr
                    else:
                        name, new_expr = _format_expression(
                            sub_sub_expr, 
                            aliases, 
                            schema, 
                            anonymize_literal
                        )
                        sub_expr.args[sub_sub_args] = new_expr
                expr.args[arg] = sub_expr

        return expr_str, expr # SUBQUERY, expr
    
    if isinstance(expr, exp.Neg):
        # Unary operator
        # e.g., -1
        # Neg(this=Literal(this=1, is_string=False))
        sub_expr = expr.args.get('this')
        sub_name, new_expr = _format_expression(sub_expr, aliases, schema, anonymize_literal=anonymize_literal)
        expr.args['this'] = new_expr
        return f"-{sub_name}", expr

    if isinstance(expr, exp.Literal):
        if anonymize_literal:
            if expr.is_string:
                return STRING_TYPE, exp.Literal.string(STRING_TYPE)
            else:
                return NUMERIC_TYPE, exp.Literal.number(NUMERIC_TYPE)
        else:
            if lower_case:
                return str(expr).lower(), expr
            else:
                return str(expr), expr

    if isinstance(expr, (exp.DataType, exp.Null, exp.Boolean)):
        if lower_case:
            return str(expr).lower(), expr
        else:
            return str(expr), expr
    
    return None, expr

def _format_select(expr: exp.Select):
    args = [
        'kind', 'hint', 'distinct', 'expressions', 'limit', 'from', 'joins', 
        'where', 'group', 'having', 'order']
    s = 'SELECT '
    s += ', '.join([str(x) for x in expr.args.get('expressions')])
    s += str(expr.args.get('from'))
    if expr.args.get('joins'):
        s += ' '.join([str(x) for x in expr.args.get('joins')])
    if expr.args.get('where'):
        s += str(expr.args.get('where'))
    if expr.args.get('group'):
        s += str(expr.args.get('group'))
    if expr.args.get('having'):
        s += str(expr.args.get('having'))
    if expr.args.get('order'):
        s += str(expr.args.get('order'))
    if expr.args.get('limit'):
        s += str(expr.args.get('limit'))
    return s

def _format_exp_in(
        expr: exp.In, 
        aliases: Dict[str, Dict[str, str]], 
        schema: Schema, 
        anonymize_literal: bool=False) -> Tuple[str, exp.In]:
    """
    # this, expressions | this, query
    
    case1: col IN (...)
    right expression is a list of literals: do not make it lower case
    e.g., 
        In(
        this=Column(
            this=Identifier(this=job_desc, quoted=False),
            table=Identifier(this=T2, quoted=False)),
        expressions=[
            Literal(this=Editor, is_string=True),
            Literal(this=Designer, is_string=True)])

    case2: col IN (SELECT ...)
    """
    args = [k for k in expr.args.keys() if expr.args.get(k)]
    left_expr = expr.args.get('this')
    left_name, new_expr = _format_expression(left_expr, aliases, schema, anonymize_literal)
    expr.args['this'] = new_expr
    if 'expressions' in args:
        # case1: col IN (...)
        right_exprs = expr.args.get('expressions')
        right_exprs_str = []
        new_right_exprs = []
        for right_expr in right_exprs:
            right_name, new_expr = _format_expression(right_expr, aliases, schema, anonymize_literal, lower_case=False)
            right_exprs_str.append(right_name)
            new_right_exprs.append(new_expr)

        expr.args['expressions'] = new_right_exprs
        return f"{left_name} {expr.key.lower()} ({', '.join(right_exprs_str)})", expr
    else:
        # case2: col IN (SELECT ...)
        right_expr = expr.args.get('query')
        right_name, new_expr = _format_expression(right_expr, aliases, schema, anonymize_literal)
        expr.args['query'] = new_expr
        return f"{left_name} {expr.key.lower()} ({right_name})", expr

def _format_exp_between(
        expr: exp.Between, 
        aliases: Dict[str, Dict[str, str]], 
        schema: Schema, 
        anonymize_literal: bool=False) -> Tuple[str, exp.In]:
    left = expr.args.get('this')
    col, col_expr = _format_expression(
        left, aliases, schema, anonymize_literal)
    low, low_expr = _format_expression(
        expr.args.get('low'), aliases, schema, anonymize_literal)
    high, high_expr = _format_expression(
        expr.args.get('high'), aliases, schema, anonymize_literal)
    expr.args['this'] = col_expr
    expr.args['low'] = low_expr
    expr.args['high'] = high_expr
    return f"{col} between {low} and {high}", expr

def _format_exp_not(
        expr: exp.Not,
        aliases: Dict[str, Dict[str, str]],
        schema: Schema,
        anonymize_literal: bool=False) -> Tuple[str, exp.Not]:
    child = expr.args.get('this')
    if child:
        child_name, child_expr = _format_expression(
            child, aliases, schema, anonymize_literal)
    else:
        child_name = str(expr)
        child_expr = expr
    expr.args['this'] = child_expr

    child_name = child_name.lower()

    is_case1 = False
    key = None
    if f' {exp.Between.key.lower()} ' in child_name:
        # case2: Replace first occurrence of ' between ' with ' not between '
        key = f' {exp.Between.key} '
        to = f' {exp.Not.key} {exp.Between.key} '.lower()
    elif f' {exp.In.key.lower()} ' in child_name:
        # case3: Replace first occurrence of ' in ' with ' not in '
        key = f' {exp.In.key} '.lower()
        to = f' {exp.Not.key} {exp.In.key} '.lower()
    elif f' {exp.Like.key.lower()} ' in child_name:
        # case3: Replace first occurrence of ' like ' with ' not like '
        key = f' {exp.Like.key} '.lower()
        to = f' {exp.Not.key} {exp.Like.key} '.lower()
    elif f' {exp.Is.key.lower()} ' in child_name:
        # case4: Replace first occurrence of ' is ' with ' is not '
        key = f' {exp.Is.key} '.lower()
        to = f' {exp.Is.key} {exp.Not.key} '.lower()
    else:
        is_case1 = True

    if is_case1:
        # case 1
        child_name = f"{exp.Not.key.lower()} {child_name}"
    else:
        # other cases
        idx = child_name.index(key)
        child_name = child_name[:idx] + to + child_name[idx + len(key):]

    return child_name, expr

def _get_full_column_name(col: exp.Column, aliases: Dict[str, Dict[str, str]], schema: Schema) -> str:
    column_name = col.name.lower()
    # if (not schema.check_column_exist(column_name)) and (column_name not in aliases['column']):
    #     assert False, f"Column {column_name} not found in schema"
    
    # If the column is *, map directly to __all__
    if column_name == '*':
        return schema.idMap['*']
    
    if column_name == '':
        # means it is a subquery column
        # TODO: need to find upper level table name for this subquery column
        s2a = {v.lower(): k for k, v in aliases['column'].items()}
        col_alias = s2a.get(str(col.this).lower())
        if col_alias:
            return f'__{col_alias}__'
        else:
            return f"__{str(col.this)}__"

    if column_name in aliases['column'] and 'select' in aliases['column'][column_name].lower():
        # means subquery column is in the aliases table
        # return SUBQUERY # f"__{column_name}__"
        subquery_col = aliases['column'][column_name]
        # if 'select' in subquery_col.lower() and not subquery_col.startswith('('):
        #     subquery_col = f"({subquery_col}"
        # if 'select' in subquery_col.lower() and not subquery_col.endswith(')'):
        #     subquery_col = f"{subquery_col})"
        return subquery_col

    table_alias = col.table if col.table else None
    # None: means either table has no alias or table is not present in the node
    if table_alias:
        if table_alias in aliases['table']:
            # table alias need to be replaced with the original table name
            real_table_name = aliases['table'][table_alias]
        else:
            # table alias is the original table name
            real_table_name = table_alias
    else:
        # find possible tables in the schema
        possible_tables = [v for v in aliases['table'].values() if 'select' not in v.lower()]
        real_table_name = schema.get_table_name(column_name, possible_tables)
        if not real_table_name:
            # possible in the subquery column alias
            possible_tables = [k for k, v in aliases['table'].items() if 'select' in v.lower()]
            for table_name in possible_tables:
                alias = [x.alias.lower() for x in sqlglot.parse_one(aliases['table'][table_name]).find_all(exp.Alias)]
                if column_name in alias:
                    real_table_name = f'subquery_{table_name}'
                    break

    if real_table_name:
        if 'select' in real_table_name.lower():
            key = f"({real_table_name.lower()}).{column_name}"
        else:
            key = f"{real_table_name.lower()}.{column_name}"
    else:
        key = column_name
    
    return schema.idMap.get(key, f"__{key}__")

def _extract_columns_from_expression(
        expr: exp.Expression, 
        aliases: Dict[str, Dict[str, str]], 
        schema: Schema) -> List[str]:
    columns = set()
    for col in expr.find_all(exp.Column, exp.Star):
        full_col_name = _get_full_column_name(col, aliases, schema)
        columns.update([full_col_name])
    
    return columns

def extract_condition(
        query: exp.Select, 
        aliases: Dict[str, Dict[str, str]], 
        schema: Schema, 
        anonymize_literal:bool = True) -> Tuple[Set[str], Set[str]]:
    query = deepcopy(query)
    condition_asts = set()
    operator_types = set()

    for clause_name in ("where", "having"):
        clause = query.args.get(clause_name)
        if clause:
            # clause is WHERE or HAVING. We only want the conditions inside.
            # clause.this is the actual condition expression (e.g., AND/OR tree).
            _extract_conditions(
                clause.this, aliases, schema, operator_types, condition_asts, anonymize_literal
            )
            
            assert len(condition_asts) > 0, f"Failed to extract conditions from {clause_name} clause {len(condition_asts)}"
    return condition_asts, operator_types

def _extract_conditions(
    expr: exp.Expression,
    aliases: Dict[str, Dict[str, str]],
    schema: Schema,
    operator_types: Set[str],
    condition_asts: Set[Tuple[str, exp.Expression, str]],
    anonymize_literal: bool = False,
    ) -> List[Tuple[str, exp.Expression]]:
    """
    Recursively extract a condition string from an expression in WHERE/HAVING clauses.
    Also populates operator_types with encountered operators.

    Returns:
        A single string representing the condition (which may include nested conditions).

    Logic:
    - AND/OR: combine left and right conditions with parentheses, record 'and'/'or'
    - NOT: prepend 'not ' to the child's condition. If the child contains ' between ',
      replace it with ' not between ' to form a proper condition.
    - BETWEEN: return the condition as 'col between val1 and val2', record 'between'
    - Binary operators (=, >, <, etc.): return 'left op right', record the operator
    - Leaf nodes: directly return their string representation
    """
    if isinstance(expr, exp.Paren):
        return _extract_conditions(
            expr.args.get('this'), 
            aliases, 
            schema, 
            operator_types, 
            condition_asts,
            anonymize_literal
        )
    elif isinstance(expr, (exp.And, exp.Or)):
        # Logical connectors: AND / OR
        # conditions = []
        left = expr.args.get('this')
        right = expr.args.get('expression')
        if left:
            _extract_conditions(
                left, aliases, schema, operator_types, condition_asts, anonymize_literal)
        if right:
            _extract_conditions(right, aliases, schema, operator_types, condition_asts, anonymize_literal)
    else:
        cond_name, expr = _format_expression(
            expr, aliases, schema, remove_alias=True, anonymize_literal=anonymize_literal)

        op = expr.key.lower()

        condition_asts.add((cond_name, expr, op))
        if isinstance(expr, (exp.Binary, exp.Between, exp.In, exp.Not)):
            operator_types.add(op)

def extract_aggregation(
        query: exp.Select, 
        aliases: Dict[str, Dict[str, str]], 
        schema: Schema,
        anonymize_literal: bool=True) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    query = deepcopy(query)
    unique_columns = set()
    aggregation_asts = set()

    group = query.args.get('group')
    if group:
        for g_expr in group:
            expr_str, expr = _format_expression(
                g_expr, aliases, schema, anonymize_literal)
            tag = _determine_tag(expr)
            aggregation_asts.add((expr_str, expr, tag))
            columns = _extract_columns_from_expression(expr, aliases, schema)
            unique_columns.update(columns)

    return unique_columns, aggregation_asts

def extract_orderby(
        query: exp.Select,
        aliases: Dict[str, Dict[str, str]],
        schema: Schema,
        anonymize_literal: bool = True) -> Set[str]:
    query = deepcopy(query)
    unique_columns = set()
    otherby_asts = set()

    order = query.args.get('order')
    if order and isinstance(order, exp.Order):
        # order_node.expressions is a list of Ordered expressions
        for order_expr in order.expressions:
            # ordered_expr.this is the actual expression being ordered
            expr_str, expr = _format_expression(
                order_expr.this, aliases, schema, anonymize_literal
            )
            tag = _determine_tag(expr)
            otherby_asts.add((expr_str, expr, tag))
            columns = _extract_columns_from_expression(expr, aliases, schema)
            unique_columns.update(columns)

    return unique_columns, otherby_asts

def extract_others(query: exp.Select) -> Dict[str, bool]:
    """
    Extracts:
    1. whetehr DISTINCT is used
    2. whether LIMIT is used
    """
    others = {'distinct': False, 'limit': False}

    # Check for DISTINCT
    distinct_node = list(query.find_all(exp.Distinct))
    if len(distinct_node) > 0:
        others['distinct'] = True

    # Check for LIMIT
    limit_node = list(query.find_all(exp.Limit))
    if len(limit_node) > 0:
        others['limit'] = True

    return others

def extract_all(sql: str, schema: Schema) -> Tuple[Set[str], Set[Tuple[str, str]], Set[str], Set[str], Set[Tuple[str, str]], Dict[str, Any]]:
    """
    Extract all components from a SELECT query.
    """
    sql = format_raw_sql(sql)
    parsed_query = sqlglot.parse_one(sql)
    aliases = extract_aliases(parsed_query)
    subqueries = get_subqueries(parsed_query)

    results = defaultdict(set)
    results['aliases'] = aliases  # number of table used in the query(from + joins)
    results['distinct'] = False
    results['limit'] = False
    nested = len(subqueries)
    formatted_subqueries = []

    for query in subqueries:
        query = deepcopy(query)
        tables = extract_from_join(query, aliases, schema)
        sel_cols, sel_asts  = extract_selection(query, aliases, schema)
        cond_asts, op_types = extract_condition(query, aliases, schema)
        agg_cols, agg_asts  = extract_aggregation(query, aliases, schema)
        orderby_cols, orderby_asts = extract_orderby(query, aliases, schema)
        others = extract_others(query)
        
        results['table_asts'].update(tables)
        results['sel'].update(sel_cols)
        results['sel_asts'].update(sel_asts)
        results['cond_asts'].update(cond_asts)
        results['op_types'].update(op_types)
        results['agg'].update(agg_cols)
        results['agg_asts'].update(agg_asts)
        results['orderby'].update(orderby_cols)
        results['orderby_asts'].update(orderby_asts)
        results['distinct'] |= others['distinct']
        results['limit'] |= others['limit']
        *_, new_query = _format_expression(query, aliases, schema, remove_alias=False)
        formatted_subqueries.append(new_query)
    
    results['nested'] = nested
    results['subqueries'] = formatted_subqueries

    return results

if __name__ == "__main__":
    schema = Schema(schema={
        'state': {'StateCode': 'text', 'State': 'text', 'Region': 'text'},
        'callcenterlogs': {'Date received': 'text',
        'Complaint ID': 'text',
        'rand client': 'text',
        'phonefinal': 'text',
        'vru+line': 'text',
        'call_id': 'text',
        'priority': 'text',
        'type': 'text',
        'outcome': 'text',
        'server': 'text',
        'ser_start': 'text',
        'ser_exit': 'text',
        'ser_time': 'text'},
        'client': {'client_id': 'text',
        'sex': 'text',
        'day': 'text',
        'month': 'text',
        'year': 'text',
        'age': 'text',
        'social': 'text',
        'first': 'text',
        'middle': 'text',
        'last': 'text',
        'phone': 'text',
        'email': 'text',
        'address_1': 'text',
        'address_2': 'text',
        'city': 'text',
        'state': 'text',
        'zipcode': 'text',
        'district_id': 'text'},
        'district': {'district_id': 'text',
        'city': 'text',
        'state_abbrev': 'text',
        'division': 'text'},
        'events': {'Date received': 'date',
        'Product': 'date',
        'Sub-product': 'date',
        'Issue': 'date',
        'Sub-issue': 'date',
        'Consumer complaint narrative': 'date',
        'Tags': 'date',
        'Consumer consent provided?': 'date',
        'Submitted via': 'date',
        'Date sent to company': 'date',
        'Company response to consumer': 'date',
        'Timely response?': 'date',
        'Consumer disputed?': 'date',
        'Complaint ID': 'date',
        'Client_ID': 'date'},
        'reviews': {'Date': 'text',
        'Stars': 'text',
        'Reviews': 'text',
        'Product': 'text',
        'district_id': 'text'}})
    
    sql1 = "SELECT * FROM client WHERE age > 30"
    results = extract_all(sql1, schema)
    