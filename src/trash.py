# import sqlparse
# from sqlparse.sql import (
#     Token, TokenList, IdentifierList, Identifier, Function, Statement, Parenthesis, Operation,
#     Where, Comparison
# )
# import sqlparse.tokens as tks
# from typing import Callable
# from pydantic import BaseModel, Field
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




# ==================================================================== eval_complexity.py

# from tqdm import tqdm
# from pathlib import Path
# from .process_sql import get_schema, Schema
# from .parsing_sql import extract_aliases, extract_selection, extract_condition, extract_aggregation, extract_nested_setoperation, extract_others
# import json
# import sqlparse
# import numpy as np
# import pandas as pd

# proj_path = Path(__file__).resolve().parents[1]

# def partial_match(gold_set, predict_set):
#     intersection = gold_set.intersection(predict_set)
#     union = gold_set.union(predict_set)
    
#     # IoU
#     iou = len(intersection) / len(union) if union else 0
    
#     # Precision
#     precision = len(intersection) / len(predict_set) if predict_set else 0
    
#     # Recall
#     recall = len(intersection) / len(gold_set) if gold_set else 0
    
#     # F1-Score
#     f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    
#     return iou, precision, recall, f1_score

# def derive_final_score(gold_sets: list, predict_sets: list):
#     # Calculate F1-score for each aspect
#     empty_gold = [False if g else True for g in gold_sets]
#     empty_pred = [False if p else True for p in predict_sets]
#     if all(empty_gold) and all(empty_pred):
#         return 1.0
#     elif all(empty_gold) and not all(empty_pred):
#         # If all gold sets are empty, return None
#         return 0.0
#     f1_scores = []
#     for i, (gold, pred) in enumerate(zip(gold_sets, predict_sets)):
#         if not empty_gold[i]:
#             *_, f1_score = partial_match(gold, pred)
#             f1_scores.append(f1_score)
#     assert len(f1_scores) > 0, 'No F1-score calculated'
#     # modified weights if the gold set is empty

#     weights = [1/len(f1_scores)]*len(f1_scores)
#     weighted_f1_score = np.array(f1_scores) * np.array(weights)
#     final_score = weighted_f1_score.sum()
#     # Combine the scores using the specified weights
#     return final_score

# def normalize_values(x, min_value=0, max_value=6):
#     normalized = (x - min_value) / (max_value - min_value)
#     return normalized

# def tanh(x: np.ndarray, k: float):
#     normalized = normalize_values(x, max_value=k)
#     return np.tanh(np.log(1+normalized.sum()))

# def derive_complexity(x: list[int], k=6):
#     score = tanh(np.array(x), k)
#     return score

# def eval_selection_aggregation_condition(gold: tuple[set, set], pred: tuple[set, set], k=6):
#     # partial match
#     final_score = derive_final_score(gold, pred)
#     gold_complexity = derive_complexity([len(gold[0]), len(gold[1])], k)
#     pred_complexity = derive_complexity([len(pred[0]), len(pred[1])], k)

#     return final_score, gold_complexity, pred_complexity

# def eval_nested(gold: int, pred: int, k=6):
#     if gold == 0 and pred == 0:  # gold is always true
#         final_score = 1.0
#     elif gold < pred:
#         final_score = 0.0
#     else:
#         final_score = np.round(min(gold, pred) / (max(gold, pred) + 1e-10), 6)
#     gold_complexity = derive_complexity([gold], k)
#     pred_complexity = derive_complexity([pred], k)
#     return final_score, gold_complexity, pred_complexity

# def eval_others(gold: dict[str, set|bool], pred: dict[str, set|bool], k=6):
#     # partial match
#     gold_distinct = gold['distinct']
#     pred_distinct = pred['distinct']
#     gold_orderby = gold['order by']
#     pred_orderby = pred['order by']
#     gold_limit = set([int(gold['limit'])])
#     pred_limit = set([int(pred['limit'])])
    
#     final_score = derive_final_score(
#         [gold_distinct, gold_orderby, gold_limit], 
#         [pred_distinct, pred_orderby, pred_limit], 
#     )
#     gold_complexity = derive_complexity([len(gold_distinct), len(gold_orderby), len(gold_limit)], k)
#     pred_complexity = derive_complexity([len(pred_distinct), len(pred_orderby), len(pred_limit)], k)

#     return final_score, gold_complexity, pred_complexity
    
# def eval_all(x, k=6):
#     selection_score, selecton_gold_complexity, selection_pred_complexity = \
#         eval_selection_aggregation_condition(x['gold_selection'], x['pred_selection'], k)
#     condition_score, condition_gold_complexity, condition_pred_complexity = \
#         eval_selection_aggregation_condition(x['gold_condition'], x['pred_condition'], k)
#     aggregation_score, aggregation_gold_complexity, aggregation_pred_complexity = \
#         eval_selection_aggregation_condition(x['gold_aggregation'], x['pred_aggregation'], k)
#     nested_score, nested_gold_complexity, nested_pred_complexity = \
#         eval_nested(x['gold_nested'], x['pred_nested'], k)
#     others_score, others_gold_complexity, others_pred_complexity = \
#         eval_others(x['gold_others'], x['pred_others'], k)
    
#     results = {
#         'score': {
#             'selection': selection_score,
#             'condition': condition_score,
#             'aggregation': aggregation_score,
#             'nested': nested_score,
#             'others': others_score
#         },
#         'complexity': {
#             'selection': [selecton_gold_complexity, selection_pred_complexity],
#             'condition': [condition_gold_complexity, condition_pred_complexity],
#             'aggregation': [aggregation_gold_complexity, aggregation_pred_complexity],
#             'nested': [nested_gold_complexity, nested_pred_complexity],
#             'others': [others_gold_complexity, others_pred_complexity]
#         }
#     }
#     return results

# def eval_all_dataset(data_plus: list[dict]):
#     eval_results = []
#     for x in tqdm(data_plus):
#         results = eval_all(x)
#         gold_c = np.mean([x[0] for x in results['complexity'].values()])
#         pred_c = np.mean([x[1] for x in results['complexity'].values()])
#         new_x = {
#             'sample_id': x['sample_id'],
#             'db_id': x['db_id'],
#             'score': x['score'],
#             'question': x['question'],
#             'gold_sql': x['gold_sql'],
#             'pred_sql': x['pred_sql'],
#             'source_tables': x['source_tables'],
#             'len_tbls': len(x['source_tables']),
#             's_sel': results['score']['selection'],
#             's_cond': results['score']['condition'],
#             's_agg': results['score']['aggregation'],
#             's_nest': results['score']['nested'],
#             's_oth': results['score']['others'],
#             'gold_c_sel': results['complexity']['selection'][0],
#             'gold_c_cond': results['complexity']['condition'][0],
#             'gold_c_agg': results['complexity']['aggregation'][0],
#             'gold_c_nest': results['complexity']['nested'][0],
#             'gold_c_oth': results['complexity']['others'][0],
#             'pred_c_sel': results['complexity']['selection'][1],
#             'pred_c_cond': results['complexity']['condition'][1],
#             'pred_c_agg': results['complexity']['aggregation'][1],
#             'pred_c_nest': results['complexity']['nested'][1],
#             'pred_c_oth': results['complexity']['others'][1],
#             'gold_c': gold_c,
#             'pred_c': pred_c
#         }
#         eval_results.append(new_x)

#     df = pd.DataFrame(eval_results)
#     return df

# def get_output_result_plus(output_result, filename: str):
#     output_results_plus = []
#     errors = []
#     for x in tqdm(output_result):
#         has_error = False
#         schema = get_schema(str(proj_path / 'data' / 'spider' / 'database' / x['db_id'] / f'{x["db_id"]}.sqlite'))
#         schema = Schema(schema)
        
#         for s in ['gold', 'pred']:
#             try:
#                 sql = x[f'{s}_sql']
#                 statement = sqlparse.parse(sql.strip())[0]
#                 aliases = extract_aliases(statement)
#                 selection = extract_selection(statement, aliases, schema)
#                 condition = extract_condition(statement)
#                 aggregation = extract_aggregation(statement, aliases, schema)
#                 nested = extract_nested_setoperation(statement)
#                 others = extract_others(statement, aliases, schema)
                
#                 x[s + '_selection'] = list(map(list, selection))
#                 x[s + '_condition'] = (condition[0], list(condition[1]))
#                 x[s + '_aggregation'] = list(map(list, aggregation))
#                 x[s + '_nested'] = nested
#                 x[s + '_others'] = {'distinct': list(others['distinct']), 
#                                     'order by': list(others['order by']), 
#                                     'limit': others['limit']}
#             except Exception as e:
#                 has_error = True
#                 errors.append((x['sample_id'], s, str(e)))
#                 break
    
#         if not has_error:
#             output_results_plus.append(x)

#     with open(proj_path / 'experiments' / 'evals' / f'{filename}.json', 'w') as f:
#         # json.dumps(output_results_plus, f)
#         for x in output_results_plus:
#             l = json.dumps(x)
#             f.write(l + '\n')

#     with open(proj_path / 'experiments' / 'evals' / f'{filename}_errors.json', 'w') as f:
#         json.dump(errors, f)

#     print(f'Report: {len(output_results_plus)}/{len(output_result)} - errors = {len(errors)}')

#     return output_results_plus

# def reverse_mapping(x: dict):
#     def post(li: list):
#         return [c for c in li if c != '']
    
#     for s in ['gold', 'pred']:
#         x[f'{s}_selection'] = set(x[f'{s}_selection'][0]), set(list(map(tuple, x[f'{s}_selection'][1])))
#         x[f'{s}_condition'] = (set(sorted(post(x[f'{s}_condition'][0]))), set(post(x[f'{s}_condition'][1])))
#         x[f'{s}_aggregation'] = set(x[f'{s}_aggregation'][0]), set(list(map(tuple, x[f'{s}_aggregation'][1])))
#         x[f'{s}_others']['distinct'] = set(x[f'{s}_others']['distinct'])
#         x[f'{s}_others']['order by'] = set(x[f'{s}_others']['order by'])
#     return x

# def load_plus_data(filename: str):
#     data = []
#     with open(proj_path / 'experiments' / 'evals' / f'{filename}.json', 'r') as f:
#         for l in f:
#             x = reverse_mapping(json.loads(l))
#             data.append(x)
#     return data
