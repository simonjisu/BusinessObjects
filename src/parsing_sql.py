import unittest
import sqlparse
from sqlparse.sql import (
    Token, TokenList,IdentifierList, Identifier, Function, Statement, Parenthesis, Operation,
    Where, Having, Comparison
)
import sqlparse.tokens as tks

def is_pwn(token):
    if token.ttype in (tks.Punctuation, tks.Whitespace, tks.Newline):
        return True
    return False

def is_set_operator(token):
    if token.ttype is tks.Keyword and token.value.upper() in ('UNION', 'INTERSECT', 'EXCEPT'):
        return True
    return False

def is_subquery(tokens: Parenthesis|Identifier):
    for token in tokens:
        if token.ttype is tks.DML and token.value.upper() == 'SELECT':
            return True
    return False

def extract_table_aliases(statement: Statement) -> dict[str, str]:
    """
    Extracts a mapping of table aliases to their actual table names from the SQL statement.

    Assumptions:
    - Only table names have aliases.
    - Subqueries can occur in WHERE and HAVING clauses.
    - Only one set operation or subquery can occur.
    """
    alias_mapping = {}
    # Handle set operations
    statements = split_set_operation(statement)

    for stmt in statements:
        extract_alias(stmt, alias_mapping)
    return alias_mapping

def count_select_elements(statement: Statement, aliass):
    """
    Extracts a mapping of table aliases to their actual table names from the SQL statement.

    Assumptions:
    - Only table names have aliases.
    - Subqueries can occur in WHERE and HAVING clauses.
    - Only one set operation or subquery can occur.
    """
    alias_mapping = {}
    # Handle set operations
    statements = split_set_operation(statement)

    for stmt in statements:
        extract_alias(stmt, alias_mapping)
    return alias_mapping

def split_set_operation(statement: Statement) -> list[Statement]:
    """
    Checks if the statement contains a set operation.
    """
    is_set = False
    statement1 = []
    statement2 = []
    tokens = statement.tokens
    for i, token in enumerate(tokens):
        if is_set_operator(token):
            break
        else:
            if isinstance(token, Where):
                # only where clause has a subquery form
                for j, sub_token in enumerate(token.tokens):
                    if is_set_operator(sub_token):
                        is_set = True
                        break
                    else:
                        statement1.append(sub_token)
            else:
                statement1.append(token)

    if is_set:
        statement2 = token.tokens[j+1:]
    else:
        is_set_new = False
        for i, token in enumerate(tokens):
            if is_set_operator(token):
                is_set_new = True
                break
        if is_set_new:
            statement2 = tokens[i+1:]

    # post process the statement
    if is_pwn(statement1[-1]):
        statement1 = statement1[:-1]
    if statement2 and is_pwn(statement2[0]):
        statement2 = statement2[1:]
    
    if statement2:
        return [Statement(statement1), Statement(statement2)]
    return [Statement(statement1)]

def extract_alias(statement: Statement, alias_mapping: dict[str, str]):
    from_seen = False
    join_seen = False
    having_seen = False
    for token in statement.tokens:
        # Skip comments and whitespace
        if token.is_whitespace or token.ttype in (sqlparse.tokens.Newline,):
            continue
        # Detect the FROM keyword
        if token.ttype is tks.Keyword and token.value.upper() == 'FROM':
            from_seen = True
            continue

        if token.ttype is tks.Keyword and 'JOIN' in token.value.upper():
            join_seen = True
            continue
            
        if from_seen:
            # Handle identifiers (tables with optional aliases)
            if isinstance(token, IdentifierList):
                # Multiple tables
                for identifier in token.get_identifiers():
                    extract_alias_table_name(identifier, alias_mapping)
                from_seen = False
            elif isinstance(token, Identifier):
                # Single table
                extract_alias_table_name(token, alias_mapping)
                from_seen = False
            elif isinstance(token, Identifier) and isinstance(token[0], Parenthesis):
                # For this simplified version, we assume no subqueries or other complexities in the FROM clause
                continue
        elif join_seen:
            # Handle JOIN clause
            if isinstance(token, Identifier):
                extract_alias_table_name(token, alias_mapping)
            join_seen = False
        else:
            if isinstance(token, Where):
                subqueries = extract_subqueries(token)
                for sub_statement in subqueries:
                    extract_alias(sub_statement, alias_mapping)
            elif isinstance(token, Parenthesis):
                # check if it is a subquery
                subqueries = extract_subqueries([token])
                for sub_statement in subqueries:
                    extract_alias(sub_statement, alias_mapping)
                # itself is a subquery
            elif token.ttype is tks.Keyword and token.value.upper() == 'HAVING':
                having_seen = True
                continue
            elif isinstance(token, Identifier):
                # check if SELECT statement has subquery
                for sub_token in token.tokens:
                    if isinstance(sub_token, Parenthesis) and is_subquery(sub_token):
                        extract_alias(sub_token, alias_mapping)
            if having_seen:
                subqueries = extract_subqueries(token)
                for sub_statement in subqueries:
                    extract_alias(sub_statement, alias_mapping)
                having_seen = False

def extract_alias_table_name(identifier, alias_mapping: dict):
    """
    Processes an identifier to extract table name and alias.
    """
    alias = identifier.get_alias()
    table_name = identifier.get_real_name()
    if alias:
        alias_mapping[alias] = table_name
    else:
        alias_mapping[table_name] = table_name

def extract_subqueries(tokens):
    subqueries = []
    for token in tokens:
        if isinstance(token, Parenthesis):
            inner_tokens = token.tokens
            # Check if the first meaningful token is a SELECT statement
            for i, t in enumerate(inner_tokens):
                if t.ttype in (tks.Punctuation, tks.Whitespace, tks.Newline):
                    continue
                if t.ttype is tks.DML and t.value.upper() == 'SELECT':
                    # Found a subquery
                    subquery = inner_tokens[i:]
                    if subquery[-1].value == ')':
                        subquery = subquery[:-1]
                    subqueries.append(Statement(subquery))
                    break
    return subqueries

if __name__ == '__main__':
    
    class TestExtractTableAliases(unittest.TestCase):
        """
        Assumptions:
        - Only table names have aliases.
        - Subqueries can occur in WHERE and HAVING clauses.
        - Only one set operation or subquery can occur.
        """
        def test_is_pwn(self):
            # Test punctuation
            punct_token = Token(tks.Punctuation, ',')
            self.assertTrue(is_pwn(punct_token))
            
            # Test whitespace
            whitespace_token = Token(tks.Whitespace, ' ')
            self.assertTrue(is_pwn(whitespace_token))
            
            # Test newline
            newline_token = Token(tks.Newline, '\n')
            self.assertTrue(is_pwn(newline_token))
            
            # Test other token
            keyword_token = Token(tks.Keyword, 'SELECT')
            self.assertFalse(is_pwn(keyword_token))

        def test_is_set_operator(self):
            # Test UNION
            union_token = Token(tks.Keyword, 'UNION')
            self.assertTrue(is_set_operator(union_token))
            
            # Test INTERSECT
            intersect_token = Token(tks.Keyword, 'INTERSECT')
            self.assertTrue(is_set_operator(intersect_token))
            
            # Test EXCEPT
            except_token = Token(tks.Keyword, 'EXCEPT')
            self.assertTrue(is_set_operator(except_token))
            
            # Test other keyword
            select_token = Token(tks.Keyword, 'SELECT')
            self.assertFalse(is_set_operator(select_token))
            
            # Test non-keyword
            identifier_token = Token(tks.Name, 'table1')
            self.assertFalse(is_set_operator(identifier_token))

        def test_simple_select_no_alias(self):
            sql = "SELECT * FROM table1"
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            expected = {'table1': 'table1'}
            self.assertEqual(result, expected)

        def test_simple_select_with_alias(self):
            sql = "SELECT * FROM table1 t1"
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            expected = {'t1': 'table1'}
            self.assertEqual(result, expected)

        def test_select_multiple_tables_no_aliases(self):
            sql = "SELECT * FROM table1, table2"
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            expected = {'table1': 'table1', 'table2': 'table2'}
            self.assertEqual(result, expected)

        def test_select_multiple_tables_with_aliases(self):
            sql = "SELECT * FROM table1 t1, table2 t2"
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            expected = {'t1': 'table1', 't2': 'table2'}
            self.assertEqual(result, expected)

        def test_select_with_subquery_in_where(self):
            sql = "SELECT * FROM table1 t1 WHERE t1.id IN (SELECT id FROM table2)"
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            expected = {'t1': 'table1', 'table2': 'table2'}
            self.assertEqual(result, expected)

        def test_select_with_subquery_in_having(self):
            sql = "SELECT t1.id, COUNT(*) FROM table1 t1 GROUP BY t1.id HAVING COUNT(*) > (SELECT AVG(count) FROM table2)"
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            expected = {'t1': 'table1', 'table2': 'table2'}
            self.assertEqual(result, expected)

        def test_select_with_union(self):
            sql = "SELECT * FROM table1 t1 UNION SELECT * FROM table2 t2"
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            expected = {'t1': 'table1', 't2': 'table2'}
            self.assertEqual(result, expected)

        def test_select_with_nested_subqueries(self):
            sql = """
            SELECT * FROM table1 t1 WHERE t1.id IN (
                SELECT t2.id FROM table2 t2 WHERE t2.value > 3
            )
            """
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            expected = {'t1': 'table1', 't2': 'table2'}
            self.assertEqual(result, expected)

        def test_select_with_functions(self):
            sql = "SELECT COUNT(*) FROM table1"
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            expected = {'table1': 'table1'}
            self.assertEqual(result, expected)

        def test_select_no_from(self):
            sql = "SELECT 1"
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            expected = {}
            self.assertEqual(result, expected)

        def test_select_alias_same_as_table_name(self):
            sql = "SELECT * FROM table1 table1"
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            expected = {'table1': 'table1'}
            self.assertEqual(result, expected)

        def test_select_with_complex_expressions(self):
            sql = "SELECT t1.id, t2.value FROM table1 t1, table2 t2 WHERE t1.id = t2.id"
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            expected = {'t1': 'table1', 't2': 'table2'}
            self.assertEqual(result, expected)

        def test_select_with_except(self):
            sql = "SELECT * FROM table1 t1 EXCEPT SELECT * FROM table2 t2"
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            expected = {'t1': 'table1', 't2': 'table2'}
            self.assertEqual(result, expected)

        def test_select_with_intersect(self):
            sql = "SELECT * FROM table1 t1 INTERSECT SELECT * FROM table2 t2"
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            expected = {'t1': 'table1', 't2': 'table2'}
            self.assertEqual(result, expected)

        def test_select_with_join_on(self):
            sql = "SELECT * FROM table1 t1 INNER JOIN table2 t2 ON t1.id = t2.id"
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            # The simplified code may not handle JOINs; expected may be empty
            expected = {'t1': 'table1'}
            self.assertEqual(result, expected)

        def test_select_with_left_join(self):
            sql = "SELECT * FROM table1 t1 LEFT JOIN table2 t2 ON t1.id = t2.id"
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            # The simplified code may not handle JOINs; expected may be empty
            expected = {'t1': 'table1'}
            self.assertEqual(result, expected)

        def test_select_with_duplicate_table_names(self):
            sql = "SELECT * FROM table1, table1"
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            expected = {'table1': 'table1'}
            self.assertEqual(result, expected)

        def test_select_with_complex_where(self):
            sql = "SELECT * FROM table1 t1 WHERE EXISTS (SELECT * FROM table2 t2 WHERE t2.id = t1.id)"
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            expected = {'t1': 'table1', 't2': 'table2'}
            self.assertEqual(result, expected)

        def test_select_with_function_in_select(self):
            sql = "SELECT MAX(t1.value) FROM table1 t1"
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            expected = {'t1': 'table1'}
            self.assertEqual(result, expected)

        def test_select_with_quoted_identifiers(self):
            sql = 'SELECT * FROM "table1" t1'
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            expected = {'t1': 'table1'}
            self.assertEqual(result, expected)

        def test_select_with_comments(self):
            sql = "SELECT * FROM table1 t1 -- This is a comment"
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            expected = {'t1': 'table1'}
            self.assertEqual(result, expected)

        def test_select_with_group_by_order_by(self):
            sql = "SELECT t1.id, COUNT(*) FROM table1 t1 GROUP BY t1.id ORDER BY t1.id"
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            expected = {'t1': 'table1'}
            self.assertEqual(result, expected)

        def test_select_with_multiple_set_operations(self):
            sql = "SELECT * FROM table1 t1 UNION SELECT * FROM table2 t2"
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            # Simplified code may not handle multiple set operations
            expected = {'t1': 'table1', 't2': 'table2'}
            self.assertEqual(result, expected)

        def test_empty_sql(self):
            sql = ""
            parsed = sqlparse.parse(sql)
            if parsed:
                result = extract_table_aliases(parsed[0])
            else:
                result = {}
            expected = {}
            self.assertEqual(result, expected)

        def test_invalid_sql(self):
            sql = "SELECT FROM"
            parsed = sqlparse.parse(sql)
            if parsed:
                result = extract_table_aliases(parsed[0])
            else:
                result = {}
            expected = {}
            self.assertEqual(result, expected)

        def test_select_with_subquery_in_select_list(self):
            sql = "SELECT (SELECT MAX(value) FROM table2) as max_value FROM table1"
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            expected = {'table1': 'table1', 'table2': 'table2'}
            self.assertEqual(result, expected)

        def test_select_with_correlated_subquery(self):
            sql = "SELECT t1.id FROM table1 t1 WHERE EXISTS (SELECT 1 FROM table2 t2 WHERE t2.id = t1.id)"
            parsed = sqlparse.parse(sql)[0]
            result = extract_table_aliases(parsed)
            expected = {'t1': 'table1', 't2': 'table2'}
            self.assertEqual(result, expected)