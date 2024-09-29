import unittest
import sqlparse
from sqlparse.sql import Token
import sqlparse.tokens as tks

# Assume the functions from the provided code are imported
from parsing_sql import (
    is_pwn, is_set_operator, extract_table_aliases,
)

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
        expected = {'t1': 'table1', 't2': 'table2'}
        self.assertEqual(result, expected)

    def test_select_with_left_join(self):
        sql = "SELECT * FROM table1 t1 LEFT JOIN table2 t2 ON t1.id = t2.id"
        parsed = sqlparse.parse(sql)[0]
        result = extract_table_aliases(parsed)
        # The simplified code may not handle JOINs; expected may be empty
        expected = {'t1': 'table1', 't2': 'table2'}
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

# Add more tests if needed

if __name__ == '__main__':
    unittest.main()
