class Prompts:
    dbschema_description = '''### Task
You are tasked with writing one line short description for each column name in a database to help users understand the data better.
You will be proveded a schema with table names and column names.

### Formatting
Your output should be of the following JSON format with `output` as key and value as a dictionary.
the value dictionary contains a dictionary with table name as key and a dictionary of column names with their descriptions as value.:
{{
    "output": {{
        "<table_name1>" : {{
            "<column_name>": <str: the one line short description of column>,
            ...
        }},
    ...
    }} 
}}

### Output
<DB_ID>: {db_id}
<SCHEMA>:\n{schema}
<OUTPUT>: 
'''
    zero_shot_inference = '''### TASK
You are tasked with generating a SQL query(in a SQLite Database) according to a user input request.
You should work in step-by-step reasoning before coming to the full SQL query.

You will be provided an input NL query.

### SCHEMA
You are working with the following schema in a SQLite Database:
{schema}

### FORMATTING
Your output should be of the following JSON format:
{{
    "rationale": "<list[str]: the step-by-step reasoning to generate the SQL query>",
    "full_sql_query": "<str: the full SQL query>"
}}

### OUTPUT
<INPUT QUERY>: {input_query}
<OUTPUT>: 
'''
    gen_template = '''### TASK
You are tasked with generating a SQL query template(in a SQLite Database) according to a user input NL question.
You should work in step-by-step reasoning before coming to the full SQL query.

### SCHEMA
You are working with the following schema in a SQLite Database:
{schema}

### SQL QUERY TEMPLATE
All the values in the SQL query should be replaced with the following placeholders:
- `[PLACEHOLDER-TYPE:STRING]` for string values
- `[PLACEHOLDER-TYPE:NUMBER]` for numeric values

### HINT
You will be provided a hint to help you. 
(1) "virtual table": a query template the might be relevant to the user's question.
(2) "description": a natural language description of the virtual table.
You can use or modify the hint to generate the query template or not to use.

### FORMATTING
Your output should be of the following JSON format:
{{
    "rationale": "<list[str]: the step-by-step reasoning to generate the SQL query template>",
    "sql": "<str: the SQL query template>",
    "hint_used": "<bool: whether you used the hint or not>"
}}

### OUTPUT
<INPUT QUERY>: {input_query}
<HINT>: {hint}
<OUTPUT>: 
'''
    zero_shot_hints_inference = '''### TASK
You are tasked with generating a SQL query(in a SQLite Database) according to a user input NL question.
You should work in step-by-step reasoning before coming to the full SQL query.

### SCHEMA
You are working with the following schema in a SQLite Database:
{schema}

### HINT
You will be provided a hint to help you. It is called "virtual table".
You will get either descriptions of the virtual tables or descriptions and templates of virtual tables together.
You can use or modify the hint to generate the full SQL query.

### FORMATTING
Your output should be of the following JSON format:
{{
    "rationale": "<list[str]: the step-by-step reasoning to generate the SQL query>",
    "full_sql_query": "<str: the full SQL query>"
}}

### OUTPUT
<INPUT QUERY>: {input_query}
<HINT>: {hint}
<OUTPUT>: 
'''
    bo_description = '''### TASK
You are tasked with generating a natural language description according to database schema, and a query template.
A query template is a SQL query with placeholders, it will be a 'virtual table'. Later, user can modify the placeholders to generate the full SQL query.
So in the description, you should clearly and concisely describe a summary of the complex data model that is easily understandable to the users.
Please output the description always start with "The virtual table..." and avoid to mention the exact value in the question.

### SQL QUERY TEMPLATE
All the values in the SQL query are replaced with the following placeholders:
- `[PLACEHOLDER-TYPE:STRING]` for string values
- `[PLACEHOLDER-TYPE:NUMBER]` for numeric values


### FORMATTING
Your output should be of the following JSON format:
{{
    "rationale": "<list[str]: the step-by-step reasoning how does the SQL query generated given the question>",
    "description": "<str: the full SQL query>"
}}

### EXAMPLE

#### SCHEMA
[Table and Columns]
Table Name: Employees
  - 'employee_id' (number): Unique identifier for each employee.
  - 'first_name' (text): First name of the employee.
  - 'last_name' (text): Last name of the employee.
  - 'department' (text): Department where the employee works.
  - 'salary' (number): Annual salary of the employee.

#### OUTPUT
<Virtual Table>: SELECT first_name, last_name FROM Employees WHERE department = [PLACEHOLDER-TYPE:STRING];
<OUTPUT>: 
{{
    "rationale": [
        "The query is identifing the relevant table, which is 'Employees'.",
        "Select the columns to display, 'first_name' and 'last_name'.",
        "Also, a condition to filter the 'department' column for the specified department using a placeholder for string values.",
        "So, the query wants to know the first and last names of employees who belong to a specific department."
    ],
    "description": "The virtual table describes the first and last names of employees from the 'Employees' table who belong to a specific department. The placeholder in the WHERE clause represents the department's name."
}}


### SCHEMA
You are working with the following schema in a SQLite Database:
{schema}

### BEGIN
<Virtual Table>: {virtual_table}
<OUTPUT>: 
'''
