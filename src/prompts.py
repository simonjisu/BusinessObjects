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
# pipeline inference
    gen_template_with_bos = '''### TASK
You are tasked with generating a SQL query template(in a SQLite Database) according to a user input NL question.
You should work in step-by-step reasoning before coming to the full SQL query.

### SCHEMA
You are working with the following schema in a SQLite Database:
{schema}

### SQL QUERY TEMPLATE
All the values in the SQL query should be replaced with the following placeholders:
- `[PLACEHOLDER-TYPE:STRING]` for string values
- `[PLACEHOLDER-TYPE:NUMERIC]` for numeric values

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
    gen_template_no_bos = '''### TASK
You are tasked with generating a SQL query template(in a SQLite Database) according to a user input NL question.
You should work in step-by-step reasoning before coming to the full SQL query.

### SCHEMA
You are working with the following schema in a SQLite Database:
{schema}

### SQL QUERY TEMPLATE
All the values in the SQL query should be replaced with the following placeholders:
- `[PLACEHOLDER-TYPE:STRING]` for string values
- `[PLACEHOLDER-TYPE:NUMERIC]` for numeric values

### FORMATTING
Your output should be of the following JSON format:
{{
    "rationale": "<list[str]: the step-by-step reasoning to generate the SQL query template>",
    "sql": "<str: the SQL query template>",
}}

### OUTPUT
<INPUT QUERY>: {input_query}
<OUTPUT>: 
'''
    keyword_extraction_spider = '''
### Objective 
Analyze the given input query(a natural language question) and a sql template to identify and extract keywords, keyphrases, and named entities for each column. 
These elements are crucial for understanding the core components of the inquiry and the guidance provided. This process involves recognizing and isolating significant terms and phrases that could be instrumental in formulating searches or queries related to the posed question.

### Instructions
Read the input query carefully: Understand the primary focus and specific details of the question. Look for any named entities (such as organizations, locations, etc.), technical terms, and other phrases that encapsulate important aspects of the inquiry.

### SQL query template
All the values in the SQL query should be replaced with the following placeholders:
- `[PLACEHOLDER-TYPE:STRING]` for string values
- `[PLACEHOLDER-TYPE:NUMERIC]` for numeric values

### Keyphrases and Entities
Combine your findings from both the question and sql template for columns that need a value to fill in the placeholder. 
This list should contain:
- Keywords: Single words that capture essential aspects of the question.
- Keyphrases: Short phrases or named entities that represent specific concepts, locations, organizations, or other significant details.
- Ensure to maintain the original phrasing or terminology used in the question.

### Example 1
<INPUT QUERY>: "What is the annual revenue of Acme Corp in the United States for 2022?"
<SQL TEMPLATE>: "SELECT annual_revenue FROM financial_reports WHERE company_name = [PLACEHOLDER-TYPE:STRING] AND country = [PLACEHOLDER-TYPE:STRING] AND fiscal_year = [PLACEHOLDER-TYPE:NUMERIC];"
<OUTPUT>:
{{
    "extraction": {{
        "company_name": ["Acme Corp", "Acme Corporation", "Acme", "A.C."],
        "country": ["United States", "U.S."],
        "fiscal_year": ["2022"]
    }}
}}

### Example 2
<INPUT QUERY>: "In the Winter and Summer Olympics of 1988, which game has the most number of competitors? Find the difference of the number of competitors between the two games."
<SQL TEMPLATE>: "SELECT games_name, COUNT(person_id) AS num_competitors FROM olympics_participation WHERE games_year = [PLACEHOLDER-TYPE:NUMERIC] AND games_season IN ([PLACEHOLDER-TYPE:STRING], [PLACEHOLDER-TYPE:STRING]) GROUP BY games_name ORDER BY num_competitors DESC LIMIT 1;"
<OUTPUT>:
{{
    "extraction": {{
        "games_year": ["1988"],
        "games_season": ["1988 Winter", "1988 Summer"]
    }}
}}

### Example 3
<INPUT QUERY>: "How many Men's 200 Metres Freestyle events did Ian James Thorpe compete in?"
<SQL TEMPLATE>: "SELECT COUNT(event_id) FROM athlete_participation WHERE athlete_name = [PLACEHOLDER-TYPE:STRING] AND event_name = [PLACEHOLDER-TYPE:STRING];"
<OUTPUT>:
{{
    "extraction": {{
        "athlete_name": ["Ian James Thorpe"],
        "event_name": ["Men's 200 Metres Freestyle", "Swimming Men's 200 metres Freestyle"],
    }}
}}

### TASK
Given the following question and schema, identify and list all relevant keywords, keyphrases, and named entities for columns that need a value to fill in the placeholder.
Remember, the keyphrases and entities can only be values to be fill into the sql template. Do not include any functions, operators, or other SQL syntax.
Please provide your findings as `FORMATTING` instructions, capturing the essence of both the question and evidence through the identified terms and phrases. 

### SCHEMA
You are working with the following schema in a SQLite Database:
{schema}

### FORMATTING
Your output should be of the following JSON format:
{{
    "rationale": "<list[str]: the step-by-step reasoning for the extraction>",
    "extraction": "<dict[str, list[str]]: keys for the dictionary are column names and values are list of keywords, keyphrases, and named entities extracted from the question, hint and sql template>",
}}
Make sure the keyphrases and entities are string types in the extraction list. 
The key of the `extraction` dictionary should not be empty like `""`.
If values in `extraction` dictionary is empty, please leave it as an empty list `[]`, not a empty string `""`.

<INPUT QUERY>: {input_query}
<SQL TEMPLATE>: {sql_template}
<OUTPUT>: 
'''
    keyword_extraction_bird = '''
### Objective 
Analyze the given input query(a natural language question), a evidence and a sql template to identify and extract keywords, keyphrases, and named entities for each column. 
These elements are crucial for understanding the core components of the inquiry and the guidance provided. This process involves recognizing and isolating significant terms and phrases that could be instrumental in formulating searches or queries related to the posed question.

### Analyze the evidence
The evidence is designed to direct attention toward certain elements relevant to answering the question. Extract any keywords, phrases, or named entities that could provide further clarity or direction in formulating an answer.

### Instructions
Read the input query carefully: Understand the primary focus and specific details of the question. Look for any named entities (such as organizations, locations, etc.), technical terms, and other phrases that encapsulate important aspects of the inquiry.

### SQL query template
All the values in the SQL query should be replaced with the following placeholders:
- `[PLACEHOLDER-TYPE:STRING]` for string values
- `[PLACEHOLDER-TYPE:NUMERIC]` for numeric values

### Keyphrases and Entities
Combine your findings from both the question, evidence and sql template for columns that need a value to fill in the placeholder. 
This list should contain:
- Keywords: Single words that capture essential aspects of the question or the evidence.
- Keyphrases: Short phrases or named entities that represent specific concepts, locations, organizations, or other significant details.
- Ensure to maintain the original phrasing or terminology used in the question and the evidence.

### Example 1
<INPUT QUERY>: "What is the annual revenue of Acme Corp in the United States for 2022?"
<EVIDENCE>: "Focus on financial reports and U.S. market performance for the fiscal year 2022."
<SQL TEMPLATE>: "SELECT annual_revenue FROM financial_reports WHERE company_name = [PLACEHOLDER-TYPE:STRING] AND country = [PLACEHOLDER-TYPE:STRING] AND fiscal_year = [PLACEHOLDER-TYPE:NUMERIC];"
<OUTPUT>:
{{
    "extraction": {{
        "company_name": ["Acme Corp", "Acme Corporation", "Acme", "A.C."],
        "country": ["United States", "U.S."],
        "fiscal_year": ["2022"]
    }}
}}

### Example 2
<INPUT QUERY>: "In the Winter and Summer Olympics of 1988, which game has the most number of competitors? Find the difference of the number of competitors between the two games."
<EVIDENCE>: "the most number of competitors refer to MAX(COUNT(person_id)); SUBTRACT(COUNT(person_id where games_name = '1988 Summer'), COUNT(person_id where games_name = '1988 Winter'));"
<SQL TEMPLATE>: "SELECT games_name, COUNT(person_id) AS num_competitors FROM olympics_participation WHERE games_year = [PLACEHOLDER-TYPE:NUMERIC] AND games_season IN ([PLACEHOLDER-TYPE:STRING], [PLACEHOLDER-TYPE:STRING]) GROUP BY games_name ORDER BY num_competitors DESC LIMIT 1;"
<OUTPUT>:
{{
    "extraction": {{
        "games_year": ["1988"],
        "games_season": ["1988 Winter", "1988 Summer"]
    }}
}}

### Example 3
<INPUT QUERY>: "How many Men's 200 Metres Freestyle events did Ian James Thorpe compete in?"
<EVIDENCE>: "Men's 200 Metres Freestyle events refer to event_name = 'Swimming Men''s 200 metres Freestyle'; events compete in refers to event_id;"
<SQL TEMPLATE>: "SELECT COUNT(event_id) FROM athlete_participation WHERE athlete_name = [PLACEHOLDER-TYPE:STRING] AND event_name = [PLACEHOLDER-TYPE:STRING];"
<OUTPUT>:
{{
    "extraction": {{
        "athlete_name": ["Ian James Thorpe"],
        "event_name": ["Men's 200 Metres Freestyle", "Swimming Men's 200 metres Freestyle"],
    }}
}}

### TASK
Given the following question, evidence and schema, identify and list all relevant keywords, keyphrases, and named entities for columns that need a value to fill in the placeholder.
Remember, the keyphrases and entities can only be values to be fill into the sql template. Do not include any functions, operators, or other SQL syntax.
Please provide your findings as `FORMATTING` instructions, capturing the essence of both the question and evidence through the identified terms and phrases. 

### SCHEMA
You are working with the following schema in a SQLite Database:
{schema}

### FORMATTING
Your output should be of the following JSON format:
{{
    "rationale": "<list[str]: the step-by-step reasoning for the extraction>",
    "extraction": "<dict[str, list[str]]: keys for the dictionary are column names and values are list of keywords, keyphrases, and named entities extracted from the question, hint and sql template>",
}}
Make sure the keyphrases and entities are string types in the extraction list. 
The key of the `extraction` dictionary should not be empty like `""`.
If values in `extraction` dictionary is empty, please leave it as an empty list `[]`, not a empty string `""`.

<INPUT QUERY>: {input_query}
<EVIDENCE>: {evidence}
<SQL TEMPLATE>: {sql_template}
<OUTPUT>: 
'''
    fill_in = '''### SCHEMA
You are working with the following schema in a SQLite Database:
{schema}

### SQL QUERY TEMPLATE
All the values in the SQL query are replaced with the following placeholders:
- `[PLACEHOLDER-TYPE:STRING]` for string values
- `[PLACEHOLDER-TYPE:NUMERIC]` for numeric values

### HINT
You will be provided a hint to help you. The hint contains a possible values that refers to the column in the SQL query template.
The structure of the hint as follows:
```json
{{
    <table_name: str>: {{  // table 1
        <column_name: str>: [<value1: str>, <value2: str>, ...],  // column 1 and its possible values
        ... // other columns
    }},
    ...  // other tables
}}
```
Please use the hint to fill in the placeholder if exists. 
If the hint is empty for certain columns, you should force to fill in the proper values by yourself. 
Do not leave any placeholder unfilled.

### FORMATTING
Your output should be of the following JSON format:
{{
    "rationale": "<list[str]: the step-by-step reasoning to generate the SQL query template>",
    "sql": "<str: the full SQL query>",
}}

### TASK
You are tasked with fill in values to the specific placeholder(`[PLACEHOLDER-TYPE:STRING]` or `[PLACEHOLDER-TYPE:NUMERIC]`) according to a user input NL question.
If the type of the placeholder is string, you should fill in the string value. If the type of the placeholder is number, you should fill in the numeric value.
You should work in step-by-step reasoning for the placeholder fill-in task to complete the sql template. 
Do not leave any placeholder unfilled.

### OUTPUT
<INPUT QUERY>: {input_query}
<HINT>: {hint}
<SQL TEMPLATE>: {sql_template}
<OUTPUT>: 
'''
# direct inference
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
    "sql": "<str: the full SQL query>"
}}

### OUTPUT
<INPUT QUERY>: {input_query}
<OUTPUT>: 
'''
    zero_shot_hints_inference = '''### TASK
You are tasked with generating a SQL query(in a SQLite Database) according to a user input NL question.
You should work in step-by-step reasoning before coming to the full SQL query.

### SCHEMA
You are working with the following schema in a SQLite Database:
{schema}

### HINT
You will be provided a hint to help you. 
(1) "virtual table": a query template the might be relevant to the user's question.
(2) "description": a natural language description of the virtual table.
All the values in the SQL query are replaced with the following placeholders:
- `[PLACEHOLDER-TYPE:STRING]` for string values
- `[PLACEHOLDER-TYPE:NUMERIC]` for numeric values
You can use or modify the hint to generate the final SQL query or if it is not useful please ignore the hint.

### FORMATTING
Your output should be of the following JSON format:
{{
    "rationale": "<list[str]: the step-by-step reasoning to generate the SQL query>",
    "sql": "<str: the full SQL query>",
    "hint_used": "<bool: whether you used the hint or not>"
}}
Please generate the final sql directly in the output, do not include any placeholders.

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
- `[PLACEHOLDER-TYPE:NUMERIC]` for numeric values

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
