import os
import json
import numpy as np
import pandas as pd
import logging

from pathlib import Path
from src.db_utils import get_schema_str, get_data_dict, get_schema_str_with_tables
from src.database import SqliteDatabase, DuckDBDatabase
from func_timeout import func_timeout, FunctionTimedOut

import sqlite3

import os 
from dotenv import load_dotenv, find_dotenv
from collections import defaultdict
from tqdm import tqdm
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.utils.json import parse_json_markdown

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI

_ = load_dotenv(find_dotenv())

logger = logging.getLogger()

from langchain_openai import OpenAIEmbeddings
from uuid import uuid4
from langchain_core.documents import Document
from langchain_chroma import Chroma

from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_O1_KEY"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

db_path = ''
def get_database_schema(db_path, tables_list) -> str:

    stmt = ''

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Fetch names of all tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cur.fetchall()

    # Fech create statements for all tables
    for table in tables:
        table_name = table[0]
        if tables_list and table_name not in tables_list:
            continue
        cur.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        create_statement = cur.fetchone()[0]

        stmt += create_statement + '\n\n'

    conn.close()
    return stmt

def db_execute(database, sql):
    return database.execute(sql)

def execute_sql(database, sql, timeout=60*1):
    # timeout = max 1 minutes
    res = func_timeout(timeout, db_execute, args=(database, sql))
    return res

## Select Questions and Business Objects
def select_questions(root_path, question_path, bo_path, ref_query="all"):
    with open(question_path, 'r') as f:
        questions = json.load(f)
    for question in questions:
        with open(os.path.join(root_path, question['gold_sql']), 'r') as f:
            question['gold_sql'] = f.read()

    with open(bo_path, 'r') as f:
        bos = json.load(f)
    bos_dict = {}
    for bo in bos:
        ref_id = bo['ref_id']
        with open(os.path.join(root_path, bo['virtual_table']), 'r') as f:
            bo['virtual_table'] = f.read()
        if not bos_dict.get(ref_id):
            bos_dict[ref_id] = []
        bos_dict[ref_id].append(bo)

    NLsamples = []
    for i in range(len(questions)):
        q = questions[i]
        ref_id = q['ref_id']
        bos = bos_dict[ref_id]
        NLsamples.append({'instance_id': q['instance_id'],
                          'ref_id': q['ref_id'],
                          'question': q['question'], 
                          'gold_sql': q['gold_sql'],
                          'hint': q['hints'],
                          'bo': '\n'.join(bo['business_abstraction'] for bo in bos),
                          'virtual_table': '\n'.join(bo['virtual_table'] for bo in bos)
                         })
    NLsamples = pd.DataFrame.from_dict(NLsamples)
    
    if ref_query == "all":
        return NLsamples.to_dict('records')
    else:
        testNLsamples = NLsamples[NLsamples['ref_id'] == ref_query]
        testNLsamples = testNLsamples.to_dict('records')
        return testNLsamples

def create_bo_store(root_path, bo_path, general_bo=False):
    # get business object
    with open(bo_path, 'r') as f:
        bos = json.load(f)
    bos_dict = {}
    for bo in bos:
        ref_id = bo['ref_id']
        with open(os.path.join(root_path, bo['virtual_table']), 'r') as f:
            bo['virtual_table'] = f.read()
        if not bos_dict.get(ref_id):
            bos_dict[ref_id] = []
        bos_dict[ref_id].append(bo)
    
    # create vector store
    if general_bo:
        store_name = "bo_store_general"
    else:
        store_name = "bo_store_full"
    vector_store = Chroma(
        collection_name=store_name,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=os.path.join(root_path, store_name),
    )
    vector_store.reset_collection()
    docs = []
    for ref_id in bos_dict.keys():
        bos = bos_dict.get(ref_id)
        if basic_bo:
            bos = bos[0:1]
        ba = '\n'.join(bo['business_abstraction'] for bo in bos)
        vt = '\n'.join(bo['virtual_table'] for bo in bos)
        content = ba+'\n\n### Query template:\n'+vt
        doc = Document(page_content=content,
                       metadata={"ref_id": ref_id},
                       id=ref_id)
        docs.append(doc)

    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(documents=docs, ids=uuids)
    return vector_store

template_sql_generation = '''### Task
You are a data science expert.
Below, you are presented with a database schema and a question (and potentially a hint).
Your task is to read the schema, understand the question, and generate a valid SQLite query to answer the question.
Before generating the final SQL query think step by step on how to write the query.

### Database Schema
You are working with the following schema:
{schema}

### Database admin instructions:
1. When you need to find the highest or lowest values based on a certain condition, using ORDER BY + LIMIT 1 is prefered over using MAX/MIN within sub queries.
2. If predicted query includes an ORDER BY clause to sort the results, you should only include the column(s) used for sorting in the SELECT clause if the question specifically ask for them. Otherwise, omit these columns from the SELECT.
3. If the question doesn't specify exactly which columns to select, between name column and id column, prefer to select id column.
4. Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
5. Predicted query should return all of the information asked in the question without any missing or extra information.
6. For key phrases mentioned in the question, we have provided the most similar values within the columns denoted by "-- examples" in front of the corresponding column names. This is a crucial hint indicating the correct columns to use for your SQL query.
7. No matter of how many things the question asks, you should only return one SQL query as the answer having all the information asked in the question, seperated by a comma.
8. Never use || to concatenate columns in the SELECT. Rather output the columns as they are.
9. If you are joining multiple tables, make sure to use alias names for the tables and use the alias names to reference the columns in the query. Use T1, T2, T3, ... as alias names.
10. If you are doing a logical operation on a column, such as mathematical operations and sorting, make sure to filter null values within those columns.
11. If you are working with SQL functions, make sure to use VALID SQLite functions such as date/time, string,...

### Formatting
Your output should be of the following JSON format:
{{
    "rationale": "<str: the step-by-step reasoning to generate the SQL query>",
    "sql": "<str: the SQL query>"
}}

Question: 
{question}

Hint: 
{hint}

Take a deep breath and think step by step to find the correct sqlite SQL query. 
If you follow all the instructions and generate the correct query, I will give you 1 million dollars.
'''

template_sql_generation_gemini = [
    ("system", """### Task
You are a data science expert.
Below, you are presented with a database schema and a question (and potentially a hint).
Your task is to read the schema, understand the question, and generate a valid SQLite query to answer the question.
Before generating the final SQL query think step by step on how to write the query.

### Database Schema
You are working with the following schema:
{schema}

### Database admin instructions:
1. When you need to find the highest or lowest values based on a certain condition, using ORDER BY + LIMIT 1 is prefered over using MAX/MIN within sub queries.
2. If predicted query includes an ORDER BY clause to sort the results, you should only include the column(s) used for sorting in the SELECT clause if the question specifically ask for them. Otherwise, omit these columns from the SELECT.
3. If the question doesn't specify exactly which columns to select, between name column and id column, prefer to select id column.
4. Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
5. Predicted query should return all of the information asked in the question without any missing or extra information.
6. For key phrases mentioned in the question, we have provided the most similar values within the columns denoted by "-- examples" in front of the corresponding column names. This is a crucial hint indicating the correct columns to use for your SQL query.
7. No matter of how many things the question asks, you should only return one SQL query as the answer having all the information asked in the question, seperated by a comma.
8. Never use || to concatenate columns in the SELECT. Rather output the columns as they are.
9. If you are joining multiple tables, make sure to use alias names for the tables and use the alias names to reference the columns in the query. Use T1, T2, T3, ... as alias names.
10. If you are doing a logical operation on a column, such as mathematical operations and sorting, make sure to filter null values within those columns.
11. If you are working with SQL functions, make sure to use VALID SQLite functions such as date/time, string,...

Hint: 
{hint}
"""),
    ("human", "{question}"),
]

### Chain of Thought Prompt
class SQLGenerationOutputFormat(BaseModel):
    sql: str = Field(description='The SQL query.')
    rationale: str = Field(description='The step-by-step reasoning to generate the SQL query.')
        
def sql_generation(lm_model, testNLsamples, bo_store, option=0):
    prompt = PromptTemplate(
        template=template_sql_generation,
        input_variables=['schema', 'question', 'hint']
    )

    model = lm_model.with_structured_output(SQLGenerationOutputFormat)
    chain = (prompt | model)
    
    all_full_sql = list()
    for idx in tqdm(range(len(testNLsamples))):
        data = testNLsamples[idx]
        db_id = 'tpc-h'
        db_schema = get_database_schema(db_path=db_path, tables_list=[])
        hint = data['hint']
        if option == 1:
            hint = 'Please refer to the following business object:\n'+data['bo']
        elif option >= 2:
            results = bo_store.similarity_search_with_relevance_scores(
                data['question'],
                k=1,
            )
            logger.info(f"q_id: {data['instance_id']}, score: {results[0][1]}")
            if results[0][1] <= 0.4:
                results = bo_store.similarity_search_with_relevance_scores(
                    data['question'],
                    k=3,
                )
                bo = ''
                for res in results:
                    bo += res[0].page_content + '\n\n'
            else:
                bo = results[0][0].page_content
            hint = 'Please refer to the following business object:\n' + bo
        input_data = {'schema': db_schema, 'question': data['question'], 'hint': hint}
        #print(input_data)
        output = chain.invoke(input=input_data)
        #print(output)
        full_sql_output = {}
        full_sql_output['q_id'] = data['instance_id']
        full_sql_output['db_id'] = db_id
        full_sql_output['question'] = data['question']
        full_sql_output['rationale'] = output.rationale
        full_sql_output['sql'] = output.sql
        full_sql_output['gold_sql'] = data['gold_sql']
        full_sql_output['hint'] = hint
        all_full_sql.append(full_sql_output)
    for t in all_full_sql:
        for k, v in t.items():
            logger.info("'{}': {}".format(k,v))
    return all_full_sql

def sql_generation_o1(llm, testNLsamples, bo_store, option=0):
    prompt = PromptTemplate(
        template=template_sql_generation,
        input_variables=['schema', 'question', 'hint']
    )

    all_full_sql = list()
    for idx in tqdm(range(len(testNLsamples))):
        data = testNLsamples[idx]
        db_id = 'tpc-h'
        db_schema = get_database_schema(db_path=db_path, tables_list=[])
        hint = data['hint']
        if option == 1:
            hint = 'Please refer to the following business object:\n'+data['bo']
        elif option >= 2:
            results = bo_store.similarity_search_with_relevance_scores(
                data['question'],
                k=1,
            )
            logger.info(f"q_id: {data['instance_id']}, score: {results[0][1]}")
            if results[0][1] <= 0.4:
                results = bo_store.similarity_search_with_relevance_scores(
                    data['question'],
                    k=3,
                )
                bo = ''
                for res in results:
                    bo += res[0].page_content + '\n\n'
            else:
                bo = results[0][0].page_content
            hint = 'Please refer to the following business object:\n' + bo
        formatted_prompt = prompt.format(schema=db_schema, question=data['question'], hint=data['hint'])
        #print(formatted_prompt)
        
        response = client.chat.completions.create(
            model=llm,
            messages=[
                {
                    "role": "user", 
                    "content": formatted_prompt
                }
            ]
        )
        response = response.choices[0].message.content
        output = parse_json_markdown(response)
        #print(output)
        
        full_sql_output = {}
        full_sql_output['q_id'] = data['instance_id']
        full_sql_output['db_id'] = db_id
        full_sql_output['question'] = data['question']
        full_sql_output['rationale'] = output['rationale']
        full_sql_output['sql'] = output['sql']
        full_sql_output['gold_sql'] = data['gold_sql']
        full_sql_output['hint'] = hint
        all_full_sql.append(full_sql_output)
    for t in all_full_sql:
        for k, v in t.items():
            logger.info("'{}': {}".format(k,v))
    return all_full_sql

def sql_generation_gemini(gemini_model, testNLsamples, bo_store, option=0):
    prompt = ChatPromptTemplate.from_messages(template_sql_generation_gemini)

    model = gemini_model.with_structured_output(SQLGenerationOutputFormat)
    chain = (prompt | model)
    
    all_full_sql = list()
    for idx in tqdm(range(len(testNLsamples))):
        data = testNLsamples[idx]
        db_id = 'tpc-h'
        db_schema = get_database_schema(db_path=db_path, tables_list=[])
        hint = data['hint']
        if option == 1:
            hint = 'You are provided the following business object schema to help you answer the user question:\n'+data['bo']
        elif option >= 2:
            results = bo_store.similarity_search_with_relevance_scores(
                data['question'],
                k=1,
            )
            logger.info(f"q_id: {data['instance_id']}, score: {results[0][1]}")
            if results[0][1] <= 0.4:
                results = bo_store.similarity_search_with_relevance_scores(
                    data['question'],
                    k=3,
                )
                bo = ''
                for res in results:
                    bo += res[0].page_content + '\n\n'
            else:
                bo = results[0][0].page_content
            hint = 'You are provided the following business object schema to help you answer the user question:\n' + bo
        input_data = {'schema': db_schema, 'question': data['question'], 'hint': hint}
        #print(input_data)
        output = chain.invoke(input=input_data)
        #print(output)
        if not output: continue
        full_sql_output = {}
        full_sql_output['q_id'] = data['instance_id']
        full_sql_output['db_id'] = db_id
        full_sql_output['question'] = data['question']
        full_sql_output['rationale'] = output.rationale
        full_sql_output['sql'] = output.sql
        full_sql_output['gold_sql'] = data['gold_sql']
        full_sql_output['hint'] = hint
        all_full_sql.append(full_sql_output)
    for t in all_full_sql:
        for k, v in t.items():
            logger.info("'{}': {}".format(k,v))
    return all_full_sql

### SQL Revision
class SQLRevisionOutputFormat(BaseModel):
    revised_sql: str = Field(description='The revised SQL query.')
    rationale: str = Field(description='Your thought process on how you arrived at the solution.')

template_sql_revision = '''### Task
Your task is to make sure a query follows the database admin instructions and use the correct conditions.

### Database Schema
You are working with the following schema:
{schema}

Database admin instructions:
1. When you need to find the highest or lowest values based on a certain condition, using ORDER BY + LIMIT 1 is prefered over using MAX/MIN within sub queries.
2. If predicted query includes an ORDER BY clause to sort the results, you should only include the column(s) used for sorting in the SELECT clause if the question specifically ask for them. Otherwise, omit these columns from the SELECT.
3. If the question doesn't specify exactly which columns to select, between name column and id column, prefer to select id column.
4. Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
5. Predicted query should return all of the information asked in the question without any missing or extra information.
7. For key phrases mentioned in the question, we have provided the most similar values within the columns denoted by "-- examples" in front of the corresponding column names. This is a crucial hint indicating the correct columns to use for your SQL query.
8. No matter of how many things the question asks, you should only return one SQL query as the answer having all the information asked in the question, seperated by a comma.
9. Using || ' ' ||  to concatenate is string is banned and using that is punishable by death. Never concatenate columns in the SELECT clause.
10. If you are joining multiple tables, make sure to use alias names for the tables and use the alias names to reference the columns in the query. Use T1, T2, T3, ... as alias names.
11. If you are doing a logical operation on a column, such as mathematical operations and sorting, make sure to filter null values within those columns.
12. When ORDER BY is used, just include the column name in the ORDER BY in the SELECT clause when explicitly asked in the question. Otherwise, do not include the column name in the SELECT clause.
13. Make sure to use valid SQLite functions such as date/time, string,...
14. Be attention in case of Query result has Errors or Empty to revise your SQL.

Question:
{question}

Hint:
{hint}

Predicted SQL:
{sql}

Query result:
{query_result}

Please respond with a JSON object structured as follows (if the sql query is correct, return the query as it is):

{{
    "rationale": "Your thought process on how you arrived at the solution. You don't need to explain the instructions that are satisfied.",
    "revised_sql": "Your revised SQL query."
}}

Take a deep breath and think step by step to find the correct sqlite SQL query. 
If you follow all the instructions and generate the correct query, I will give you 1 million dollars.
'''

template_sql_revision_gemini = [
    ("system", """### Task
Your task is to make sure a query follows the database admin instructions and use the correct conditions.

### Database Schema
You are working with the following schema:
{schema}

Database admin instructions:
1. When you need to find the highest or lowest values based on a certain condition, using ORDER BY + LIMIT 1 is prefered over using MAX/MIN within sub queries.
2. If predicted query includes an ORDER BY clause to sort the results, you should only include the column(s) used for sorting in the SELECT clause if the question specifically ask for them. Otherwise, omit these columns from the SELECT.
3. If the question doesn't specify exactly which columns to select, between name column and id column, prefer to select id column.
4. Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
5. Predicted query should return all of the information asked in the question without any missing or extra information.
7. For key phrases mentioned in the question, we have provided the most similar values within the columns denoted by "-- examples" in front of the corresponding column names. This is a crucial hint indicating the correct columns to use for your SQL query.
8. No matter of how many things the question asks, you should only return one SQL query as the answer having all the information asked in the question, seperated by a comma.
9. Using || ' ' ||  to concatenate is string is banned and using that is punishable by death. Never concatenate columns in the SELECT clause.
10. If you are joining multiple tables, make sure to use alias names for the tables and use the alias names to reference the columns in the query. Use T1, T2, T3, ... as alias names.
11. If you are doing a logical operation on a column, such as mathematical operations and sorting, make sure to filter null values within those columns.
12. When ORDER BY is used, just include the column name in the ORDER BY in the SELECT clause when explicitly asked in the question. Otherwise, do not include the column name in the SELECT clause.
13. Make sure to use valid SQLite functions such as date/time, string,...
14. Be attention in case of Query result has Errors or Empty to revise your SQL.

Hint:
{hint}

Predicted SQL:
{sql}

Query result:
{query_result}

Take a deep breath and think step by step to find the correct sqlite SQL query. 
If you follow all the instructions and generate the correct query, I will give you 1 million dollars.
"""),
    ("human", "{question}"),
]
        
def sql_revision(lm_model, all_full_sql):
    prompt = PromptTemplate(
        template=template_sql_revision,
        input_variables=['schema', 'question', 'hint', 'sql', 'query_result']
    )

    model = lm_model.with_structured_output(SQLRevisionOutputFormat)
    chain = (prompt | model)
    
    revised_full_sql = list()
    for idx in tqdm(range(len(all_full_sql))):
        data = all_full_sql[idx]
        db_id = 'tpc-h'
        db_schema = get_database_schema(db_path=db_path, tables_list=[])
        database = SqliteDatabase(db_file=db_path)
        try:
            pred_result = execute_sql(database, data['sql'])
        except FunctionTimedOut as e:
            pred_result = "Database Execution TimeOut Error"
        except Exception as ex:
            pred_result = ex
        #print(pred_result)
        input_data = {'schema': db_schema, 'question': data['question'], 
                      'sql': data['sql'], 'hint': data['hint'],
                      'query_result': pred_result}
        #print(input_data)
        output = chain.invoke(input=input_data)
        #print(output)
        full_sql_output = {}
        full_sql_output['q_id'] = data['q_id']
        full_sql_output['db_id'] = db_id
        full_sql_output['question'] = data['question']
        full_sql_output['rationale'] = output.rationale
        full_sql_output['sql'] = output.revised_sql
        full_sql_output['gold_sql'] = data['gold_sql']
        full_sql_output['hint'] = data['hint']
        revised_full_sql.append(full_sql_output)
    for t in revised_full_sql:
        for k, v in t.items():
            logger.info("'{}': {}".format(k,v))
    return revised_full_sql

def sql_revision_gemini(gemini_model, all_full_sql):
    prompt = ChatPromptTemplate.from_messages(template_sql_revision_gemini)

    model = gemini_model.with_structured_output(SQLRevisionOutputFormat)
    chain = (prompt | model)
    
    revised_full_sql = list()
    for idx in tqdm(range(len(all_full_sql))):
        data = all_full_sql[idx]
        db_id = 'tpc-h'
        db_schema = get_database_schema(db_path=db_path, tables_list=[])
        database = SqliteDatabase(db_file=db_path)
        try:
            pred_result = execute_sql(database, data['sql'])
        except FunctionTimedOut as e:
            pred_result = "Database Execution TimeOut Error"
        except Exception as ex:
            pred_result = ex
        #print(pred_result)
        input_data = {'schema': db_schema, 'question': data['question'], 
                      'sql': data['sql'], 'hint': data['hint'],
                      'query_result': pred_result}
        #print(input_data)
        output = chain.invoke(input=input_data)
        #print(output)
        if not output: continue
        full_sql_output = {}
        full_sql_output['q_id'] = data['q_id']
        full_sql_output['db_id'] = db_id
        full_sql_output['question'] = data['question']
        full_sql_output['rationale'] = output.rationale
        full_sql_output['sql'] = output.revised_sql.replace('\\"', '"').replace('\\n', '\n')
        full_sql_output['gold_sql'] = data['gold_sql']
        full_sql_output['hint'] = data['hint']
        revised_full_sql.append(full_sql_output)
    for t in revised_full_sql:
        for k, v in t.items():
            logger.info("'{}': {}".format(k,v))
    return revised_full_sql

### Database execution validation
def execution_validation(revised_full_sql, output_dir):
    from src.evaluate import compare_execution

    output_results = []
    for data in tqdm(revised_full_sql, total=len(revised_full_sql)):
        q_id = data['q_id']
        db_id = data['db_id']
        database = SqliteDatabase(db_file=db_path)
        error_info = None
        try:
            gold_result = database.execute(data['gold_sql'])
            gold_result.to_csv(os.path.join(output_dir, q_id+"_gold.csv"), index=False)
            print('gold_result:\n', gold_result)
            
            with open(os.path.join(output_dir, q_id+"_pred.sql"), 'w') as f:
                f.write(data['sql'])
            pred_result = execute_sql(database, data['sql'])
            pred_result.to_csv(os.path.join(output_dir, q_id+"_pred.csv"), index=False)
            print('pred_result:\n', pred_result)
            try:
                score = compare_execution(pred_result, gold_result)
            except Exception as e:
                print(f"An error occurred: {e}")
                score = 0
                error_info = 'Python Script Error: ' + str(e)
        except FunctionTimedOut as e:
            print(f"Database Execution TimeOut Error: {e}")
            score = 0
            error_info = 'Execution TimeOut Error'
        except Exception as e:
            print(f"Database execution error: {e}")
            score = 0
            error_info = 'Execution Error: ' + str(e)
        if score == 0 and error_info is None:
            error_info = 'Incorrect Query Result'
        output_results.append(
            {
                "instance_id": q_id, 
                "score": score,
                "error_info": error_info
            }
        )

    # save results
    cur_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    with open(os.path.join(output_dir, 'output_results_{}.json'.format(cur_time)), 'w') as f:
        json.dump(output_results, f, indent=4)
    logger.info({item['instance_id']: item['score'] for item in output_results})      
    score = sum([item['score'] for item in output_results]) / len(output_results)
    logger.info(f"Final score: {score}")
    return output_results, score

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default="data/tpch", type=str)
    parser.add_argument("--output_dir", default="data/tpch/outputs", type=str)
    parser.add_argument("--db_path", default="data/tpch/TPC-H.db", type=str)
    parser.add_argument("--question_path", default="data/tpch/questions.json", type=str)
    parser.add_argument("--bo_path", default="data/tpch/business_objects.json", type=str)
    parser.add_argument("--llm", default="gpt-4o-mini", type=str)
    parser.add_argument("--ref_query", default="q1", type=str, 
                        help="The TPC-H query to work on, 'all' for all queries")
    parser.add_argument("--option", default=0, type=int, 
                        help="0: only schema, 1: with business abstract, 2: with Specific BO, 3: with General BO")
    return parser.parse_args()

from datetime import datetime
from pathlib import Path
if __name__ == "__main__":
    args = parse_args()
    
    output_dir = os.path.join(args.output_dir, args.llm + "_" + str(args.option) + "_" + args.ref_query)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(filename=os.path.join(output_dir, 'running_{}.log'.format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))), level=logging.INFO)
    
    db_path = args.db_path
    
    if "gpt-" in args.llm:
        lm_model = ChatOpenAI(
            model=args.llm,
            temperature=0.0,
        )
    elif "gemini-" in args.llm:
        # init project
        import vertexai
        VERTEXAI_PROJECT_NAME = os.environ.get("VERTEXAI_PROJECT_NAME")
        vertexai.init(project=VERTEXAI_PROJECT_NAME, location="us-central1")
        # init llm
        gemini_llm = ChatVertexAI(
            model=args.llm,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            verbose=True,
        )
    logger.info('Working with LLM: ' + args.llm)
    
    # read questions and business objects
    NLsamples = select_questions(args.root_path, args.question_path, args.bo_path, args.ref_query)
    if args.option == 2:
        bo_store = create_bo_store(args.root_path, args.bo_path, general_bo=False)
    elif args.option == 3:
        bo_store = create_bo_store(args.root_path, args.bo_path, general_bo=True)
    
    if "o1" in args.llm:
        all_sql_generation = sql_generation_o1(args.llm, NLsamples, bo_store, args.option)
        revised_full_sql = all_sql_generation
    elif "gemini-" in args.llm:
        all_sql_generation = sql_generation_gemini(gemini_llm, NLsamples, bo_store, args.option)
        revised_full_sql = sql_revision_gemini(gemini_llm, all_sql_generation)
    elif "gpt-" in args.llm:
        all_sql_generation = sql_generation(lm_model, NLsamples, bo_store, args.option)
        revised_full_sql = sql_revision(lm_model, all_sql_generation)
    
    # db execution validation
    output_results, score = execution_validation(revised_full_sql, output_dir)
    print(output_results, score)

