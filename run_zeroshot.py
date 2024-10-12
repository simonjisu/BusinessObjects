import json
import argparse
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate

# Models
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from src.spider_sparc_preprocess import DatabaseModel, SpiderSample
from src.pymodels import SQLResponse
from src.prompts import Prompts
from src.eval import result_eq, check_if_exists_orderby
from src.db_utils import get_schema_str
from src.database import SqliteDatabase
from src.spider_sparc_preprocess import (
    process_all_tables, 
    load_samples_spider,
)
from src.eval_complexity import (
    load_plus_data, 
    get_output_result_plus, 
    eval_all_dataset
)

_ = load_dotenv(find_dotenv())

def predict_sql(
        samples: list[SpiderSample], 
        spider_tables: dict[str, DatabaseModel], 
        chain: RunnableSequence, 
        k: int = 500, 
        file_name: str = 'full_sql_output'
    ) -> list[dict]:
    all_full_sql = list()
    for i, data in tqdm(enumerate(samples), total=len(samples)):
        db_schema = get_schema_str(
            schema=spider_tables[data.db_id].db_schema, 
            foreign_keys=spider_tables[data.db_id].foreign_keys,
            col_explanation=spider_tables[data.db_id].col_explanation
        )
        input_data = {'schema': db_schema, 'input_query': data.final.question}
        output = chain.invoke(input=input_data)

        full_sql_output = {}
        full_sql_output['sample_id'] = data.sample_id
        full_sql_output['db_id'] = data.db_id
        full_sql_output['question'] = data.final.question
        full_sql_output['rationale'] = output.rationale
        full_sql_output['pred_sql'] = output.full_sql_query
        full_sql_output['gold_sql'] = data.final.sql
        full_sql_output['source_tables'] = data.final.source_tables
        all_full_sql.append(full_sql_output)

        if len(all_full_sql) == k:
            with open(proj_path / 'experiments' / f'{file_name}_{i//k}.jsonl', 'w') as f:
                for d in all_full_sql:
                    f.write(json.dumps(d) + '\n')
            all_full_sql = list()

    if len(all_full_sql) > 0:
        with open(proj_path / 'experiments' / f'{file_name}_{i//k}.jsonl', 'w') as f:
            for d in all_full_sql:
                f.write(json.dumps(d) + '\n')

def load_predictions(task: str, file_pattern: str) -> list[dict]:
    predictions = []
    for p in sorted((proj_path / 'experiments' / task).glob(file_pattern), key=lambda x: int(x.stem.split('_')[-1])):
        with p.open() as f:
            for line in f:
                predictions.append(json.loads(line))
    return predictions


def get_output_results(predictions: list[dict], spider_tables: dict[str, DatabaseModel]) -> tuple[list[dict], dict[str, list]]:
    output_results = []
    error_infos = {
        'pred_exec': [],
        'gold_exec': [],
        'python_script': [],
        'result': []
    }

    iterator = tqdm(predictions, total=len(predictions))
    for data in iterator:
        iterator.set_description(f'pred_exec: {len(error_infos["pred_exec"])} | gold_exec: {len(error_infos["gold_exec"])} | python_script: {len(error_infos["python_script"])} | result: {len(error_infos["result"])}')
        sample_id = data['sample_id']
        db_id = data['db_id']
        table = spider_tables[db_id]
        database = SqliteDatabase(str(proj_path / 'data' / 'spider' / 'database' / db_id / f'{db_id}.sqlite'), foreign_keys=table.foreign_keys)
        pred_sql = data['pred_sql'] # sqlglot.parse_one(, read='sqlite').sql()
        gold_sql = data['gold_sql']
        question = data['question']
        
        error_info = ''
        try:
            pred_result = database.execute(pred_sql, rt_pandas=False)
        except Exception as e:
            pred_result = []
            error_infos['pred_exec'].append(sample_id)
            error_info = 'Predction Execution Error:' + str(e)
            score = 0
        try:
            gold_result = database.execute(gold_sql, rt_pandas=False)
        except Exception as e:
            error_infos['gold_exec'].append(sample_id)
            error_info = 'Gold Execution Error:' + str(e)

        if 'Gold Execution Error' in error_info:
            continue
        elif 'Predction Execution Error' in error_info:
            output_results.append(
                {
                    'sample_id': sample_id, 
                    'db_id': db_id,
                    'question': question,
                    'score': score,
                    'gold_sql': gold_sql,
                    'pred_sql': pred_sql,
                    'source_tables': data['source_tables'],
                    'error_info': error_info
                }
            )
            continue
        else:
            exists_orderby = check_if_exists_orderby(gold_sql)
            
            try:
                score = int(result_eq(pred_result, gold_result, order_matters=exists_orderby))
            except Exception as e:
                print(f"An error occurred: {e}")
                score = 0
                error_info = 'Python Script Error:' + str(e)
                error_infos['python_script'].append(sample_id)

            if score == 0 and error_info == '':
                error_info = 'Result not equal'
                error_infos['result'].append(sample_id)
            output_results.append(
                {
                    'sample_id': sample_id, 
                    'db_id': db_id,
                    'question': question,
                    'score': score,
                    'gold_sql': gold_sql,
                    'pred_sql': pred_sql,
                    'source_tables': data['source_tables'],
                    'error_info': error_info
                }
            )

    return output_results, error_infos

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot SQL generation with OpenAI')
    parser.add_argument('--ds', type=str, default='spider', help='Dataset to use for training.')
    parser.add_argument('--table_file', type=str, default='tables.json', help='File containing the tables.')
    parser.add_argument('--description_file', type=str, default='description.json', help='File containing the descriptions.')
    parser.add_argument('--type', type=str, default='train', help='Type of data to use for .')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='Model to use for training.')
    parser.add_argument('--task', type=str, default='zero_shot', help='`zero_shot` or `post_process` or `output_result_plus`')
    
    args = parser.parse_args()

    proj_path = Path('.').resolve()
    assert proj_path.name == 'BusinessObjects', f'Expected project path to be BusinessObjects, but got {proj_path.name}'

    model_dict = {
        'gpt-4o-mini': ChatOpenAI,
        'google-generative-ai': ChatGoogleGenerativeAI
    }

    with (proj_path / 'data' / args.ds / args.table_file).open() as f:
        tables = json.load(f)

    with (proj_path / 'data' / args.description_file).open() as f:
        all_descriptions = json.load(f)
    spider_tables = process_all_tables(tables, descriptions=all_descriptions)

    eval_path = proj_path / 'experiments' / 'evals'
    if not eval_path.exists():
        eval_path.mkdir(parents=True)
        
    if args.task == 'zero_shot':
        samples = load_samples_spider(proj_path / 'data' / f'{args.ds}_{args.type}.json')
        print(f'{args.ds}-{args.type} samples loaded: {len(samples)}')    
        prompt = PromptTemplate(
            template=Prompts.zero_shot_inference,
            input_variables=['schema', 'input_query']
        )

        model_openai = ChatOpenAI(
            model='gpt-4o-mini',
            temperature=0.0,
        )

        model = model_openai.with_structured_output(SQLResponse)
        chain = (prompt | model)

        # run zero-shot SQL generation
        predict_sql(samples, spider_tables, chain, k=500, file_name=f'{args.ds}_{args.type}')

    elif args.task == 'post_process':
        
        predictions = load_predictions(f'{args.ds}_{args.type}_*')
        output_results, errors = get_output_results(predictions, spider_tables)
        with open(eval_path / f'{args.ds}_{args.type}_eval.json', 'w') as f:
            json.dump(output_results, f)
        
        with open(eval_path / f'{args.ds}_{args.type}_errors.json', 'w') as f:
            json.dump(errors, f)

    elif args.task == 'output_result_plus':
        with open(eval_path / f'{args.ds}_{args.type}_eval.json') as f:
            output_results = json.load(f)
        get_output_result_plus(output_results, f'{args.ds}_{args.type}_plus')
        data_plus = load_plus_data(f'{args.ds}_{args.type}_plus')
        df_eval = eval_all_dataset(data_plus)
        df_eval.to_csv(eval_path / f'{args.ds}_{args.type}_eval_plus.csv', index=False)
    else:
        raise ValueError(f'Unknown task: {args.task}, must be one of `zero_shot`, `post_process`, `output_result_plus`')