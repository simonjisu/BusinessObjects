import json
import pickle
import argparse
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate

# Models
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.callbacks.manager import get_openai_callback

from src.pymodels import SQLResponse, SpiderSample, BirdSample, DatabaseModel
from src.prompts import Prompts
from src.eval import result_eq, check_if_exists_orderby
from src.db_utils import get_schema_str
from src.database import SqliteDatabase
from src.data_preprocess import (
    process_all_tables, 
    load_samples_spider_bird,
    load_raw_data
)
from src.parsing_sql import Schema, extract_all
from collections import defaultdict
_ = load_dotenv(find_dotenv())

def predict_sql(
        samples: list[SpiderSample|BirdSample], 
        tables: dict[str, DatabaseModel], 
        chain: RunnableSequence, 
        prediction_path: Path,
        split_k: int = 2,
        file_name: str = '[args.ds]_[args.type]',
    ) -> list[dict]:
    processed_db_ids = [p.stem.split('_', split_k)[-1] for p in prediction_path.glob(f'{file_name}_*.json')]
    # restart from checkpoint
    if processed_db_ids:
        samples = [sample for sample in samples if sample.db_id not in processed_db_ids]
        print(f'Skip some processed db_ids: {len(processed_db_ids)} {processed_db_ids[-5:]}')

    samples_by_db_id = defaultdict(list)
    for sample in samples:
        samples_by_db_id[sample.db_id].append(sample)

    for db_id, samples in samples_by_db_id.items():
        results = []
        for sample in tqdm(samples, total=len(samples), desc=f"{db_id}"):
            res = {
                'sample_id': sample.sample_id, 
                'db_id': sample.db_id,
                'gold_sql': sample.final.sql,
            }

            db_schema = get_schema_str(
                schema=tables[sample.db_id].db_schema, 
                foreign_keys=tables[sample.db_id].foreign_keys,
                col_explanation=tables[sample.db_id].col_explanation
            )
            input_data = {'schema': db_schema, 'input_query': sample.final.question}
            
            with get_openai_callback() as cb:
                output = chain.invoke(input=input_data)

            res['rationale'] = output.rationale
            res['pred_sql'] = output.full_sql_query
            res['token_usage'] = {'tokens': cb.total_tokens, 'cost': cb.total_cost}
            results.append(res)
        
        with (prediction_path / f'{file_name}_{db_id}.json').open('w') as f:
            json.dump(results, f, indent=4)

def load_predictions(file_pattern: str, prediction_path: Path) -> list[dict]:
    predictions = []
    for p in sorted((prediction_path).glob(file_pattern), key=lambda x: int(x.stem.split('_')[-1])):
        with p.open() as f:
            for line in f:
                predictions.append(json.loads(line))
    return predictions

if __name__ == '__main__':
    """
    bird:
    python run_zeroshot.py --ds bird \
        --type train \
        --model gpt-4o-mini \
        --task zero_shot \
        --description_file bird_description.json \
        --k 500
    """
    parser = argparse.ArgumentParser(description='Zero-shot SQL generation with OpenAI')
    parser.add_argument('--ds', type=str, default='spider', help='Dataset to use for training. spider or bird') 
    parser.add_argument('--description_file', type=str, default='description.json', help='File containing the descriptions.')
    parser.add_argument('--type', type=str, default='train', help='Type of data to use for .')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='Model to use for training.')
    parser.add_argument('--task', type=str, default='zero_shot', help='`zero_shot` or `post_process` or `output_result_plus`')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()

    proj_path = Path('.').resolve()
    assert proj_path.name == 'BusinessObjects', f'Expected project path to be BusinessObjects, but got {proj_path.name}'

    model_dict = {
        'gpt-4o-mini': ChatOpenAI,
        'google-generative-ai': ChatGoogleGenerativeAI
    }

    data_path = proj_path / 'data' / args.ds
    experiment_folder = proj_path / 'experiments' / f'{args.ds}'
    prediction_path = experiment_folder / 'predictions' / f'{args.task}'
    eval_path = experiment_folder / 'evals' / f'{args.task}'
    for p in [prediction_path, eval_path]:
        if not p.exists():
            p.mkdir(parents=True)
    
    tables, *_ = load_raw_data(data_path, load_test=False)

    with (proj_path / 'data' / args.description_file).open() as f:
        all_descriptions = json.load(f)
    tables = process_all_tables(tables, descriptions=all_descriptions)

    if args.task == 'zero_shot':
        samples = load_samples_spider_bird(proj_path / 'data' / f'{args.ds}_{args.type}.json')
        print(f'{args.ds}-{args.type} samples loaded: {len(samples)}')    
        prompt = PromptTemplate(
            template=Prompts.zero_shot_inference,
            input_variables=['schema', 'input_query']
        )

        model_openai = ChatOpenAI(
            model='gpt-4o-mini',
            temperature=0.0,
            frequency_penalty=0.1,
        )

        model = model_openai.with_structured_output(SQLResponse)
        chain = (prompt | model)

        # run zero-shot SQL generation
        print(f'Start SQL generation {prediction_path}')
        predict_sql(samples, tables, chain, prediction_path, split_k=2,
                    file_name=f'{args.ds}_{args.type}')
        
        predictions = []
        for p in prediction_path.glob(f'{args.ds}_{args.type}_*.json'):
            with open(p) as f:
                pred = json.load(f)
                new_pred = []
                for x in pred:
                    x.pop('rationale')
                    new_pred.append(x)
                predictions.extend(new_pred)

        with open(prediction_path / f'final_{args.ds}_{args.type}.jsonl', 'w') as f:
            for p in predictions:
                f.write(json.dumps(p) + '\n')


    # elif args.task == 'post_process':
    #     predictions = load_predictions(f'{args.ds}_{args.type}_*', prediction_path)
    #     output_results, errors = get_output_results(predictions, tables)
    #     with open(eval_path / f'{args.ds}_{args.type}_eval.json', 'w') as f:
    #         json.dump(output_results, f)
        
    #     with open(eval_path / f'{args.ds}_{args.type}_errors.json', 'w') as f:
    #         json.dump(errors, f)

    # elif args.task == 'output_result_plus':
    #     # with open(eval_path / f'{args.ds}_{args.type}_eval.json') as f:
    #     #     output_results = json.load(f)
    #     # get_output_result_plus(output_results, f'{args.ds}_{args.type}_plus')
    #     # data_plus = load_plus_data(f'{args.ds}_{args.type}_plus')
    #     # df_eval = eval_all_dataset(data_plus)
    #     # df_eval.to_csv(eval_path / f'{args.ds}_{args.type}_eval_plus.csv', index=False)
    #     pass
    # else:
    #     raise ValueError(f'Unknown task: {args.task}, must be one of `zero_shot`, `post_process`, `output_result_plus`')