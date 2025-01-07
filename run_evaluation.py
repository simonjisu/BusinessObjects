import json
import pickle
import argparse
import sqlglot
import sqlglot.expressions as exp
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from typing import Any
from src.database import SqliteDatabase
from src.pymodels import DatabaseModel
from src.parsing_sql import Schema, extract_all
from src.eval_utils import (
    result_eq, 
    check_if_exists_orderby,
    get_complexity,
    get_all_partial_score

)
from src.data_preprocess import (
    load_raw_data, 
    process_all_tables,
    load_samples_spider_bird,
    
)

def load_predictions(prediction_path: Path, filename: str):
    predictions = []
    with open(prediction_path / filename, 'r') as f:
        preds = f.readlines()
        for p in preds:
            predictions.append(json.loads(p))
    return predictions

def get_target_parsed_sql(samples: list, tables: dict):
    error_ids = []
    parsed = defaultdict(dict)
    iterator = tqdm(samples, total=len(samples))
    for sample in iterator:
        db_id = sample.db_id
        sample_id = sample.sample_id
        iterator.set_description(f"{db_id}")
        schema = Schema(tables[db_id].db_schema)
        sql_i = sample.final.sql
        try:
            ei = extract_all(sql_i, schema)
            assert len(ei['sel']) > 0, f'No selection found-{db_id}-{sample_id}'
        except Exception as e:
            error_ids.append((db_id, sample_id, str(e)))
            parsed[db_id][sample_id] = None
            continue
        parsed[db_id][sample_id] = ei
    return parsed, error_ids

def get_prediction_parsed_sql(predictions: list, tables: dict):
    error_ids = []
    parsed = defaultdict(dict)
    iterator = tqdm(predictions, total=len(predictions))
    for pred in iterator:
        db_id = pred['db_id']
        pred_sql = pred['pred_sql']
        sample_id = pred['sample_id']
        iterator.set_description(f"{db_id}")
        schema = Schema(tables[db_id].db_schema)
        try:
            ei = extract_all(pred_sql, schema)
            assert len(ei['sel']) > 0, f'No selection found-{db_id}-{sample_id}'
        except Exception as e:
            error_ids.append((db_id, sample_id, str(e)))
            parsed[db_id][sample_id] = None
            continue
        parsed[db_id][sample_id] = ei
    return parsed, error_ids

def get_pred_results(
        proj_path: Path,
        eval_path: Path,
        predictions: list[dict],
        target_parsed: dict[str, dict[str, Any]],
        pred_parsed: dict[str, dict[str, Any]],
        tables: dict[str, DatabaseModel],
        file_name: str='[args.ds]_[args.type]',
        ds: str = 'bird', # spider or bird,
        split_k: int = 2
    ) -> tuple[list[dict], dict[str, list]]:
    
    error_infos = {
        'pred_exec': [],
        'gold_exec': [],
        'python_script': [],
        'result': []
    }

    processed_db_ids = [p.stem.split('_', split_k)[-1] for p in eval_path.glob(f'{file_name}_*.json')]
    # restart from checkpoint
    if processed_db_ids:
        predictions = [pred for pred in predictions if pred['db_id'] not in processed_db_ids]
        print(f'Skip some processed db_ids: {len(processed_db_ids)} {processed_db_ids[-5:]}')

    predictions_by_db_id = defaultdict(list)
    for pred in predictions:
        predictions_by_db_id[pred['db_id']].append(pred)
    
    for db_id, preds in predictions_by_db_id.items():
        output_results = []
        if ds == 'bird':
            try:
                database = SqliteDatabase(
                    db_file=str(proj_path / 'data' / ds / 'train' / 'train_databases' / db_id / f'{db_id}.sqlite'),
                    foreign_keys=tables[db_id].foreign_keys
                )
            except:
                database = SqliteDatabase(
                    db_file=str(proj_path / 'data' / ds / 'dev' / 'dev_databases' / db_id / f'{db_id}.sqlite'),
                    foreign_keys=tables[db_id].foreign_keys
                )
        else:
            database = SqliteDatabase(
                db_file=str(proj_path / 'data' / 'spider' / 'database' / db_id / f'{db_id}.sqlite'), 
                foreign_keys=tables[db_id].foreign_keys
            )
        
        iterator = tqdm(preds, total=len(preds))
        for pred in iterator:
            iterator.set_description(f'{db_id} | pred_exec: {len(error_infos["pred_exec"])} | gold_exec: {len(error_infos["gold_exec"])} | python_script: {len(error_infos["python_script"])} | result: {len(error_infos["result"])}')
            # Evaluate Structural and Semantic score
            sample_id = pred['sample_id']
            target_parsed_output = target_parsed[db_id][sample_id]
            gold_complexity = get_complexity(target_parsed_output)

            pred_parsed_output = pred_parsed[db_id][sample_id]
            if pred_parsed_output is None:
                structural_score = 0.0
                semantic_score = 0.0
                f1_score = 0.0
            else:
                _, all_score = get_all_partial_score(
                    source_output=pred_parsed_output,
                    target_output=target_parsed_output,
                    build_type='apted',
                    criteria='tsed',
                    penalty=0.01,
                    use_bert=True,
                    rescale_with_baseline=True
                )
                structural_score = all_score['structural']
                semantic_score = all_score['semantic']
                f1_score = all_score['overall']

            # Evaluate Execution Results
            pred_sql = pred['pred_sql'] 
            gold_sql = pred['gold_sql']
            
            error_info = ''
            try:
                pred_result = database.execute(pred_sql, rt_pandas=False)
            except Exception as e:
                pred_result = []
                error_infos['pred_exec'].append((db_id, sample_id))
                error_info = 'Predction Execution Error:' + str(e)
            
            try:
                gold_result = database.execute(gold_sql, rt_pandas=False)
            except Exception as e:
                error_infos['gold_exec'].append((db_id, sample_id))
                error_info = 'Gold Execution Error:' + str(e)
            
            if 'Gold Execution Error' in error_info:
                continue
            elif 'Predction Execution Error' in error_info:
                score = 0
            else:
                exists_orderby = check_if_exists_orderby(gold_sql)
                try:
                    score = int(result_eq(pred_result, gold_result, order_matters=exists_orderby))
                except Exception as e:
                    # print(f"An error occurred: {e}")
                    score = 0
                    error_info = 'Python Script Error:' + str(e)
                    error_infos['python_script'].append((db_id, sample_id))

                if score == 0 and error_info == '':
                    error_info = 'Result not equal'
                    error_infos['result'].append((db_id, sample_id))
            output_results.append(
                {
                    'sample_id': sample_id, 
                    'db_id': db_id,
                    'gold_complexity': gold_complexity,
                    'structural_score': structural_score,
                    'semantic_score': semantic_score,
                    'f1_score': f1_score,
                    'exec_result': score,
                }
            )

        with open(eval_path / f'{file_name}_{db_id}.json', 'w') as f:
            json.dump(output_results, f, indent=4)

    with open(proj_path / 'experiments' / ds / 'evals' / f'{file_name}_error_infos.json', 'w') as f:
        json.dump(error_infos, f, indent=4)

def get_pred_results_valid_bo(
        proj_path: Path,
        eval_path: Path,
        paths: list[Path],
        target_parsed: dict[str, dict[str, Any]],
        tables: dict[str, DatabaseModel],
        file_name: str='[args.ds]_[args.type]',
        ds: str = 'bird', # spider or bird,
    ) -> tuple[list[dict], dict[str, list]]:
    
    error_infos = {
        'pred_exec': [],
        'gold_exec': [],
        'python_script': [],
        'result': []
    }

    processed_files = [p.stem for p in eval_path.glob(f'{file_name}_*.json')]
    if processed_files:
        paths = dict([x for x in paths if x.stem not in processed_files])
        print(f'Skip some processed files: {len(processed_files)} {processed_files[-5:]}')

    pred_res = defaultdict(dict)  # db_id -> train_bo -> list[dict]

    for p in paths:
        output_results = []
        db_id = p.stem.split('_', 2)[-1].split('-')[0]

        if ds == 'bird':
            try:
                database = SqliteDatabase(
                    db_file=str(proj_path / 'data' / ds / 'train' / 'train_databases' / db_id / f'{db_id}.sqlite'),
                    foreign_keys=tables[db_id].foreign_keys
                )
            except:
                database = SqliteDatabase(
                    db_file=str(proj_path / 'data' / ds / 'dev' / 'dev_databases' / db_id / f'{db_id}.sqlite'),
                    foreign_keys=tables[db_id].foreign_keys
                )
        else:
            database = SqliteDatabase(
                db_file=str(proj_path / 'data' / 'spider' / 'database' / db_id / f'{db_id}.sqlite'), 
                foreign_keys=tables[db_id].foreign_keys
            )

        with p.open() as f:
            preds = json.load(f)  # list[dict]

        # get pred_parsed_sql 
        file_name = f'{p.stem}_parsed_pred.pkl'
        if not (eval_path / file_name).exists():
            pred_parsed, _ = get_prediction_parsed_sql(preds, tables)
            with open(eval_path / file_name, 'wb') as f:
                pickle.dump(pred_parsed, f)
            # print(f'Error parsing pred {args.ds}_{args.type}: {len(error_ids)}')

        with (eval_path / file_name).open('rb') as f:
            pred_parsed = pickle.load(f)

        iterator = tqdm(preds, total=len(preds))
        for pred in iterator:
            iterator.set_description(f'{db_id} | pred_exec: {len(error_infos["pred_exec"])} | gold_exec: {len(error_infos["gold_exec"])} | python_script: {len(error_infos["python_script"])} | result: {len(error_infos["result"])}')
            train_bo_id = pred['retrieved']
            if not pred_res[db_id].get(train_bo_id):
                pred_res[db_id][train_bo_id] = []
            
            # Evaluate Structural and Semantic score
            sample_id = pred['sample_id']
            target_parsed_output = target_parsed[db_id][sample_id]
            gold_complexity = get_complexity(target_parsed_output)

            pred_parsed_output = pred_parsed[db_id][sample_id]
            if pred_parsed_output is None:
                structural_score = 0.0
                semantic_score = 0.0
                f1_score = 0.0
            else:
                _, all_score = get_all_partial_score(
                    source_output=pred_parsed_output,
                    target_output=target_parsed_output,
                    build_type='apted',
                    criteria='tsed',
                    penalty=0.01,
                    use_bert=True,
                    rescale_with_baseline=True
                )
                structural_score = all_score['structural']
                semantic_score = all_score['semantic']
                f1_score = all_score['overall']

            # Evaluate Execution Results
            pred_sql = pred['pred_sql'] 
            gold_sql = pred['gold_sql']
            
            error_info = ''
            try:
                pred_result = database.execute(pred_sql, rt_pandas=False)
            except Exception as e:
                pred_result = []
                error_infos['pred_exec'].append((db_id, sample_id))
                error_info = 'Predction Execution Error:' + str(e)
            
            try:
                gold_result = database.execute(gold_sql, rt_pandas=False)
            except Exception as e:
                error_infos['gold_exec'].append((db_id, sample_id))
                error_info = 'Gold Execution Error:' + str(e)
            
            if 'Gold Execution Error' in error_info:
                continue
            elif 'Predction Execution Error' in error_info:
                score = 0
            else:
                exists_orderby = check_if_exists_orderby(gold_sql)
                try:
                    score = int(result_eq(pred_result, gold_result, order_matters=exists_orderby))
                except Exception as e:
                    # print(f"An error occurred: {e}")
                    score = 0
                    error_info = 'Python Script Error:' + str(e)
                    error_infos['python_script'].append((db_id, sample_id))

                if score == 0 and error_info == '':
                    error_info = 'Result not equal'
                    error_infos['result'].append((db_id, sample_id))
            
            # add score to the list
            output_results.append(
                {
                    'sample_id': sample_id, 
                    'db_id': db_id,
                    'retrieved': train_bo_id,
                    'gold_complexity': gold_complexity,
                    'structural_score': structural_score,
                    'semantic_score': semantic_score,
                    'f1_score': f1_score,
                    'exec_result': score,
                }
            )

        with open(eval_path / p.name, 'w') as f:
            json.dump(output_results, f, indent=4)

    # with open(proj_path / 'experiments' / ds / 'evals' / f'{file_name}_error_infos.json', 'w') as f:
    #     json.dump(error_infos, f, indent=4)

if __name__ == '__main__':
    """
    bird:
    python run_zeroshot.py --ds bird \
        --type train \
        --model gpt-4o-mini \
        --task zero_shot \
        --description_file bird_description.json \
    """
    parser = argparse.ArgumentParser(description='Zero-shot SQL generation with OpenAI')
    parser.add_argument('--ds', type=str, default='spider', help='Dataset to use for training. spider or bird') 
    parser.add_argument('--description_file', type=str, default='description.json', help='File containing the descriptions.')
    parser.add_argument('--type', type=str, default='dev', help='Type of data to use for. dev or test')
    parser.add_argument('--task', type=str, default='zero_shot', help='`zero_shot`, `zero_shot_hint`')
    args = parser.parse_args()

    proj_path = Path('.').resolve()
    assert proj_path.name == 'BusinessObjects', f'Expected project path to be BusinessObjects, but got {proj_path.name}'
    
    experiment_folder = proj_path / 'experiments' / args.ds
    prediction_path = experiment_folder / 'predictions' / args.task
    eval_path = experiment_folder / 'evals' / args.task
    for p in [prediction_path, eval_path]:
        if not p.exists():
            p.mkdir(parents=True)

    # must load components
    tables, *_ = load_raw_data(proj_path / 'data' / args.ds, load_test=False)
    with (proj_path / 'data' / args.description_file).open() as f:
        all_descriptions = json.load(f)
    tables = process_all_tables(tables, descriptions=all_descriptions)

    # sql predictions
    
    if args.task in ('zero_shot', 'zero_shot_hint'):
        predictions = []
        with open(prediction_path / f'final_{args.ds}_{args.type}.jsonl', 'r') as f:
            preds = f.readlines()
            for p in preds:
                predictions.append(json.loads(p))
    
        # get target_parsed_sql
        file_name = f'{args.ds}_{args.type}_parsed.pkl'
        samples = load_samples_spider_bird(proj_path / 'data' / f'{args.ds}_{args.type}.json')
        if not (eval_path / file_name).exists():
            target_parsed, error_ids = get_target_parsed_sql(samples, tables)
            with open(eval_path / file_name, 'wb') as f:
                pickle.dump(target_parsed, f)
            print(f'Error parsing target {args.ds}_{args.type}: {len(error_ids)}')

        with (eval_path / file_name).open('rb') as f:
            target_parsed = pickle.load(f)

        # get pred_parsed_sql 
        file_name = f'{args.ds}_{args.type}_parsed_pred.pkl'
        if not (eval_path / file_name).exists():
            pred_parsed, error_ids = get_prediction_parsed_sql(predictions, tables)
            with open(eval_path / file_name, 'wb') as f:
                pickle.dump(pred_parsed, f)
            print(f'Error parsing pred {args.ds}_{args.type}: {len(error_ids)}')

        with (eval_path / file_name).open('rb') as f:
            pred_parsed = pickle.load(f)

        get_pred_results(
            proj_path,
            eval_path,
            predictions, 
            target_parsed,
            pred_parsed,
            tables,
            file_name=f'{args.ds}_{args.type}',
            ds=args.ds,
            split_k=2
        )

        # save it as dataframe
        df = []
        for p in eval_path.glob(f'{args.ds}_{args.type}_*.json'):
            with p.open() as f:
                eval_data = json.load(f)
            df.extend(eval_data)
        pd.DataFrame(df).to_csv(eval_path / f'{args.ds}_{args.type}.csv', index=False)

    elif args.task == 'valid_bo':
        # valid_bo
        paths = list(prediction_path.glob(f'{args.ds}_{args.type}_*.jsonl'))
        print(f'Found {len(paths)} files')
        # get target_parsed_sql
        file_name = f'{args.ds}_{args.type}_parsed.pkl'
        samples = load_samples_spider_bird(proj_path / 'data' / f'{args.ds}_{args.type}.json')
        if not (eval_path / file_name).exists():
            target_parsed, error_ids = get_target_parsed_sql(samples, tables)
            with open(eval_path / file_name, 'wb') as f:
                pickle.dump(target_parsed, f)
            print(f'Error parsing target {args.ds}_{args.type}: {len(error_ids)}')

        with (eval_path / file_name).open('rb') as f:
            target_parsed = pickle.load(f)

        get_pred_results_valid_bo(
            proj_path,
            eval_path,
            paths,
            target_parsed,
            tables,
            file_name=f'{args.ds}_{args.type}',
            ds=args.ds,
        )
    else:
        raise ValueError(f'Invalid task: {args.task}')