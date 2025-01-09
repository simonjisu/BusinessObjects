import json
import pickle
import argparse
import sqlglot
import sqlglot.expressions as exp
import numpy as np
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
    get_all_partial_score,
    get_all_semantic_score,
    get_all_structural_score

)
from src.data_preprocess import (
    load_raw_data, 
    process_all_tables,
    load_samples_spider_bird,   
)
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from run_bo_sql import _get_categories, _format_interval

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

def get_prediction_parsed_sql(predictions: list, tables: dict, patch_db_id: str=None):
    error_ids = []
    parsed = defaultdict(dict)
    iterator = tqdm(predictions, total=len(predictions))
    for pred in iterator:
        db_id = pred['db_id'] if patch_db_id is None else patch_db_id
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

        preds = [x for x in preds if pred_parsed[db_id].get(x['sample_id'])]
        target_parsed_outputs = [target_parsed[db_id][x['sample_id']] for x in preds]
        pred_parsed_outputs = [pred_parsed[db_id][x['sample_id']] for x in preds]
        
        structural_scores = get_all_structural_score(pred_parsed_outputs, target_parsed_outputs)
        semantic_scores = get_all_semantic_score(pred_parsed_outputs, target_parsed_outputs)

        epsilon = 1e-9
        structural_scores = np.array(structural_scores)
        semantic_scores = np.array(semantic_scores)
        f1_scores = 2 * (structural_scores * semantic_scores) / (structural_scores + semantic_scores + epsilon)
        
        iterator = tqdm(enumerate(preds), total=len(preds))
        for k, pred in iterator:
            iterator.set_description(f'{db_id} | pred_exec: {len(error_infos["pred_exec"])} | gold_exec: {len(error_infos["gold_exec"])} | python_script: {len(error_infos["python_script"])} | result: {len(error_infos["result"])}')
            train_bo_id = pred['retrieved']  # list
            
            # Evaluate Structural and Semantic score
            sample_id = pred['sample_id']
            target_parsed_output = target_parsed[db_id][sample_id]
            gold_complexity = get_complexity(target_parsed_output)

            # pred_parsed_output = pred_parsed[db_id][sample_id]
            # if pred_parsed_output is None:
            #     structural_score = 0.0
            #     semantic_score = 0.0
            #     f1_score = 0.0
            # else:
            #     _, all_score = get_all_partial_score(
            #         source_output=pred_parsed_output,
            #         target_output=target_parsed_output,
            #         build_type='apted',
            #         criteria='tsed',
            #         penalty=0.01,
            #         use_bert=True,
            #         rescale_with_baseline=True
            #     )
            #     structural_score = all_score['structural']
            #     semantic_score = all_score['semantic']
            #     f1_score = all_score['overall']

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
                    'structural_score': float(structural_scores[k]),
                    'semantic_score': float(semantic_scores[k]),
                    'f1_score': float(f1_scores[k]),
                    'exec_result': score,
                }
            )

        with open(eval_path / f'{file_name}_{db_id}.json', 'w') as f:
            json.dump(output_results, f, indent=4)

    # with open(proj_path / 'experiments' / ds / 'evals' / f'{file_name}_error_infos.json', 'w') as f:
    #     json.dump(error_infos, f, indent=4)

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
        paths = [x for x in paths if x.stem not in processed_files]
        print(f'Skip some processed files: {len(processed_files)} {processed_files[-5:]}')

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

        if not preds:
            continue
        # get pred_parsed_sql 
        file_name = f'{p.stem}_parsed_pred.pkl'
        if not (eval_path / file_name).exists():
            pred_parsed, _ = get_prediction_parsed_sql(preds, tables, patch_db_id=db_id)
            with open(eval_path / file_name, 'wb') as f:
                pickle.dump(pred_parsed, f)
        with (eval_path / file_name).open('rb') as f:
            pred_parsed = pickle.load(f)

        # calculate structural, semantic and f1 scores
        preds = [x for x in preds if pred_parsed[db_id].get(x['sample_id'])]
        target_parsed_outputs = [target_parsed[db_id][x['sample_id']] for x in preds]
        pred_parsed_outputs = [pred_parsed[db_id][x['sample_id']] for x in preds]
        
        structural_scores = get_all_structural_score(pred_parsed_outputs, target_parsed_outputs)
        semantic_scores = get_all_semantic_score(pred_parsed_outputs, target_parsed_outputs)

        epsilon = 1e-9
        structural_scores = np.array(structural_scores)
        semantic_scores = np.array(semantic_scores)
        f1_scores = 2 * (structural_scores * semantic_scores) / (structural_scores + semantic_scores + epsilon)

        
        iterator = tqdm(enumerate(preds), total=len(preds))
        for k, pred in iterator:
            iterator.set_description(f'{p.stem} | pred_exec: {len(error_infos["pred_exec"])} | gold_exec: {len(error_infos["gold_exec"])} | python_script: {len(error_infos["python_script"])} | result: {len(error_infos["result"])}')
            train_bo_id = pred['retrieved']  # list
            
            # Evaluate Structural and Semantic score
            sample_id = pred['sample_id']
            target_parsed_output = target_parsed[db_id][sample_id]
            gold_complexity = get_complexity(target_parsed_output)

            # Evaluate Execution Results
            pred_sql = pred['pred_sql'] 
            gold_sql = pred['gold_sql']
            
            error_info = ''
            # pred_sql = pred_sql.split('LIMIT')[0].split('limit')[0] + ' LIMIT 100;'
            # gold_sql = gold_sql.split('LIMIT')[0].split('limit')[0] + ' LIMIT 100;'
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
                    'structural_score': float(structural_scores[k]),
                    'semantic_score': float(semantic_scores[k]),
                    'f1_score': float(f1_scores[k]),
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
    parser.add_argument('--s', type=int, default=-1, help='start')
    parser.add_argument('--e', type=int, default=-1, help='end')
    parser.add_argument('--scenario', type=int, default=-1, help='Scenario to consider')
    
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
        filter_db_ids = ['bike_share_1', 'language_corpus', 'donor', 'menu', 
                         'movie_platform', 'talkingdata', 'authors', 'image_and_language']
        file_post_fix = f'{args.ds}_{args.type}' if args.scenario < 0 else f'{args.ds}_{args.type}_{args.scenario}'
        split_k = 2 if args.scenario < 0 else 3
        samples = load_samples_spider_bird(proj_path / 'data' / f'{args.ds}_{args.type}.json')
        
        final_file = f'final_{file_post_fix}.json'
        if not (prediction_path / final_file).exists():
            all_results = []
            paths = sorted(list(prediction_path.glob(f'{file_post_fix}_*.json')))
            for p in paths:
                with p.open() as f:
                    results = json.load(f)
                
                for r in results:
                    r.pop('rationale')
                    r['db_id'] = p.stem.split('_', split_k)[-1]

                    found = False
                    for s in samples:
                        if r['sample_id'] == s.sample_id:
                            found = True
                            break
                    r['gold_sql'] = s.final.sql
                    assert found, r['sample_id']

                all_results.extend(results)
            with open(prediction_path / final_file, 'w') as f:
                json.dump(all_results, f, indent=4)

        with open(prediction_path / final_file, 'r') as f:
            preds = json.load(f)
        # filter samples with db_id
        res_db_ids = {r['db_id'] for r in preds}
        # filter db_ids 
        res_db_ids = res_db_ids - set(filter_db_ids)
        print(f'Found {len(res_db_ids)} db_ids')
        preds = [p for p in preds if p['db_id'] in res_db_ids]
        samples = [s for s in samples if s.db_id in res_db_ids]
        
        # get target_parsed_sql
        file_name = f'{file_post_fix}_parsed.pkl'  
        if not (eval_path / file_name).exists():
            target_parsed, _ = get_target_parsed_sql(samples, tables)
            with open(eval_path / file_name, 'wb') as f:
                pickle.dump(target_parsed, f)

        with (eval_path / file_name).open('rb') as f:
            target_parsed = pickle.load(f)

        # get pred_parsed_sql 
        file_name = f'{file_post_fix}_parsed_pred.pkl'
        if not (eval_path / file_name).exists():
            pred_parsed, _ = get_prediction_parsed_sql(preds, tables)
            with open(eval_path / file_name, 'wb') as f:
                pickle.dump(pred_parsed, f)

        with (eval_path / file_name).open('rb') as f:
            pred_parsed = pickle.load(f)

        # filter samples with pred_parsed
        error_ids = [sample_id for _, sample_id_ast in pred_parsed.items() for sample_id, ast in sample_id_ast.items() if not ast]
        preds = [p for p in preds if p['sample_id'] not in error_ids]
        samples = [s for s in samples if s.sample_id not in error_ids]

        get_pred_results(
            proj_path,
            eval_path,
            preds, 
            target_parsed,
            pred_parsed,
            tables,
            file_name=file_post_fix,
            ds=args.ds,
            split_k=split_k
        )

        # save it as dataframe
        df = []
        for p in eval_path.glob(f'{file_post_fix}_*.json'):
            with p.open() as f:
                eval_data = json.load(f)
            df.extend(eval_data)
        pd.DataFrame(df).to_csv(eval_path / f'{file_post_fix}.csv', index=False)

    elif args.task == 'valid_bo':
        filter_db_ids = ['bike_share_1', 'language_corpus', 'donor', 'menu', 
                         'movie_platform', 'talkingdata', 'authors']
        # valid_bo
        paths = sorted(list(prediction_path.glob(f'{args.ds}_{args.type}_*.json')))
        paths = [p for p in paths if p.stem.split('_', 2)[-1].split('-')[0] not in filter_db_ids]
        if args.s >= 0 and args.e >= 0:
            paths = paths[args.s:args.e]
        
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
    elif args.task == 'cal_merits':
        # dev or test
        df_base = pd.read_csv(experiment_folder / 'evals' / 'zero_shot' / f'{args.ds}_{args.type}.csv')
        df_bo = pd.read_csv(experiment_folder / 'evals' / 'valid_bo' / f'{args.ds}_{args.type}.csv')
        df_cates = df_base.groupby('db_id')['gold_complexity'].apply(_get_categories).rename('category').apply(_format_interval)
        df_base = pd.merge(df_base, df_cates.reset_index('db_id', drop=True), left_index=True, right_index=True)

        df = pd.merge(
            left=df_bo,
            right=df_base,
            how='inner',
            on=['db_id', 'sample_id', 'gold_complexity'],
            suffixes=('_bo', '_base')
        )

        group_column = ['db_id', 'retrieved'] # , 
        execution_improvement = df.groupby(group_column)[['exec_result_base', 'exec_result_bo']].sum().diff(axis=1)['exec_result_bo'].rename('execution_improvement')
        merit_structural = df.groupby(group_column)[['structural_score_base', 'structural_score_bo']].mean().diff(axis=1)['structural_score_bo'].rename('merit_structural')
        merit_semantic = df.groupby(group_column)[['semantic_score_base', 'semantic_score_bo']].mean().diff(axis=1)['semantic_score_bo'].rename('merit_semantic')
        merit = df.groupby(group_column)[['f1_score_base', 'f1_score_bo']].mean().diff(axis=1)['f1_score_bo'].rename('merit')

        ranks = merit.reset_index().groupby(['db_id'])['merit'].rank(method='first', ascending=False).rename('rank').astype(np.int64)
        merit = pd.concat([merit.reset_index(), ranks], axis=1)
        merit_by_rank = merit.sort_values(by=['db_id', 'rank'], ascending=True)

        merit_by_rank.to_csv(experiment_folder / 'evals' / f'merits_{args.ds}_{args.type}.csv', index=False)

        # create scenarios
        test_bos = defaultdict(list)
        for x in merit_by_rank.loc[:, ['db_id', 'retrieved']].to_dict(orient='records'):
            test_bos[x['db_id']].append(x['retrieved'])

        n_bos = range(5, 26, 5)
        test_scenarios = defaultdict(dict)
        for n_bo in n_bos:
            for db_id in test_bos:
                test_scenarios[n_bo][db_id] = test_bos[db_id][:n_bo]

        with (experiment_folder / 'test_scenarios.json').open('w') as f:
            json.dump(test_scenarios, f)

    else:
        raise ValueError(f'Invalid task: {args.task}')