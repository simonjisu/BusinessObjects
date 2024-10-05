from tqdm import tqdm
from pathlib import Path
from process_sql import get_schema, Schema
from parsing_sql import extract_aliases, extract_selection, extract_condition, extract_aggregation, extract_nested_setoperation, extract_others
import json
import sqlparse

proj_path = Path(__file__).resolve().parents[1]

def get_output_result_plus(output_result, filename: str):
    output_results_plus = []
    errors = []
    for x in tqdm(output_result):
        has_error = False
        schema = get_schema(str(proj_path / 'data' / 'spider' / 'database' / x['db_id'] / f'{x["db_id"]}.sqlite'))
        schema = Schema(schema)
        
        for s in ['gold', 'pred']:
            try:
                sql = x[f'{s}_sql']
                statement = sqlparse.parse(sql.strip())[0]
                aliases = extract_aliases(statement)
                selection = extract_selection(statement, aliases, schema)
                condition = extract_condition(statement)
                aggregation = extract_aggregation(statement, aliases, schema)
                nested = extract_nested_setoperation(statement)
                others = extract_others(statement, aliases, schema)
                
                x[s + '_selection'] = list(map(list, selection))
                x[s + '_condition'] = (condition[0], list(condition[1]))
                x[s + '_aggregation'] = list(map(list, aggregation))
                x[s + '_nested'] = nested
                x[s + '_others'] = {'distinct': list(others['distinct']), 
                                    'order by': list(others['order by']), 
                                    'limit': others['limit']}
            except Exception as e:
                has_error = True
                errors.append((x['sample_id'], s, str(e)))
                break
    
        if not has_error:
            output_results_plus.append(x)

    with open(proj_path / 'experiments' / 'evals' / f'{filename}.json', 'w') as f:
        # json.dumps(output_results_plus, f)
        for x in output_results_plus:
            l = json.dumps(x)
            f.write(l + '\n')

    with open(proj_path / 'experiments' / 'evals' / f'{filename}_errors.json', 'w') as f:
        json.dump(errors, f)

    print(f'Report: {len(output_results_plus)}/{len(output_result)} - errors = {len(errors)}')

    return output_results_plus

def reverse_mapping(x: dict):
    def post(li: list):
        return [c for c in li if c != '']
    
    for s in ['gold', 'pred']:
        x[f'{s}_selection'] = set(x[f'{s}_selection'][0]), set(list(map(tuple, x[f'{s}_selection'][1])))
        x[f'{s}_condition'] = (set(sorted(post(x[f'{s}_condition'][0]))), set(post(x[f'{s}_condition'][1])))
        x[f'{s}_aggregation'] = set(x[f'{s}_aggregation'][0]), set(list(map(tuple, x[f'{s}_aggregation'][1])))
        x[f'{s}_others']['distinct'] = set(x[f'{s}_others']['distinct'])
        x[f'{s}_others']['order by'] = set(x[f'{s}_others']['order by'])
    return x

def load_plus_data(filename: str):
    data = []
    with open(proj_path / 'experiments' / 'evals' / f'{filename}.json', 'r') as f:
        for l in f:
            x = reverse_mapping(json.loads(l))
            data.append(x)
    return data


if __name__ == '__main__':
    with open(proj_path / 'experiments' / 'evals' / 'spider_train_eval.json', 'r') as f:
        train_output_results = json.load(f)

    with open(proj_path / 'experiments' / 'evals' / 'spider_dev_eval.json', 'r') as f:
        dev_output_results = json.load(f)

    train_plus = get_output_result_plus(train_output_results, 'spider_train_plus')
    dev_plus = get_output_result_plus(dev_output_results, 'spider_dev_plus')

    train_plus = load_plus_data('spider_train_plus')
    dev_plus = load_plus_data('spider_dev_plus')





























# import nltk
# nltk.download('punkt_tab')
# import pandas as pd 
# from src.process_sql import get_sql, get_schema, Schema
# from src.eval import get_nestedSQL, Evaluator
# import sqlglot

# evaluator = Evaluator()

# def is_nested(x: pd.Series, keyword: str, proj_path: Path) -> bool:
#     schema = get_schema(str(proj_path / 'data' / 'spider' / 'database' / x['db_id'] / f'{x["db_id"]}.sqlite'))
#     schema = Schema(schema)
#     sql = get_sql(schema, x['gold_sql'])
#     if keyword in ('intersect', 'except', 'union'):
#         if sql[keyword] is not None:
#             return True
#     else:
#         conds = sql[keyword]['conds'][::2] if keyword == 'from' else sql[keyword][::2]
#         for cond_unit in conds:
#             if type(cond_unit[3]) is dict:  # val1
#                 return True
#             if type(cond_unit[4]) is dict:  # val2
#                 return True
#     return False

# def get_count(sql: dict, category: str, count: int):
#     if category == 'selection':
#         # total number of columns used in the query
#         count += len(sql['select'][1])
#     elif category == 'condition':
#         count += (len(sql['where'][::2]) + len(sql['having'][::2]))
#     elif category == 'aggregation':
#         count += len(sql['groupBy'])
#     elif category == 'ordering':
#         count += 1 if sql['orderBy'] else 0
#     elif category == 'limitation':
#         count += 1 if sql['limit'] is not None else 0
#     else:
#         raise ValueError(f'Category {category} is not supported.')
#     return count 

# def get_number_of_components(x: pd.Series, category: str, proj_path: Path) -> int:
#     schema = get_schema(str(proj_path / 'data' / 'spider' / 'database' / x['db_id'] / f'{x["db_id"]}.sqlite'))
#     schema = Schema(schema)
#     sql = get_sql(schema, x['gold_sql'])
#     count = 0
#     count = get_count(sql, category, count)
#     if x['is_nested']:
#         nested_sql = get_nestedSQL(sql)
#         for nested in nested_sql:
#             count = get_count(nested, category, count)
#     return count

# def eval_hardness(x: pd.Series, proj_path: Path) -> str:
#     schema = get_schema(str(proj_path / 'data' / 'spider' / 'database' / x['db_id'] / f'{x["db_id"]}.sqlite'))
#     schema = Schema(schema)
#     sql = get_sql(schema, x['gold_sql'])
#     return evaluator.eval_hardness(sql)

# def process_results(output_results, proj_path: Path, evaluator: Evaluator) -> pd.DataFrame:
#     df = pd.DataFrame(output_results)
#     df['tbls'] = df['source_tables'].str.join(',')
#     df['len_tbls'] = df['source_tables'].apply(len)
#     for keyword in ('from', 'where', 'having', 'intersect', 'except', 'union'):
#         df[f'is_nested_{keyword}'] = df.apply(is_nested, keyword=keyword, proj_path=proj_path, axis=1)
#     df['is_nested'] = df[[f'is_nested_{keyword}' for keyword in ('from', 'where', 'having', 'intersect', 'except', 'union')]].any(axis=1)
    
#     for category in ('selection', 'condition', 'aggregation', 'ordering', 'limitation'):
#         df[category] = df.apply(get_number_of_components, category=category, proj_path=proj_path, axis=1)
    
#     df['hardness'] = df.apply(eval_hardness, proj_path=proj_path, axis=1)
#     return df

# df_train_results = process_results(train_output_results, proj_path, evaluator)
# df_dev_results = process_results(dev_output_results, proj_path, evaluator)
# df_train_results.to_csv(eval_path / 'spider_train_eval.csv', index=False)
# df_dev_results.to_csv(eval_path / 'spider_dev_eval.csv', index=False)
# print(f'pred_exec: {len(train_errors["pred_exec"])} | gold_exec: {len(train_errors["gold_exec"])} | python_script: {len(train_errors["python_script"])} | result: {len(train_errors["result"])}')
# print(f'pred_exec: {len(dev_errors["pred_exec"])} | gold_exec: {len(dev_errors["gold_exec"])} | python_script: {len(dev_errors["python_script"])} | result: {len(dev_errors["result"])}')