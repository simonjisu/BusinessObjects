from bert_score import score as bscore
from transformers import logging as tfloggings
tfloggings.set_verbosity_error()
import warnings
from pathlib import Path
import json
from tqdm import tqdm 
from collections import defaultdict
from run_bo_sql import _get_categories, _format_interval
import pandas as pd
from src.data_preprocess import load_samples_spider_bird

def get_categories(df_base, df_bo):
    df_cates = df_base.groupby('db_id')['gold_complexity'].apply(_get_categories).rename('category').apply(_format_interval)
    df_base = pd.merge(df_base, df_cates.reset_index('db_id', drop=True), left_index=True, right_index=True)

    df = pd.merge(
        left=df_bo,
        right=df_base,
        how='inner',
        on=['db_id', 'sample_id', 'gold_complexity'],
        suffixes=('_bo', '_base')
    )

    return df

def get_merits(df: pd.DataFrame):
    group_column = ['db_id', 'sample_id', 'category'] # , 
    execution_improvement = df.groupby(group_column)[['exec_result_base', 'exec_result_bo']].sum().diff(axis=1)['exec_result_bo'].rename('execution_improvement')
    merit_structural = df.groupby(group_column)[['structural_score_base', 'structural_score_bo']].mean().diff(axis=1)['structural_score_bo'].rename('merit_structural')
    merit_semantic = df.groupby(group_column)[['semantic_score_base', 'semantic_score_bo']].mean().diff(axis=1)['semantic_score_bo'].rename('merit_semantic')
    merit = df.groupby(group_column)[['f1_score_base', 'f1_score_bo']].mean().diff(axis=1)['f1_score_bo'].rename('merit')

    return {
        'execution_improvement': execution_improvement,
        'merit_structural': merit_structural,
        'merit_semantic': merit_semantic,
        'merit': merit
    }

ds = 'spider'
task = 'zero_shot_hint'
typ = 'test'
proj_path = Path('.').resolve()

experiment_folder = proj_path / 'experiments' / ds
eval_path = experiment_folder / 'evals' / task
# load test data
df_base = pd.read_csv(experiment_folder / 'evals' / 'zero_shot' / f'{ds}_test.csv')
df_0 = pd.read_csv(eval_path / f'{ds}_{typ}_0.csv')
df_1 = pd.read_csv(eval_path / f'{ds}_{typ}_1.csv')
df_2 = pd.read_csv(eval_path / f'{ds}_{typ}_2.csv')
df_3 = pd.read_csv(eval_path / f'{ds}_{typ}_3.csv')
common_ids = set(df_base['sample_id']) & set(df_0['sample_id']) & set(df_1['sample_id']) & set(df_2['sample_id']) & set(df_3['sample_id'])
test_samples = load_samples_spider_bird(proj_path / 'data' / f'{ds}_test.json')
test_samples = [s for s in test_samples if s.sample_id in common_ids]

experiment_folder = proj_path / 'experiments' / ds
# question similarity
scenario = 0
bo_path = experiment_folder / 'predictions' / 'create_bo' / f'final_{ds}_train_bo.json'
with bo_path.open() as f:
    all_bos = json.load(f)
sce = {0: "10", 1: "15", 2: "25", 3: "25"}[scenario]

# test_scenarios
with (experiment_folder / 'test_scenarios.json').open('r') as f:
    test_scenarios = json.load(f)

test_bo_ids = test_scenarios[sce]
test_bos = defaultdict(list)
for db_id, bos in all_bos.items():
    if db_id in test_bo_ids:
        bo_ids = test_bo_ids[db_id]
        test_bos[db_id].extend(list(filter(lambda x: x['sample_id'] in bo_ids, bos)))

# question similarity
q_pairs = defaultdict(list)
ba_pairs = defaultdict(list)
sample_ids = defaultdict(list)
for test_sample in test_samples:
    for bo in test_bos[test_sample.db_id]:
        sample_ids[test_sample.db_id].append((test_sample.sample_id, bo['sample_id']))
        q_pairs[test_sample.db_id].append((test_sample.final.question, bo['question']))
        ba_pairs[test_sample.db_id].append((test_sample.final.question, bo['ba']))

scores = []
db_ids = list(q_pairs.keys())
iterator = tqdm(db_ids, total=len(db_ids))
for db_id in iterator:
    sample_id, bo_id = zip(*sample_ids[db_id])
    qs, qt = zip(*q_pairs[db_id])
    bas, bat = zip(*ba_pairs[db_id])
    with warnings.catch_warnings(action='ignore'):
        iterator.set_description(f'{db_id} - q')
        *_, f1_q = bscore(qs, qt, lang='en', rescale_with_baseline=True, device='cuda')
        iterator.set_description(f'{db_id} - ba')
        *_, f1_ba = bscore(bas, bat, lang='en', rescale_with_baseline=True, device='cuda')
    
    for q, ba, sid, bid in zip(f1_q.numpy().tolist(), f1_ba.numpy().tolist(), sample_id, bo_id):
        scores.append(
            {
                'db_id': db_id,
                'sample_id': sid,
                'bo_id': bid,
                'f1_q': q,
                'f1_ba': ba
            }
        )

pd.DataFrame(scores).to_csv(experiment_folder / 'question_ba_similarity.csv', index=False)
# with open(experiment_folder / 'question_ba_similarity.pkl', 'wb') as f:
#     pickle.dump(scores, f)