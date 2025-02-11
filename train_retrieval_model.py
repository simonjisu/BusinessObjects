from sentence_transformers import (
    SentenceTransformer, 
    SentenceTransformerTrainer, 
    SentenceTransformerTrainingArguments, 
    losses
)
from datasets import Dataset
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
import multiprocessing as mp
import pandas as pd
import json
from pathlib import Path
import argparse
from src.data_preprocess import (
    load_raw_data,
    process_all_tables,
)
import random

class RetrievalDataset():
    def __init__(self, data: dict, is_train: bool=True):
        self.samples = self._map_samples_key_to_int(data['samples'])
        self.sample_ids = self._align_sample_ids(data['sample_ids'])
        self.is_train = is_train
        
    def __len__(self):
        return len(self.sample_ids)
    
    def _map_samples_key_to_int(self, samples: dict[str, dict[int|str, str]]):
        samples['question'] = {int(k): v for k, v in samples['question'].items()}
        samples['ba'] = {int(k): v for k, v in samples['ba'].items()}
        return samples

    def _align_sample_ids(self, sample_ids: list[dict[str, int|list[int]]]):
        # {'pos: pos, 'neg': [neg1, neg2, ...]}
        # -> list of [pos, neg] to make (anchor, positive, negative)
        flatten_sample_ids = []
        for x in sample_ids:
            pos_id = int(x['pos'])
            neg_ids = [int(i) for i in x['neg']]
            if self.is_train:
                flatten_sample_ids.append({'pos': pos_id, 'neg': random.choice(neg_ids)})
            else:
                for neg_id in neg_ids:
                    flatten_sample_ids.append({'pos': pos_id, 'neg': neg_id})

        return flatten_sample_ids
        
    def __getitem__(self, idx):
        x = self.sample_ids[idx]
        question = self.samples['question'][x['pos']]
        pos_ba = self.samples['ba'][x['pos']]
        neg_ba = self.samples['ba'][x['neg']]

        return {
            'anchor': question,
            'positive': pos_ba,
            'negative': neg_ba
        }
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def run_parallel(func, args, num_cpus: int=1):
    pool = mp.Pool(processes=num_cpus)
    results = pool.starmap(func, args)
    pool.close()
    pool.join()
    return results


def get_hard_negative_samples(df: pd.DataFrame, n_neg_each_db: int=1):
    db_ids = df['db_id'].unique()
    samples = df.loc[:, ['sample_id', 'question', 'ba']].set_index('sample_id').to_dict('dict')  # question, ba key -- {sample_id: value}
    sample_ids = []

    for db_id in db_ids:
        positive_samples = df.loc[df['db_id'] == db_id, 'sample_id'].tolist()
        for pos in positive_samples:
            negative_samples = df.loc[df['db_id'] != db_id, ['sample_id', 'db_id']].groupby(['db_id']).sample(n=n_neg_each_db)['sample_id'].tolist()
            sample_ids.append({'pos': pos, 'neg': negative_samples})  # anchor: pos-question | positive: pos-ba | negative: neg-ba

    return {'samples': samples, 'sample_ids': sample_ids}

def split_train_dev_retrieval_data(train_bo_path: Path|str, frac: float=0.9, n_qcut: int =5, n_neg_each_db: int=1, random_state: int=42, num_cpus: int=1):
    with open(train_bo_path, 'r') as f:
        train_bo = json.load(f)
    # create dataframe
    df = []
    for db_id, xs in train_bo.items():
        for x in xs:
            x['db_id'] = db_id
            df.append(x)

    df = pd.DataFrame(df)
    cates = pd.qcut(df['gold_complexity'], q=n_qcut)
    df['gold_complexity_cates'] = cates
    df['gold_complexity_codes'] = cates.cat.codes

    # split train and dev by equal complexity distribution
    df_train = df.groupby('gold_complexity_codes').sample(frac=frac, random_state=random_state)
    df_dev = df.drop(df_train.index)

    train_data = get_hard_negative_samples(df_train, n_neg_each_db)
    dev_data = get_hard_negative_samples(df_dev, n_neg_each_db)
    # train_data, dev_data = run_parallel(get_hard_negative_samples, [(df_train, n_neg_each_db), (df_dev, n_neg_each_db)], num_cpus=num_cpus)

    return {'train': train_data, 'dev': dev_data}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train retrieval model')
    parser.add_argument('--task', type=str, default='retrieval', help='`retrieval`, `data_prep`')
    parser.add_argument('--ds', type=str, default='bird', help='Dataset to use for training. spider or bird') 
    parser.add_argument('--num_cpus', type=int, default=1, help='Number of cpus to use for parallel processing')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--per_device_train_batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--logging_steps', type=int, default=10, help='Logging steps')
    
    args = parser.parse_args()
    proj_path = Path('.').resolve()
    description_file = f'description.json' if args.ds == 'spider' else f'{args.ds}_description.json'

    experiment_folder = proj_path / 'experiments' / args.ds

    eval_path = experiment_folder / 'evals' / args.task
    prediction_path = experiment_folder / 'predictions' / args.task

    tables, *_ = load_raw_data(proj_path / 'data' / args.ds, load_test=False)
    with (proj_path / 'data' / description_file).open() as f:
        all_descriptions = json.load(f)
    tables = process_all_tables(tables, descriptions=all_descriptions)


    # data preparation
    if args.task == 'data_prep':
        assert (experiment_folder / 'predictions' / 'create_bo' / f'final_{args.ds}_train_bo.json').exists(), f'Please create bo first to generate the data'
        data = split_train_dev_retrieval_data(experiment_folder / 'predictions' / 'create_bo' / f'final_{args.ds}_train_bo.json', num_cpus=args.num_cpus)
        with open(experiment_folder / 'predictions' / 'create_bo' / f'hard_negative.json', 'w') as f:
            json.dump(data, f)

    elif args.task == 'retrieval':
        # train retrieval model
        with open(experiment_folder / 'predictions' / 'create_bo' / f'hard_negative.json', 'r') as f:
            data = json.load(f)

        model = SentenceTransformer('all-mpnet-base-v2')

        train_ds = Dataset.from_generator(RetrievalDataset(data['train'], is_train=True).__iter__)
        dev_ds = Dataset.from_generator(RetrievalDataset(data['dev'], is_train=False).__iter__)

        exp_name = 'all-mpnet-base-v2-q_ba'

        args = SentenceTransformerTrainingArguments(
            output_dir=f'models/{exp_name}',
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_train_batch_size,
            warmup_ratio=0.1,
            logging_steps=args.logging_steps,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            eval_strategy="steps",
            torch_empty_cache_steps=100,
            save_strategy="steps",
            save_steps=args.logging_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            logging_dir=f'logs/{exp_name}',
        )
        loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.per_device_train_batch_size // 2)

        dev_evaluator = TripletEvaluator(
            anchors=dev_ds['anchor'],
            positives =dev_ds['positive'],
            negatives=dev_ds['negative'],
            name='q_ba_dev',
        )

        trainer = SentenceTransformerTrainer(
            model=model,
            train_dataset=train_ds,
            loss=loss,
            args=args,
            eval_dataset=dev_ds,
            evaluator=dev_evaluator,
        )

        trainer.train()
