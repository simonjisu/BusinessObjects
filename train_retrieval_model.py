import json
import random
import argparse
import warnings
import pandas as pd
import multiprocessing as mp
import numpy as np

from tqdm import tqdm
from pathlib import Path
from itertools import product, combinations
from torch.utils.data import DataLoader
from collections import defaultdict

from src.data_preprocess import (
    load_raw_data,
    process_all_tables,
)

from sentence_transformers import (
    SentenceTransformer, 
    SentenceTransformerTrainer, 
    SentenceTransformerTrainingArguments, 
    losses,
    CrossEncoder,
    InputExample
)
from datasets import Dataset
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator

from bert_score import score as bscore
from transformers import logging as tfloggings
tfloggings.set_verbosity_error()

from typing import Optional

class BADataset():
    def __init__(
            self, 
            data: dict[str, dict[str, str|dict[str, int]]],
            bert_scores: dict[str, float],
            task: str, 
            n_neg: int=1,
            threshold: float=0.1,
        ):
        """
        samples: dict[str, dict[int|str, str]] = {
            'question': {sample_id: question, ...},
            'ba': {sample_id: ba, ...},
            'gold_complexity_codes': {sample_id: gold_complexity_codes, ...}
        }
        bert_scores: dict[str, float] = {
            'sample_id1-sample_id2': score, ...  # intra-db sample pairs
        }

        # task: retrieval
        Instead of using MultipleNegativesRankingLoss we use TripletLoss, since most samples are similar. 
        So we need to design triplets that negative samples are from 
        1. same databases: harder to train (avoid ba is too similar to positive with certain threshold)
        2. different databases: easier to train

        construct: (anchor, positive, negative) 
        - anchor: question
        - positive: ba
        - negative: ba (`n_neg` from same database and `n_neg` from different database)
        sample number: `n_neg` (same database) + `n_neg` (different database)
        
        # task: rerank
        We train a binary classifier to predict whether the question and ba are relevant (share the complexity code) or not.
        If we do random sampling, it will be too easy for the model to predict the label.
        So we need sample negative samples from:
        1. same database: harder to train
        2. different database: easier to train

        construct: (text, label)
        - text: question, ba
        - label: gold_complexity_codes
        sample number: 5 (1 positive, `n_neg` negative)
        - positive: 1 query, ba (target code, db_id)
        - `n_neg` negative: from same db, different code to the target code (if impossible for a certain code, try to replace with other codes)
        - `n_neg` negative: from different db, different code to the target code
        """
        self.task = task
        self.bert_scores: dict[str, float] = bert_scores
        # if threshold < 0 and task == 'retrieval':
        #     self.use_adaptive_threshold = True
        #     self.stats: dict[str, float] = self._check_stats_bert_score()['raw']
        # else:
        #     self.use_adaptive_threshold = False
        self.threshold = threshold
    
        self.n_neg = n_neg
        self._reorganize_samples(data)
        # self.samples = self.create_samples_by_task()

    def save_samples(self, sample_save_path):
        with open(sample_save_path, 'w') as f:
            json.dump(self.samples, f)

    def load_samples(self, sample_save_path):
        with open(sample_save_path, 'r') as f:
            self.samples = json.load(f)

    def _reorganize_samples(self, samples: dict[str, dict[int|str, str]]):
        # sid == sample_id, qid is from 0 to len(samples['question'])
        # [sid1, sid2, ...] -- index of this list is qid
        self.sample_ids = list(samples['question'].keys())
        
        # question: {qid: question}
        # ba: {sid: ba}
        # db_id: {qid: db_id}
        # db_id2qids: {db_id: [qid1, qid2, ...]}
        # codes: {qid: gold_complexity_codes}
        # codes2qids: {db_id: {gold_complexity_codes: [qid1, qid2, ...]}}
        self.questions = {}
        self.bas = {}
        self.db_ids = {}
        self.db_id2qids: dict[str, list[int]] = defaultdict(list)
        self.codes = {}
        self.codes2qids: dict[str, dict[int, list[int]]] = defaultdict(dict)

        for qid, sid in enumerate(self.sample_ids):
            self.questions[qid] = samples['question'][sid]
            self.bas[sid] = samples['ba'][sid]
            
            db_id = samples['db_id'][sid]
            self.db_ids[qid] = db_id
            self.db_id2qids[db_id].append(qid)

            code = samples['gold_complexity_codes'][sid]
            self.codes[qid] = code
            if code not in self.codes2qids[db_id]:
                self.codes2qids[db_id][code] = []
            self.codes2qids[db_id][code].append(qid)

        self.n_unique_codes = len(set(self.codes.values()))

    def _check_stats_bert_score(self):
        dist = defaultdict(list)
        unique_keys = defaultdict(set)
        for key in self.bert_scores:
            k1, k2 = key.split('-')
            unique_keys[k1].add(k2)
            unique_keys[k2].add(k1)

        for key, cand_keys in unique_keys.items():
            scores = [] 
            for cand_key in cand_keys:
                search_key = f'{key}-{cand_key}' if f'{key}-{cand_key}' in self.bert_scores else f'{cand_key}-{key}'
                scores.append(self.bert_scores[search_key])

            dist[key] = np.mean(scores)

        x = list(dist.values())
        
        return {
            'mean': np.mean(x), 'std': np.std(x), 'min': np.min(x), 'max': np.max(x), 'raw': x
        }

    def create_samples_by_task(self):
        """
        question: {qid: question}
        ba: {cid: ba}
        db_id: {qid: db_id}
        db_id2qids: {db_id: [qid1, qid2, ...]}
        codes: {qid: gold_complexity_codes}
        codes2qids: {db_id: {gold_complexity_codes: [qid1, qid2, ...]}}
        """
        if self.task == 'retrieval':
            # create triplets: (anchor, positive, negative)
            # sample number: n_neg (same database) + n_neg (different database)
            self.samples = self._retrieval_sample_generation()
        elif self.task == 'rerank':
            # create pairs: (text=[question, ba], label)
            # sample number: <=11 (3 positive, 8 negative)
            # - 1 query, ba (target code, db_id)
            # - 1 positive: from same db, same code to the target code
            # - 1 positive: from different db, same code to the target code
            # - 4 (at least) negative: from same db, different code to the target code
            # - 4 negative: from different db, different code to the target code
            self.samples = self._rerank_sample_generation()
        else:  # task not found
            raise ValueError(f'Invalid task: {self.task}')

    def _retrieval_sample_generation(self):
        samples = []
        distributions = defaultdict(lambda: {'same_db': 0, 'diff_db': 0})  # check how many hard samples are generated(hard samples = neg from same db)
        iterator = tqdm(enumerate(self.sample_ids), total=len(self.sample_ids))
        for qid, sid in iterator:
            s: dict[str, int|str|list[int]] = {
                'anchor': qid,
                'positive': sid,
                'negative': []  # [neg_sid1, neg_sid2, ...]
            }
            db_id = self.db_ids[qid]

            # negative samples in same database
            iterator.set_postfix_str(f'[{self.task}-{qid}] sampling neg sample from same database')
            i = 0
            sampled = set()  # {qid}
            neg_qids_same_db = []
            n_neg_same_db = self.n_neg
            n_neg_diff_db = self.n_neg
            while i < n_neg_same_db:
                candidates = np.setdiff1d(self.db_id2qids[db_id], [qid]+list(sampled))
                if len(candidates) == 0:
                    # no existing samples that are in the same db, need to sample from different db
                    # warnings.warn(f'[{db_id}-{sid}] No more samples, will sample from different database')
                    n_neg_diff_db += 1
                    i += 1
                    continue
                
                # check sampled ba is too similar to positive ba
                keys: list[tuple[int, str]] = []
                for cqid in candidates:
                    csid = self.sample_ids[cqid]
                    if f'{sid}-{csid}' in self.bert_scores:
                        keys.append((cqid, f'{sid}-{csid}'))
                    elif f'{csid}-{sid}' in self.bert_scores:
                        keys.append((cqid, f'{csid}-{sid}'))
                    else:
                        raise KeyError(f'[{db_id}-{sid}] No bert score found for {sid}-{csid} or {csid}-{sid}')
                # filter too similar samples that over threshold
                candidates = [cqid for (cqid, key) in keys if self.bert_scores[key] < self.threshold]
                if len(candidates) == 0:
                    # warnings.warn(f'[{db_id}-{sid}] No more samples to be sampled that match criteria, it will sample from different database')
                    n_neg_diff_db += 1
                    i += 1
                    continue
                
                sampled_qid = np.random.choice(candidates)

                # check if sampled_qid not in sampled
                if sampled_qid not in sampled:
                    neg_qids_same_db.append(sampled_qid)
                    sampled.add(sampled_qid)
                    i += 1
                    distributions[qid]['same_db'] += 1

            neg_sids_same_db = [self.sample_ids[neg_qid] for neg_qid in neg_qids_same_db]
            s['negative'].extend(neg_sids_same_db)

            # negative samples in different database
            iterator.set_postfix_str(f'[{self.task}-{qid}] sampling neg sample from different database')
            i = 0
            sampled = set()  # {qid}
            neg_qids_diff_db = []
            while i < n_neg_diff_db:
                db_ids_candidates = np.setdiff1d(list(self.db_id2qids.keys()), [db_id])
                sampled_db_id = np.random.choice(db_ids_candidates)
                candidates = np.setdiff1d(self.db_id2qids[sampled_db_id], list(sampled))
                sampled_qid = np.random.choice(candidates)

                # don't need to check sampled ba is too similar to positive ba for different db
                # check if sampled_qid not in sampled
                if sampled_qid not in sampled:
                    neg_qids_diff_db.append(sampled_qid)
                    sampled.add(sampled_qid)
                    i += 1
                    distributions[qid]['diff_db'] += 1

            neg_sids_diff_db = [self.sample_ids[neg_qid] for neg_qid in neg_qids_diff_db]
            s['negative'].extend(neg_sids_diff_db)

            for neg_sid in s['negative']:
                samples.append({
                    'anchor': int(s['anchor']),
                    'positive': s['positive'],
                    'negative': neg_sid
                })

        self._distributions = distributions
        return samples

    def _rerank_sample_generation(self):
        samples = []
        iterator = tqdm(enumerate(self.sample_ids), total=len(self.sample_ids))
        
        # same db, diff code = sd
        # diff db, diff code = dd
        # since it will run for all samples, positive samples are fixed
        # so we need to sample negative samples
        def verifier():
            pass

        def sum_sampled(db_sampled):
            return sum([len(qids) for qids in db_sampled.values()])

        for qid, sid in iterator:
            s: dict[str, list[list[str]]|list[int]] = {
                'text': [],
                'label': []
            }
            # target
            target_code = self.codes[qid]
            db_id = self.db_ids[qid]
            s['text'].append([qid, sid])
            s['label'].append(1)

            # # positive from same db, same code: ss
            # iterator.set_postfix_str(f'[{self.task}-{qid}] sampling pos sample from same database')
            # ss_qid_candidates = np.setdiff1d(self.codes2qids[db_id][target_code], [qid])
            # if len(ss_qid_candidates) == 0:
            #     warnings.warn('unable to find positive sample from same database with the same code, will find from different database')
            #     n_pos_ds = 2
            # else:
            #     n_pos_ds = 1
            #     ss_qid = np.random.choice(ss_qid_candidates)
            #     ss_sid = self.sample_ids[ss_qid]
            #     same_db_sampled[db_id].add(ss_qid)
            #     s['text'].append([ss_qid, ss_sid])
            #     s['label'].append(1)

            # negative from same db, different code: sd
            iterator.set_postfix_str(f'[{self.task}-{qid}] sampling neg sample from same database')
            i = 0
            n_neg_dd = self.n_neg
            n_neg_ds = self.n_neg
            same_db_sampled = set()  # {qid}
            while i < n_neg_ds:
                # check enough qids to sample for other codes
                # code_candidate should have at least 1 sample that is not in sampled before
                sd_code_candidates = []
                for code, qids in self.codes2qids[db_id].items():
                    if (code != target_code) and \
                        (len(np.setdiff1d(qids, list(same_db_sampled))) > 0):
                        sd_code_candidates.append(code)
                if len(sd_code_candidates) == 0:
                    # means that there is no more samples in db_id, keep sampling from other dbs
                    n_neg_dd += 1
                    i += 1
                    continue
                    
                sd_code = np.random.choice(sd_code_candidates)
                sd_qid_candidates = np.setdiff1d(self.codes2qids[db_id][sd_code], 
                                                 [qid]+list(same_db_sampled))
                sd_qid = np.random.choice(sd_qid_candidates)
                sd_sid = self.sample_ids[sd_qid]
                
                # check if sd_qid not in sampled
                if sd_qid not in same_db_sampled:
                    s['text'].append([qid, sd_sid])
                    s['label'].append(0)
                    same_db_sampled.add(sd_qid)
                    i += 1

            # positive from different db, same code: ds
            # iterator.set_postfix_str(f'[{self.task}-{qid}] sampling pos sample from different database')
            # i = 0
            
            # ds_sampled_db_ids = set()  # {db_id} only added if there is no samples that have the same code in different db
            # while i < n_pos_ds:
            #     # if there is no more samples to be sampled, raise error
            #     if len(diff_db_sampled) == len(self.sample_ids):
            #         raise ValueError(f'No more samples to be sampled that match criteria, please decrease `n_neg` or lower the `threshold`')
                
            #     db_ids_candidates = np.setdiff1d(list(self.db_id2qids.keys()), [db_id]+list(ds_sampled_db_ids))
            #     ds_sampled_db_id = np.random.choice(db_ids_candidates)
            #     # check existance of target code in the sampled db
            #     if target_code not in self.codes2qids[ds_sampled_db_id].keys():
            #         ds_sampled_db_ids.add(ds_sampled_db_id)
            #         continue

            #     # check enough qids to sample for target code
            #     ds_qid_candidates = np.setdiff1d(self.codes2qids[ds_sampled_db_id][target_code], list(diff_db_sampled))
            #     if len(ds_qid_candidates) == 0:
            #         # there is no samples that have the same code in different db, check other dbs
            #         ds_sampled_db_ids.add(ds_sampled_db_id)
            #         continue

            #     ds_qid = np.random.choice(ds_qid_candidates)
            #     ds_sid = self.sample_ids[ds_qid]

            #     # check if sampled_qid not in sampled
            #     if ds_qid not in diff_db_sampled:
            #         s['text'].append([ds_qid, ds_sid])
            #         s['label'].append(1)
            #         diff_db_sampled.add(ds_qid)
            #         i += 1
            
            # negative from different db, different code
            iterator.set_postfix_str(f'[{self.task}-{qid}] sampling neg sample from different database')
            i = 0
            diff_db_sampled = defaultdict(set)  # {db_id: {qid}}
            dd_sampled_db_ids = set()  # {db_id} only added if there is no samples that have the same code in different db
            while i < n_neg_dd:
                dd_db_ids_candidates = np.setdiff1d(list(self.db_id2qids.keys()), 
                                                    [db_id]+list(dd_sampled_db_ids))
                dd_sampled_db_id = np.random.choice(dd_db_ids_candidates)

                # check enough qids to sample for other codes
                # code_candidate should have at least 1 sample that is not in sampled before
                dd_code_candidates = []
                for code, qids in self.codes2qids[dd_sampled_db_id].items():
                    if (code != target_code) and \
                        (len(np.setdiff1d(qids, list(diff_db_sampled[dd_sampled_db_id]))) > 0):
                        dd_code_candidates.append(code)
                if len(dd_code_candidates) == 0:
                    # means that there is no more samples in sampled_db_id, keep sampling from other dbs
                    dd_sampled_db_ids.add(dd_sampled_db_id)
                    continue

                dd_code = np.random.choice(dd_code_candidates)
                dd_qid_candidates = np.setdiff1d(self.codes2qids[dd_sampled_db_id][dd_code], 
                                                 list(diff_db_sampled[dd_sampled_db_id]))
                dd_qid = np.random.choice(dd_qid_candidates)
                dd_sid = self.sample_ids[dd_qid]

                # check if sampled_qid not in sampled
                if dd_qid not in diff_db_sampled[dd_sampled_db_id]:
                    s['text'].append([qid, dd_sid])
                    s['label'].append(0)
                    diff_db_sampled[dd_sampled_db_id].add(dd_qid)
                    i += 1

            for i in range(len(s['text'])):
                samples.append({
                    'text': [int(s['text'][i][0]), s['text'][i][1]], # [qid, sid]
                    'label': s['label'][i]
                })

        return samples


    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.task == 'retrieval':
            return self._retrieval_task_itemgetter(idx)
        elif self.task == 'rerank':
            return self._rerank_task_itemgetter(idx)
        else:
            raise ValueError(f'Invalid task: {self.task}')
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def _retrieval_task_itemgetter(self, idx):
        # {anchor: qid, positive: sid, negative: sid}
        sample = self.samples[idx]
        qid = sample['anchor']
        pos_id = sample['positive']
        neg_id = sample['negative']
        
        question = self.questions[qid]
        pos_ba = self.bas[pos_id]
        neg_ba = self.bas[neg_id]

        return {
            'anchor': question,
            'positive': pos_ba,
            'negative': neg_ba
        }
    
    def _rerank_task_itemgetter(self, idx):
        # `sentence_transfomer` will update there code th newer version later. 
        # Here, we use old version for training cross-encoder
        # {text: [question, ba], label: label}
        sample = self.samples[idx]
        qid, sid = sample['text']
        label = sample['label']

        question = self.questions[qid]
        ba = self.bas[sid]
        return {
            'text': [question, ba],
            'label': label
        }
    
    def get_rerank_samples(self, is_train: bool=True):
        samples = []
        if is_train:
            for x in self:
                samples.append(InputExample(texts=x['text'], label=x['label']))
        else:
            # {qid: {'positive': [sid1, sid2, ...], 'negative': [sid1, sid2, ...]}}
            sample_ids = defaultdict(lambda : {'positive': [], 'negative': []})
            for s in self.samples:
                qid, sid = s['text']
                label = s['label']
                if label == 1:
                    sample_ids[qid]['positive'].append(sid)
                else:
                    sample_ids[qid]['negative'].append(sid)

            for qid, v in sample_ids.items():
                samples.append({
                    'query': self.questions[qid],
                    'positive': [self.bas[sid] for sid in v['positive']],
                    'negative': [self.bas[sid] for sid in v['negative']]
                })
        return samples
    

def run_parallel(func, args, num_cpus: int=1):
    pool = mp.Pool(processes=num_cpus)
    results = pool.starmap(func, args)
    pool.close()
    pool.join()
    return results

def extract_first_number_from_index(x: pd.Index|pd.CategoricalIndex):  
    x = x.tolist()
    x = list(map(lambda y: float(y.lstrip('(').split(',')[0]), x))
    return x

# def get_hard_negative_samples(df: pd.DataFrame, n_neg_each_db: int=1):
#     """
#     samples: dict[str, dict[int|str, str]] = {
#         'question': {sample_id: question, ...},
#         'ba': {sample_id: ba, ...},
#         'gold_complexity_codes': {sample_id: gold_complexity_codes, ...}
#     }

#     sample_ids for reranker:

#     sample_ids: dict[db_id] = {
#         'pos': [pos1, pos2, ...],
#         'neg_per_code': {
#             0: [neg1, neg2, ...],
#             1: [neg1, neg2, ...],
#             2: [neg1, neg2, ...],
#             3: [neg1, neg2, ...],
#             4: [neg1, neg2, ...],
#         }
#     }
#     """
#     db_ids = df['db_id'].unique()
#     samples = df.loc[:, ['sample_id', 'question', 'ba', 'gold_complexity_codes']].set_index('sample_id').to_dict('dict')  # question, ba key -- {sample_id: value}
#     sample_ids = {}
#     # positive sample 
#     # for each complexity code, sample 1 negative from each db (then we have 4*len(db) negatives)
#     for db_id in tqdm(db_ids, total=len(db_ids)):
#         positive_samples = df.loc[df['db_id'] == db_id, 'sample_id'].tolist()
#         negative_samples = df.loc[df['db_id'] != db_id, ['sample_id', 'db_id', 'gold_complexity_codes']]\
#             .groupby(['gold_complexity_codes', 'db_id']).sample(n=n_neg_each_db)
#         negative_samples = negative_samples.groupby(['gold_complexity_codes'])['sample_id'].apply(list).to_dict()
#         sample_ids[db_id] = {'pos': positive_samples, 'neg_per_code': negative_samples}

#     return {'samples': samples, 'sample_ids': sample_ids}

def split_train_dev_retrieval_data(
        train_bo_path: Path|str, frac: float=0.9, n_qcut: int =5, 
        n_neg_each_db: int=1, random_state: int=42, num_cpus: int=1):
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
    df['gold_complexity_codes'] = cates.cat.codes.astype(int)
    # get bert score per db
    bert_scores = defaultdict(dict) # {sample_id: {candidate_sample_id: f1 score}}
    combs_per_db = df.groupby('db_id')['sample_id'].apply(lambda x: list(combinations(x.values.tolist(), 2)))
    iterator = tqdm(combs_per_db.items(), total=len(combs_per_db))
    for db_id, pairs in iterator:
        iterator.set_description(f'{db_id} - {len(pairs)} samples')
        cands, refs = list(zip(*[df.loc[df['sample_id'].isin(p), 'ba'].tolist() for p in pairs]))
        with warnings.catch_warnings(action='ignore'):
            *_, f1 = bscore(cands=cands, refs=refs, lang='en', rescale_with_baseline=True, device='cuda', batch_size=256)
        assert len(f1) == len(pairs), f'Length of f1 score is not equal to pairs: {len(f1)} != {len(pairs)}'
        for pair, score in zip(pairs, f1.numpy().tolist()):
            key = '-'.join(map(str, pair))
            bert_scores[key] = score

    distributions = df.loc[:, 'gold_complexity_cates'].value_counts()
    distributions.index = distributions.index.astype(str)
    distributions = distributions.sort_index(key=extract_first_number_from_index).to_dict()

    with open(train_bo_path.parent / f'complexity_distribution.json', 'w') as f:
        json.dump(distributions, f)

    # split train and dev by equal complexity distribution
    df_train = df.groupby('gold_complexity_codes').sample(frac=frac, random_state=random_state)
    df_dev = df.drop(df_train.index)

    # print('Processing train data')
    # train_data = get_hard_negative_samples(df_train, n_neg_each_db)
    # print('Processing dev data')
    # dev_data = get_hard_negative_samples(df_dev, n_neg_each_db)
    # train_data, dev_data = run_parallel(get_hard_negative_samples, [(df_train, n_neg_each_db), (df_dev, n_neg_each_db)], num_cpus=num_cpus)
    train_data = df_train.loc[:, ['sample_id', 'db_id', 'question', 'ba', 'gold_complexity_codes']].set_index('sample_id').to_dict('dict')
    dev_data = df_dev.loc[:, ['sample_id', 'db_id', 'question', 'ba', 'gold_complexity_codes']].set_index('sample_id').to_dict('dict')
    return {
        'train': train_data, 'dev': dev_data, 'bert_scores': bert_scores
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train retrieval model')
    parser.add_argument('--task', type=str, default='retrieval', help='`retrieval`, `data_prep`')
    parser.add_argument('--ds', type=str, default='bird', help='Dataset to use for training. spider or bird') 
    parser.add_argument('--num_cpus', type=int, default=1, help='Number of cpus to use for parallel processing')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--per_device_train_batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--logging_steps', type=int, default=1, help='Logging steps')
    parser.add_argument('--save_steps', type=int, default=10, help='Save steps')
    parser.add_argument('--eval_steps', type=int, default=10, help='Eval steps')
    # parser.add_argument('--base_model_name', type=str, default='msmarco-MiniLM-L6-cos-v5', help='Base model name: msmarco-MiniLM-L6-cos-v5, cross-encoder/ms-marco-MiniLM-L-6-v2')
    parser.add_argument('--n_neg', type=int, default=2, help='Number of negative samples per batch')
    
    parser.add_argument('--create_samples', action='store_true', help='Whether to create samples for training and devset')
    
    # uv run train_retrieval_model.py --task retrieval --ds bird \
    # --per_device_train_batch_size 256 --per_device_eval_batch_size 128 --logging_steps 1 \
    # --save_steps 10 --eval_steps 10 --num_train_epochs 5

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
        data = split_train_dev_retrieval_data(
            experiment_folder / 'predictions' / 'create_bo' / f'final_{args.ds}_train_bo.json', 
            num_cpus=args.num_cpus)
        with open(experiment_folder / 'predictions' / 'create_bo' / f'data_retrieval_rerank.json', 'w') as f:
            json.dump(data, f)

    elif args.task == 'retrieval':
        with open(experiment_folder / 'predictions' / 'create_bo' / f'data_retrieval_rerank.json', 'r') as f:
                data = json.load(f)
        
        train_dataset = BADataset(data=data['train'], bert_scores=data['bert_scores'], 
                                  task=args.task, n_neg=2, threshold=0.3)
        dev_dataset = BADataset(data=data['dev'], bert_scores=data['bert_scores'], 
                                task=args.task, n_neg=1, threshold=0.3)
        train_path = experiment_folder / 'predictions' / 'create_bo' / f'{args.task}_train.json'
        dev_path = experiment_folder / 'predictions' / 'create_bo' / f'{args.task}_dev.json'
        
        if args.create_samples:
            train_dataset.create_samples_by_task()
            train_dataset.save_samples(sample_save_path=train_path)
            dev_dataset.create_samples_by_task()
            dev_dataset.save_samples(sample_save_path=dev_path)
        else:
            train_dataset.load_samples(train_path)
            dev_dataset.load_samples(dev_path)

        base_model_name = 'msmarco-MiniLM-L6-cos-v5'
        model = SentenceTransformer(base_model_name)

        train_ds = Dataset.from_generator(train_dataset.__iter__)
        dev_ds = Dataset.from_generator(dev_dataset.__iter__)
        
        exp_name = f'{base_model_name.split("/")[-1]}-q_ba'

        args = SentenceTransformerTrainingArguments(
            output_dir=f'models/{exp_name}',
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            warmup_ratio=0.1,
            logging_steps=args.logging_steps,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            torch_empty_cache_steps=100,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            logging_dir=f'logs/{exp_name}',
        )
        loss = losses.MultipleNegativesRankingLoss(model)

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

    elif args.task == 'rerank':
        # train rerank model
        # https://github.com/UKPLab/sentence-transformers/tree/master/examples/training/ms_marco
        with open(experiment_folder / 'predictions' / 'create_bo' / f'data_retrieval_rerank.json', 'r') as f:
                data = json.load(f)
        
        train_dataset = BADataset(data=data['train'], bert_scores=data['bert_scores'], 
                                  task=args.task, n_neg=4)
        dev_dataset = BADataset(data=data['dev'], bert_scores=data['bert_scores'], 
                                task=args.task, n_neg=4)
        train_path = experiment_folder / 'predictions' / 'create_bo' / f'{args.task}_train.json'
        dev_path = experiment_folder / 'predictions' / 'create_bo' / f'{args.task}_dev.json'
        
        if args.create_samples:
            train_dataset.create_samples_by_task()
            train_dataset.save_samples(sample_save_path=train_path)
            dev_dataset.create_samples_by_task()
            dev_dataset.save_samples(sample_save_path=dev_path)
        else:
            train_dataset.load_samples(train_path)
            dev_dataset.load_samples(dev_path)

        base_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        model = CrossEncoder(base_model_name, num_labels=1)

        exp_name = f'{base_model_name.split("/")[-1]}-q_ba-rerank'

        train_samples = train_dataset.get_rerank_samples()
        train_loader = DataLoader(
            train_samples, batch_size=args.per_device_train_batch_size, shuffle=True, num_workers=0)
        dev_samples = dev_dataset.get_rerank_samples(is_train=False)
        
        rank_evaluator = CERerankingEvaluator(dev_samples, name="q_ba-eval")
        warmup_steps = 5000

        # Train the model
        model.fit(
            train_dataloader=train_loader,
            evaluator=rank_evaluator,
            epochs=args.num_train_epochs,
            evaluation_steps=args.eval_steps,
            warmup_steps=warmup_steps,
            output_path=f'logs/{exp_name}',
            use_amp=False,
        )

        model.save(f'models/{exp_name}')