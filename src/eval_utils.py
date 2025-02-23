import warnings
import sqlglot
from sqlglot import expressions as exp

import os
import sys
import gc
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from zss import simple_distance, Node
from apted import APTED
from apted.helpers import Tree
from typing import Tuple
from scipy.optimize import linear_sum_assignment 
from transformers import logging as tfloggings
tfloggings.set_verbosity_error()
from itertools import product, pairwise
from func_timeout import func_timeout, FunctionTimedOut
import multiprocessing as mp
from sqlite3 import OperationalError
# import spacy
# try:
#     NLP_SPACY = spacy.load('en_core_web_md')
# except OSError:
#     from spacy.cli import download
#     download('en_core_web_md')
import logging
from bert_score import score as bscore
from src.database import SqliteDatabase

def partial_match(gold_set: set, predict_set: set):
    intersection = gold_set.intersection(predict_set)
    union = gold_set.union(predict_set)
    
    # IoU
    iou = len(intersection) / len(union) if union else 0
    
    # Precision
    precision = len(intersection) / len(predict_set) if predict_set else 0
    
    # Recall
    recall = len(intersection) / len(gold_set) if gold_set else 0
    
    # F1-Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    
    return iou, precision, recall, f1_score

# ------------------------------ Tree Edit Distance and Tree Similarity Edit Distance ------------------------------
def build_tree(ast_node: exp.Expression, build_type: str) -> Tuple[Node|Tree, int]:
    """Build a tree from an AST node.
    
    Args:
        ast_node (exp.Expression): The root AST node.
        build_type (str): The type of tree to build (zss or apted).
    """
    tree_node, node_count = _build_tree(ast_node, build_type)
    if build_type == 'apted':
        tree_node = Tree.from_text(tree_node + '}')
    return tree_node, node_count

def _build_tree(ast_node: exp.Expression, build_type: str) -> Tuple[Node|Tree, int]:
    tree_node = _build_node(ast_node, build_type)
    node_count = 1
    # Recursively add children and count nodes
    for child in ast_node.args.values():
        if isinstance(child, exp.Expression):
            child_node, child_count = _build_tree(child, build_type)
            tree_node = _add_child(child_node, tree_node, build_type)
            node_count += child_count
        elif isinstance(child, list):
            for sub_child in child:
                if isinstance(sub_child, exp.Expression):
                    sub_child_node, sub_child_count = _build_tree(sub_child, build_type)
                    tree_node = _add_child(sub_child_node, tree_node, build_type)
                    node_count += sub_child_count
    return tree_node, node_count

def _build_node(ast_node: exp.Expression, build_type: str) -> Node|str:
    node_name = f'{ast_node.key}({str(ast_node)})'
    if build_type == 'zss':
        return Node(node_name)
    elif build_type == 'apted':
        return '{' + node_name
    else:
        raise ValueError(f"Invalid build type: {build_type} (zss or apted)")
    
def _add_child(child_node: exp.Expression|str, parent_node: Node|str, build_type: str) -> Node|str:
    if build_type == 'zss':
        parent_node.addkid(child_node)
    elif build_type == 'apted':
        parent_node += child_node + '}'
    else:
        raise ValueError(f"Invalid build type: {build_type} (zss or apted)")
    return parent_node

def stringify_zsstree(node: Node, level=0) -> str:
    result = "  " * level + node.label + "\n"
    for child in node.children:
        result += stringify_zsstree(child, level + 1)
    return result

def compute_tsed(sql1: exp.Query, sql2: exp.Query, build_type: str) -> Tuple[float, float]:
    """Compute the Tree Similarity Edit Distance (TSED) between two SQL queries.
    Tree Distance computation: https://aclanthology.org/2024.acl-short.3.pdf

    * zss: 
        * https://github.com/timtadh/zhang-shasha
    * apted: 
        * https://github.com/DatabaseGroup/apted
        * https://github.com/JoaoFelipe/apted

    Args:
        sql1 (exp.Query): The first SQL query.
        sql2 (exp.Query): The second SQL query.
        build_type (str): The type of tree to build (zss or apted). 
    """
    tree1, node_count1 = build_tree(sql1, build_type)
    tree2, node_count2 = build_tree(sql2, build_type)
    if build_type == 'zss':
        distance = simple_distance(tree1, tree2)
    elif build_type == 'apted':
        distance = APTED(tree1, tree2).compute_edit_distance()
    tsed = max(1-distance/max(node_count1,node_count2), 0)
    return tsed, distance

def partial_matching_with_penalty(matrix: np.ndarray, penalty: float=0.0, maximize: bool=True, epsilon: float=1e-9):
    n, m = matrix.shape  # (# of source, # of target)
    size = max(n, m)
    score_matrix = np.full((size, size), -penalty, dtype=np.float32)
    score_matrix[:n, :m] = matrix
    row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=maximize)
    total_score = (score_matrix[row_ind, col_ind] + epsilon).mean()
    return row_ind, col_ind, round(total_score, 6)

def get_structural_score(
        source: list[exp.Expression], 
        target: list[exp.Expression], 
        build_type: str='apted',
        criteria: str='tsed',
        penalty: float=0.01,
    ) -> float:
    """
    n = len(source), m = len(target)
    if n == m: 
        it means that we can match all source to target 
        run partial matching with zero penalty
    if n != m: 
        it means that we can't match all source to target: either we over-guess or under-guess
        run partial matching with np.infty penalty
    criteria: tsed (max) or distance (min)
    """
    n = len(source)
    m = len(target)
    scores = np.zeros((n, m), dtype=np.float32)
    distance = np.zeros((n, m), dtype=np.float32)
    for i, ast1 in enumerate(source):
        for j, ast2 in enumerate(target):
            score, dis = compute_tsed(ast1, ast2, build_type)
            scores[i, j] = score
            distance[i, j] = dis

    maximize = True if criteria == 'tsed' else False
    matrix = scores if criteria == 'tsed' else distance
    *_, final_score = partial_matching_with_penalty(matrix, penalty, maximize)
    return final_score

def stringify_asts(source_asts, target_asts):
    source_str = [str(ast) for ast in source_asts]
    target_str = [str(ast) for ast in target_asts]
    # source_str = [str(ast) if use_bert else NLP_SPACY(str(ast)) for ast in source_asts]
    # target_str = [str(ast) if use_bert else NLP_SPACY(str(ast)) for ast in target_asts]
    return source_str, target_str

def get_semantic_score(
        source: list[exp.Expression],
        target: list[exp.Expression],
        use_bert: bool=True,
        penalty: float=0.01,
        rescale_with_baseline: bool=True,
    ) -> float:
    n = len(source)
    m = len(target)
    source_str, target_str = stringify_asts(source, target)
    # source_str = [str(ast) if use_bert else NLP_SPACY(str(ast)) for ast in source]
    # target_str = [str(ast) if use_bert else NLP_SPACY(str(ast)) for ast in target]
    
    if use_bert:
        source_str_list, target_str_list = list(zip(*product(source_str, target_str)))
        with warnings.catch_warnings(action='ignore'):
            *_, F1 = bscore(source_str_list, target_str_list, lang='en', verbose=False, rescale_with_baseline=rescale_with_baseline)
        matrix = F1.numpy().reshape(n, m)
    else:
        matrix = np.zeros((n, m), dtype=np.float32)
        for i, s in enumerate(source_str):
            for j, t in enumerate(target_str):
                matrix[i, j] = s.similarity(t)

    *_, final_score = partial_matching_with_penalty(matrix, penalty, maximize=True)
    return final_score

def get_final_score(
        source: list[exp.Expression],
        target: list[exp.Expression],
        build_type: str='apted',
        criteria: str='tsed',
        penalty: float=0.01,
        use_bert: bool=True,
        rescale_with_baseline: bool=True,
        epsilon: float=1e-9
    ) -> float:
    structural_score = get_structural_score(source, target, build_type, criteria, penalty)
    semantic_score = get_semantic_score(source, target, use_bert, penalty, rescale_with_baseline)
    f1 = (2 * structural_score * semantic_score + epsilon) / (structural_score + semantic_score + epsilon)
    return structural_score, semantic_score, round(f1, 6)

def get_all_structural_score(
        source_outputs: list[dict], 
        target_outputs: list[dict], 
        build_type: str='apted',
        criteria: str='tsed',
        penalty: float = 0.01,
    ) -> list[float]:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    args = ['table_asts', 'sel_asts', 'cond_asts', 'agg_asts', 'orderby_asts', 'subqueries', 'distinct', 'limit']
    assert build_type in ['apted', 'zss'], f'build_type should be either apted or zss, but got {build_type}'
    assert criteria in ['tsed', 'distance'], f'criteria should be either tsed or distance, but got {criteria}'
    
    structure_scores = []
    assert len(source_outputs) == len(target_outputs), 'source_outputs and target_outputs should have the same length'
    logging.info(f'Computing structural scores for {len(source_outputs)} samples')
    for source_output, target_output in zip(source_outputs, target_outputs):
        if source_output:
            results = {}
            for arg in args:
                source_exists = bool(source_output[arg]) if arg != 'subqueries' else bool(source_output[arg][1:])
                target_exists = bool(target_output[arg]) if arg != 'subqueries' else bool(target_output[arg][1:])
                if target_exists and source_exists:
                    if arg in ['sel_asts', 'cond_asts', 'agg_asts', 'orderby_asts', 'table_asts']:
                        source = [ast for _, ast, _ in source_output[arg]]
                        target = [ast for _, ast, _ in target_output[arg]]
                        structural_score = get_structural_score(source, target, build_type, criteria, penalty)
                    elif arg == 'subqueries':
                        source = source_output[arg][1:]
                        target = target_output[arg][1:]
                        structural_score = get_structural_score(source, target, build_type, criteria, penalty)
                    elif arg in ['distinct', 'limit']:
                        structural_score = 1.0 if criteria == 'tsed' else 0.0
                elif target_exists ^ source_exists:
                    structural_score = 0.0 if criteria == 'tsed' else np.infty
                else:
                    # they don't exist in both so, we can't measure the score
                    structural_score = None
                    # score = 0.0 if criteria == 'tsed' else np.infty
                results[arg] = structural_score
            
            scores = np.array([v for v in results.values() if v is not None])
            epsilon = 1e-9
            overall_score = np.round(np.mean(scores + epsilon), decimals=4)
        else:
            overall_score = 0.0
        structure_scores.append(overall_score)

    # close logging
    logging.shutdown()
    return structure_scores

def get_all_semantic_score(
        source_outputs: list[dict], 
        target_outputs: list[dict], 
        use_bert: bool=True,
        penalty: float=0.01,
        rescale_with_baseline: bool=True,
        criteria: str='tsed',
    ):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    args = ['table_asts', 'sel_asts', 'cond_asts', 'agg_asts', 'orderby_asts', 'subqueries', 'distinct', 'limit']

    all_pairs = []
    all_idxes = defaultdict(dict)
    all_results = defaultdict(dict)
    logging.info(f'Computing semantic scores for {len(source_outputs)} samples')
    for k, (source_output, target_output) in enumerate(zip(source_outputs, target_outputs)):
        for arg in args:
            # if not source_output:
            #     all_results[k][arg] = 0.0
            #     if arg in ['sel_asts', 'cond_asts', 'agg_asts', 'orderby_asts', 'table_asts', 'subqueries']:
            #         all_idxes[k][arg] = [-1, -1]
            #         all_pairs.extend([('None', 'None')])
            #     continue
            source_exists = bool(source_output[arg]) if arg != 'subqueries' else bool(source_output[arg][1:])
            target_exists = bool(target_output[arg]) if arg != 'subqueries' else bool(target_output[arg][1:])
            if target_exists and source_exists:
                if arg in ['sel_asts', 'cond_asts', 'agg_asts', 'orderby_asts', 'table_asts']:
                    source = [ast for _, ast, _ in source_output[arg]]
                    target = [ast for _, ast, _ in target_output[arg]]
                    source_str, target_str = stringify_asts(source, target)
                    pairs = list(zip(source_str, target_str))
                    idxes = list(range(len(all_pairs), len(all_pairs)+len(pairs)))
                    all_pairs.extend(pairs)
                    all_idxes[k][arg] = idxes
                    semantic_score = -1
                elif arg == 'subqueries':
                    source = source_output[arg][1:]
                    target = target_output[arg][1:]
                    source_str, target_str = stringify_asts(source, target)
                    pairs = list(zip(source_str, target_str))
                    idxes = list(range(len(all_pairs), len(all_pairs)+len(pairs)))
                    all_pairs.extend(pairs)
                    all_idxes[k][arg] = idxes
                    semantic_score = -1
                elif arg in ['distinct', 'limit']:
                    semantic_score = 1.0 if criteria == 'tsed' else 0.0
            elif target_exists ^ source_exists:
                semantic_score = 0.0 if criteria == 'tsed' else np.infty
            else:
                # they don't exist in both so, we can't measure the score
                semantic_score = None
            all_results[k][arg] = semantic_score

    source_str, target_str = zip(*all_pairs)
    if use_bert:
        source_str_list = []
        target_str_list = []
        sparse_idxes = []
        idx2arg = defaultdict(dict)
        for k, key_idxes in all_idxes.items():
            for arg, idxes in key_idxes.items():
                s, e = idxes[0], idxes[-1]+1
                xs = list(product(source_str[s:e], target_str[s:e]))
                s_str, t_str = list(zip(*xs))
                idx = len(xs) + (sparse_idxes[-1] if sparse_idxes else 0)
                sparse_idxes.append(idx)
                idx2arg[idx] = (k, arg)
                source_str_list.extend(s_str)
                target_str_list.extend(t_str)
        with warnings.catch_warnings(action='ignore'):
            logging.info('Computing BERTScore')
            *_, F1 = bscore(source_str_list, target_str_list, lang='en', verbose=False, rescale_with_baseline=rescale_with_baseline, device='cuda')
        scores = F1.numpy()
    else:
        scores = []
        # for k, key_idxes in all_idxes.items():
        #     for key, idxes in key_idxes.items():
        #         s, e = idxes[0], idxes[-1]+1
        #         s_str, t_str = list(zip(*product(source_str[s:e], target_str[s:e])))
        #         for i, s in enumerate(s_str):
        #             for j, t in enumerate(t_str):
        #                 scores.append(s.similarity(t))
        scores = np.array(scores)

    for i, j in pairwise([0] + sparse_idxes):
        n = int(np.sqrt(j - i))
        matrix = scores[i:j].reshape(n, n)
        k, arg = idx2arg[j]
        *_, final_score = partial_matching_with_penalty(matrix, penalty, maximize=True)
        all_results[k][arg] = final_score

    semantic_scores = []
    for k, results in all_results.items():
        scores = np.array([v for v in results.values() if v is not None])
        epsilon = 1e-9
        overall_score = np.round(np.mean(scores + epsilon), decimals=4)
        semantic_scores.append(overall_score)
    
    logging.shutdown()
    return semantic_scores

def get_partial_score(
        source_output, 
        target_output, 
        arg,
        build_type: str='apted',
        criteria: str='tsed',
        penalty: float = 0.01,
        use_bert: bool = True,
        rescale_with_baseline: bool = True
    ):
    """
    table:

    target |  prediction  |  score
    True   |  True        |  depends on arg
    True   |  False       |  tsed=0.0 or distance=np.infty
    False  |  True        |  tsed=0.0 or distance=np.infty
    False  |  False       |  None
    
    arg: 
     - use all: 'sel_asts', 'cond_asts', 'agg_asts', 'orderby_asts'
     - only use items from 2nd item in the list: 'subqueries'
     - boolean: 'distinct', 'limit'
    """
    assert build_type in ['apted', 'zss'], f'build_type should be either apted or zss, but got {build_type}'
    assert criteria in ['tsed', 'distance'], f'criteria should be either tsed or distance, but got {criteria}'
    assert arg in ['table_asts', 'sel_asts', 'cond_asts', 'agg_asts', 'orderby_asts', 'subqueries', 'distinct', 'limit'], f'arg should be either sel_asts, cond_asts, agg_asts, orderby_asts, subqueries, distinct, limit, but got {arg}'
    
    source_exists = bool(source_output[arg]) if arg != 'subqueries' else bool(source_output[arg][1:])
    target_exists = bool(target_output[arg]) if arg != 'subqueries' else bool(target_output[arg][1:])

    if target_exists and source_exists:
        if arg in ['sel_asts', 'cond_asts', 'agg_asts', 'orderby_asts', 'table_asts']:
            source = [ast for _, ast, _ in source_output[arg]]
            target = [ast for _, ast, _ in target_output[arg]]
            structural_score, semantic_score, score = get_final_score(source, target, build_type, criteria, penalty, use_bert, rescale_with_baseline)
        elif arg == 'subqueries':
            source = source_output[arg][1:]
            target = target_output[arg][1:]
            structural_score, semantic_score, score = get_final_score(source, target, build_type, criteria, penalty, use_bert, rescale_with_baseline)
        elif arg in ['distinct', 'limit']:
            score = 1.0 if criteria == 'tsed' else 0.0
            structural_score, semantic_score = score, score
    elif target_exists ^ source_exists:
        score = 0.0 if criteria == 'tsed' else np.infty
        structural_score, semantic_score = score, score 
    else:
        # they don't exist in both so, we can't measure the score
        score = None
        structural_score, semantic_score = score, score
        # score = 0.0 if criteria == 'tsed' else np.infty
    return structural_score, semantic_score, score

def get_all_partial_score(
        source_output,
        target_output,
        build_type: str='apted',
        criteria: str='tsed',
        penalty: float = 0.01,
        use_bert: bool = True,
        rescale_with_baseline: bool = True
    ) -> Tuple[dict, dict]:
    args = ['table_asts', 'sel_asts', 'cond_asts', 'agg_asts', 'orderby_asts', 'subqueries', 'distinct', 'limit']
    results = {}
    for arg in args:
        structural_score, semantic_score, score = get_partial_score(source_output, target_output, arg, build_type, criteria, penalty, use_bert, rescale_with_baseline)
        results[arg] = structural_score, semantic_score, score
    
    scores = np.array([v for v in results.values() if v[-1] is not None])
    epsilon = 1e-9
    overall_score = np.round(np.mean(scores + epsilon, axis=0), decimals=4)
    final_score = {
        'structural': overall_score[0],
        'semantic': overall_score[1],
        'overall': overall_score[2]
    }
    return results, final_score  # structural, semantic, overall

# ---------------- Complexity ---------------- 

def get_complexity(output):
    args1 = [('sel', 'sel_asts'), ('cond_asts', 'op_types'), ('agg', 'agg_asts'), ('orderby', 'orderby_asts')]
    args2 = ['distinct', 'limit', 'nested', 'table_asts']
    total_complexity = []
    for arg in args1:
        exists = all([output[arg[0]], output[arg[1]]])
        if exists:
            x = [len(output[arg[0]]), len(output[arg[1]])]
            complexity = np.sum(x)
            total_complexity.append(complexity)
    
    for arg in args2:
        if output[arg]:
            if arg == 'nested':
                complexity = output[arg]
            if arg == 'table_asts':
                complexity = len(output[arg])
            if arg in ('distinct', 'limit') and output[arg]:
                complexity = int(output[arg])
            total_complexity.append(complexity)
    return int(sum(total_complexity))

# ------------------------------ Eval execution results ------------------------------
# Source: https://github.com/taoyds/spider/blob/master/evaluation.py

def permute_tuple(element: tuple, perm: tuple) -> tuple:
    assert len(element) == len(perm)
    return tuple([element[i] for i in perm])

def unorder_row(row: tuple) -> tuple:
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))

# unorder each row in the table
# [result_1 and result_2 has the same bag of unordered row]
# is a necessary condition of
# [result_1 and result_2 are equivalent in denotation]
def quick_rej(result1: list[tuple], result2: list[tuple], order_matters: bool) -> bool:
    s1 = [unorder_row(row) for row in result1]
    s2 = [unorder_row(row) for row in result2]
    if order_matters:
        return s1 == s2
    else:
        return set(s1) == set(s2)

# return whether two bag of relations are equivalent
def multiset_eq(l1: list, l2: list) -> bool:
    if len(l1) != len(l2):
        return False
    d = defaultdict(int)
    for e in l1:
        d[e] = d[e] + 1
    for e in l2:
        d[e] = d[e] - 1
        if d[e] < 0:
            return False
    return True

def get_constraint_permutation(tab1_sets_by_columns: list[set], result2: list[tuple]):
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints)

    # we sample 20 rows and constrain the space of permutations
    for _ in range(20):
        random_tab2_row = random.choice(result2)

        for tab1_col in range(num_cols):
            for tab2_col in set(perm_constraints[tab1_col]):
                if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                    perm_constraints[tab1_col].remove(tab2_col)
    return product(*perm_constraints)

def result_eq(result1: list[tuple], result2: list[tuple], order_matters: bool) -> bool:
    # FROM: test-suqte-sql-eval/eval.py
    if len(result1) == 0 and len(result2) == 0:
        return True

    # if length is not the same, then they are definitely different bag of rows
    if len(result1) != len(result2):
        return False

    num_cols = len(result1[0])

    # if the results do not have the same number of columns, they are different
    if len(result2[0]) != num_cols:
        return False

    # unorder each row and compare whether the denotation is the same
    # this can already find most pair of denotations that are different
    if not quick_rej(result1, result2, order_matters):
        return False

    # the rest of the problem is in fact more complicated than one might think
    # we want to find a permutation of column order and a permutation of row order,
    # s.t. result_1 is the same as result_2
    # we return true if we can find such column & row permutations
    # and false if we cannot
    tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]

    # on a high level, we enumerate all possible column permutations that might make result_1 == result_2
    # we decrease the size of the column permutation space by the function get_constraint_permutation
    # if one of the permutation make result_1, result_2 equivalent, then they are equivalent
    for perm in get_constraint_permutation(tab1_sets_by_columns, result2):
        if len(perm) != len(set(perm)):
            continue
        if num_cols == 1:
            result2_perm = result2
        else:
            result2_perm = [permute_tuple(element, perm) for element in result2]
        if order_matters:
            if result1 == result2_perm:
                return True
        else:
            # in fact the first condition must hold if the second condition holds
            # but the first is way more efficient implementation-wise
            # and we use it to quickly reject impossible candidates
            if set(result1) == set(result2_perm) and multiset_eq(result1, result2_perm):
                return True
    return False

def check_if_exists_orderby(sql):
    if sqlglot.parse_one(sql).find(exp.Order) is not None:
        return True
    return False


def execute_sql(
    pred: str, 
    target: str, 
    db_file: str,
    max_rows: int = 10000 
):
    skip_db_ids = ['movie_platform']
    db_id = Path(db_file).stem
    if db_id in skip_db_ids:
        return 0, False
    db = SqliteDatabase(db_file=db_file)
    try:
        target_sql = f'SELECT * FROM ({target}) LIMIT {max_rows};'  # avoid to load too many rows
        target_res = db.execute(target_sql, rt_pandas=False)
        target_error = False
    except OperationalError as e:
        target_error = True
    try:
        pred_sql = f'SELECT * FROM ({pred}) LIMIT {max_rows};'  # avoid to load too many rows
        pred_res = db.execute(pred_sql, rt_pandas=False)
        exists_orderby = check_if_exists_orderby(target)
        res = int(result_eq(pred_res, target_res, order_matters=exists_orderby))
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        res = 0
        target_error = False

    return res, target_error

def execute_model(
    pred: str, 
    target: str, 
    db_file: str, 
    sample_id: int, 
    meta_time_out: float
):
    try:
        # with contextlib.redirect_stderr(io.StringIO()):
        res, target_error = func_timeout(
            meta_time_out,
            execute_sql,
            args=(pred, target, db_file),
        )
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f"timeout",)]
        res = 0
        target_error = False
        gc.collect()
    except Exception as e:
        result = [(f"error",)]  # possibly len(query) > 512 or not executable
        res = 0
        target_error = False
    
    result = {"sample_id": sample_id, "res": res, "target_error": target_error}
    return result

def run_sqls(eval_data, meta_time_out=30.0):
    sample_ids = eval_data['sample_ids']
    pred_queries = eval_data['pred_queries']
    target_queries = eval_data['target_queries']
    db_paths = eval_data['db_paths']
    exec_result = [None] * len(sample_ids)

    samples_by_db = defaultdict(list)
    for i, (sample_id, pred, target, db_file) in enumerate(zip(sample_ids, pred_queries, target_queries, db_paths)):
        samples_by_db[db_file].append((i, sample_id, pred, target))

    for db_file, samples in samples_by_db.items():
        
        for i, sample_id, pred, target in tqdm(
            samples, total=len(samples), desc=f"Processing {Path(db_file).stem}"
        ):
            result = execute_model(pred, target, db_file, sample_id, meta_time_out)
            exec_result[i] = result
        
    return exec_result


def worker_execute_sql(q: mp.Queue, pred: str, target: str, db_file: str):
    """
    Runs the execute_sql function and puts the (result, target_error) tuple into the queue.
    """
    try:
        res, target_error = execute_sql(pred, target, db_file)
        q.put((res, target_error))
    except Exception as e:
        logging.error(f"Worker {os.getpid()}: Exception in worker_execute_sql: {e}")
        q.put((0, False))

def aexecute_model(
        pred: str, target: str, db_file: str, sample_id: int, meta_time_out: float, order: int):
    pid = os.getpid()
    # logging.info(f"Worker {pid}: Starting execute_model for sample_id {sample_id})")
    try:
        res, target_error = func_timeout(
            meta_time_out,
            execute_sql,
            args=(pred, target, db_file)
        )
    except KeyboardInterrupt:
        logging.info(f"[{order}] Worker {pid}: Received KeyboardInterrupt. Exiting.")
        sys.exit(0)
    except FunctionTimedOut:
        logging.warning(f"[{order}] Worker {pid}: FunctionTimedOut for sample_id {sample_id} after {meta_time_out} seconds.")
        res, target_error = 0, False
        gc.collect()
    except Exception as e:
        logging.error(f"[{order}] Worker {pid}: Exception in execute_model for sample_id {sample_id}: {e}")
        res, target_error = 0, False
    finally:
        logging.info(f"[{order}] Worker {pid}: Finished execute_model for sample_id {sample_id} with res={res}, target_error={target_error}")
    
    # Include the order in the returned result
    return {"order": order, "sample_id": sample_id, "res": res, "target_error": target_error}


def run_sqls_parallel(eval_data, num_cpus=1, meta_time_out=30.0):
    """
    Runs execute_model in parallel using a multiprocessing Pool.
    Uses a tqdm progress bar to track overall progress.
    """
    sample_ids = eval_data['sample_ids']
    pred_queries = eval_data['pred_queries']
    target_queries = eval_data['target_queries']
    db_paths = eval_data['db_paths']
    exec_results = [None] * len(sample_ids)

    # pbar = tqdm(total=len(sample_ids), desc="Processing execution", position=0)
    # Create a pool that recycles workers after each task to help release memory.
    pool = mp.Pool(processes=num_cpus, maxtasksperchild=2)

    def update(result):
        order = result['order']
        exec_results[order] = {
            'sample_id': result['sample_id'],
            'res': result['res'],
            'target_error': result['target_error'],
        }
        # pbar.update(1)

    # Enumerate tasks to assign a unique order to each one.
    for order, (sample_id, pred, target, db_file) in enumerate(zip(sample_ids, pred_queries, target_queries, db_paths)):
        pool.apply_async(
            aexecute_model,
            args=(pred, target, db_file, sample_id, meta_time_out, order),
            callback=update,
        )

    pool.close()
    pool.join()
    # pbar.close()

    return exec_results


if __name__ == '__main__':
    # example
    from .parsing_sql import Schema, extract_all

    sql1 = """SELECT
    COUNT(*) AS count, lineitem.l_returnflag
    FROM lineitem
    WHERE
    lineitem.l_commitdate < lineitem.l_receiptdate
    AND lineitem.l_receiptdate >= '1993-01-01'
    AND lineitem.l_receiptdate < '1994-01-01'
    """

    sql2 = """SELECT
    COUNT(lineitem.L_RECEIPTDATE), lineitem.l_returnflag
    FROM LINEITEM L
    WHERE
    lineitem.L_RECEIPTDATE > lineitem.L_COMMITDATE
    AND STRFTIME('%Y', lineitem.L_RECEIPTDATE) = 'abcd'
    """

    schema = Schema({
        'lineitem': {'l_receiptdate': 'date', 'l_commitdate': 'date'}
    })

    source_output = extract_all(sql1, schema)
    target_output = extract_all(sql2, schema)
    
    formatted_sql1 = source_output['subqueries'][0]
    formatted_sql2 = target_output['subqueries'][0]
    tsed, distance = compute_tsed(formatted_sql1, formatted_sql2, build_type='apted')  # apted or zss
    print('[SQL1]\n', formatted_sql1.sql(pretty=True))
    print()
    print('[SQL2]\n', formatted_sql2.sql(pretty=True))
    print()
    print(f'TSED: {tsed:.4f}')
    print(f'Tree Edit Distance: {distance}')

    # partial match
    print('Partial Match Score')
    
    results, overall_score = get_all_partial_score(source_output, target_output, build_type='apted', criteria='tsed', penalty=0.01, use_bert=True, rescale_with_baseline=True)
    for k, v in results.items():
        if v:
            print(f'- {k}: {v:.4f}')
        else:
            print(f'- {k}: {v}')
    print(f'Overall Score: {overall_score:.4f}')