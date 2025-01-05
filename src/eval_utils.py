import warnings
import sqlglot
import numpy as np

from zss import simple_distance, Node
from sqlglot import expressions as exp
from apted import APTED
from apted.helpers import Tree
from typing import Tuple
from itertools import product
from scipy.optimize import linear_sum_assignment 
from transformers import logging as tfloggings
tfloggings.set_verbosity_error()

import spacy
try:
    NLP_SPACY = spacy.load('en_core_web_md')
except OSError:
    from spacy.cli import download
    download('en_core_web_md')

from bert_score import score as bscore
# from .parsing_sql import _format_select


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

def get_semantic_score(
        source: list[exp.Expression],
        target: list[exp.Expression],
        use_bert: bool=True,
        penalty: float=0.01,
        rescale_with_baseline: bool=True,
    ) -> float:
    n = len(source)
    m = len(target)
    source_str = [str(ast) if use_bert else NLP_SPACY(str(ast)) for ast in source]
    target_str = [str(ast) if use_bert else NLP_SPACY(str(ast)) for ast in target]
    
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
        source_output, 
        target_output, 
        build_type: str='apted',
        criteria: str='tsed',
        penalty: float = 0.01,
    ):
    args = ['table_asts', 'sel_asts', 'cond_asts', 'agg_asts', 'orderby_asts', 'subqueries', 'distinct', 'limit']
    assert build_type in ['apted', 'zss'], f'build_type should be either apted or zss, but got {build_type}'
    assert criteria in ['tsed', 'distance'], f'criteria should be either tsed or distance, but got {criteria}'
        
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
    return results, overall_score

def get_all_semantic_score(
        source_output, 
        target_output, 
        use_bert: bool=True,
        penalty: float=0.01,
        rescale_with_baseline: bool=True,
        criteria: str='tsed',
    ):
    args = ['table_asts', 'sel_asts', 'cond_asts', 'agg_asts', 'orderby_asts', 'subqueries', 'distinct', 'limit']

    results = {}
    for arg in args:
        source_exists = bool(source_output[arg]) if arg != 'subqueries' else bool(source_output[arg][1:])
        target_exists = bool(target_output[arg]) if arg != 'subqueries' else bool(target_output[arg][1:])
        if target_exists and source_exists:
            if arg in ['sel_asts', 'cond_asts', 'agg_asts', 'orderby_asts', 'table_asts']:
                source = [ast for _, ast, _ in source_output[arg]]
                target = [ast for _, ast, _ in target_output[arg]]
                semantic_score = get_semantic_score(source, target, use_bert, penalty, rescale_with_baseline)
            elif arg == 'subqueries':
                source = source_output[arg][1:]
                target = target_output[arg][1:]
                semantic_score = get_semantic_score(source, target, use_bert, penalty, rescale_with_baseline)
            elif arg in ['distinct', 'limit']:
                semantic_score = 1.0 if criteria == 'tsed' else 0.0
        elif target_exists ^ source_exists:
            semantic_score = 0.0 if criteria == 'tsed' else np.infty
        else:
            # they don't exist in both so, we can't measure the score
            semantic_score = None
            # score = 0.0 if criteria == 'tsed' else np.infty
        results[arg] = semantic_score
    
    scores = np.array([v for v in results.values() if v is not None])
    epsilon = 1e-9
    overall_score = np.round(np.mean(scores + epsilon), decimals=4)
    return results, overall_score

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
def normalize_values(x, min_value=0, max_value=6):
    normalized = (x - min_value) / (max_value - min_value)
    return normalized

def tanh(x: np.ndarray, k: float):
    normalized = normalize_values(x, max_value=k)
    return np.tanh(np.log(1+normalized.sum()))

def derive_complexity(x: list[int], k=6):
    complexity = tanh(np.array(x), k)
    return complexity

def get_complexity(output, k=6):
    args1 = [('sel', 'sel_asts'), ('cond_asts', 'op_types'), ('agg', 'agg_asts'), ('orderby', 'orderby_asts')]
    args2 = ['distinct', 'limit', 'nested', 'table_asts']
    total_complexity = []
    for arg in args1:
        exists = all([output[arg[0]], output[arg[1]]])
        if exists:
            x = [len(output[arg[0]]), len(output[arg[1]])]
            complexity = derive_complexity(x, k=k)
            total_complexity.append(complexity)
    
    for arg in args2:
        if output[arg]:
            if arg == 'nested':
                complexity = derive_complexity([output[arg]], k=k)
            elif arg == 'table_asts':
                complexity = derive_complexity([len(output[arg])], k=k)
            else:
                complexity = derive_complexity([int(output[arg])], k=k)
            total_complexity.append(complexity)
    return np.mean(total_complexity)

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