import sqlglot
from zss import simple_distance, Node
from sqlglot import expressions as exp
from apted import APTED
from apted.helpers import Tree
from typing import Tuple

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
def build_tree(ast_node: exp.Query, build_type: str) -> Tuple[Node|Tree, int]:
    """Build a tree from an AST node.
    
    Args:
        ast_node (exp.Expression): The root AST node.
        build_type (str): The type of tree to build (zss or apted).
    """
    tree_node, node_count = _build_tree(ast_node, build_type)
    if build_type == 'apted':
        tree_node = Tree.from_text(tree_node + '}')
    return tree_node, node_count

def _build_tree(ast_node: exp.Query, build_type: str) -> Tuple[Node|Tree, int]:
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


if __name__ == '__main__':
    # example
    build_type = 'apted' # zss or apted
    sql1 = sqlglot.parse_one("SELECT c1 FROM t1 WHERE a = 1 AND b = 2")
    sql2 = sqlglot.parse_one("SELECT c1,c2 FROM t1 WHERE a = 2 AND b = 2")
    tsed, distance = compute_tsed(sql1, sql2, build_type)
    print(f'TSED: {tsed:.4f}, Distance: {distance}')