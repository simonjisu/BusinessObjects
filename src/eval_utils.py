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