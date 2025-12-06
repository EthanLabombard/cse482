import numpy as np
from sklearn.metrics import mean_squared_error
import math

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def precision_at_k(recommended_list, ground_truth_list, k):
    rec_k = recommended_list[:k]
    hits = len(set(rec_k).intersection(set(ground_truth_list)))
    return hits / k

def recall_at_k(recommended_list, ground_truth_list, k):
    rec_k = recommended_list[:k]
    hits = len(set(rec_k).intersection(set(ground_truth_list)))
    if len(ground_truth_list) == 0:
        return 0.0
    return hits / len(ground_truth_list)

def ndcg_at_k(recommended_list, ground_truth_set, k):
    dcg = 0.0
    for i, item in enumerate(recommended_list[:k]):
        if item in ground_truth_set:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(ground_truth_set), k))])
    return dcg / idcg if idcg > 0 else 0.0
