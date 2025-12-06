import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from local_similarity import build_user_item_matrix, compute_cosine_similarity_sparse

def predict_user_based(user_id, item_id, user_index, item_index, rating_matrix, user_sim, k=20, global_mean=None):
    if user_id not in user_index or item_id not in item_index:
        return global_mean if global_mean is not None else 3.0
    u_idx = user_index[user_id]
    i_idx = item_index[item_id]
    sim_vec = user_sim[u_idx]
    col = rating_matrix[:, i_idx].toarray().ravel()
    rated_by = np.where(col > 0)[0]
    if len(rated_by) == 0:
        return global_mean if global_mean is not None else 3.0
    sims = sim_vec[rated_by]
    top_k_idx = rated_by[np.argsort(-sims)[:k]]
    top_sims = sim_vec[top_k_idx]
    ratings = col[top_k_idx]
    if top_sims.sum() <= 0:
        return np.mean(ratings)
    return (top_sims @ ratings) / (np.abs(top_sims).sum())

def predict_item_based(user_id, item_id, user_index, item_index, rating_matrix, item_sim, k=20, global_mean=None):
    if user_id not in user_index or item_id not in item_index:
        return global_mean if global_mean is not None else 3.0
    u_idx = user_index[user_id]
    i_idx = item_index[item_id]
    user_ratings = rating_matrix[u_idx].toarray().ravel()
    rated_items = np.where(user_ratings > 0)[0]
    if len(rated_items) == 0:
        return global_mean if global_mean is not None else 3.0
    sim_vec = item_sim[i_idx, rated_items]
    top_k_idx = rated_items[np.argsort(-sim_vec)[:k]]
    top_sims = item_sim[i_idx, top_k_idx]
    top_ratings = user_ratings[top_k_idx]
    if np.abs(top_sims).sum() == 0:
        return np.mean(top_ratings)
    return (top_sims @ top_ratings) / np.abs(top_sims).sum()

def recommend_for_user_userbased(user_id, user_index, item_index, rating_matrix, user_sim, top_n=10):
    inv_item_index = {v:k for k,v in item_index.items()}
    u_idx = user_index[user_id]
    user_row = rating_matrix[u_idx].toarray().ravel()
    unrated = np.where(user_row == 0)[0]
    preds = []
    for i_idx in unrated:
        pred = predict_user_based(user_id, inv_item_index[i_idx], user_index, item_index, rating_matrix, user_sim)
        preds.append((inv_item_index[i_idx], pred))
    preds.sort(key=lambda x: -x[1])
    return preds[:top_n]

def recommend_for_user_itembased(user_id, user_index, item_index, rating_matrix, item_sim, top_n=10):
    inv_item_index = {v:k for k,v in item_index.items()}
    u_idx = user_index[user_id]
    user_row = rating_matrix[u_idx].toarray().ravel()
    unrated = np.where(user_row == 0)[0]
    preds = []
    for i_idx in unrated:
        pred = predict_item_based(user_id, inv_item_index[i_idx], user_index, item_index, rating_matrix, item_sim)
        preds.append((inv_item_index[i_idx], pred))
    preds.sort(key=lambda x: -x[1])
    return preds[:top_n]

