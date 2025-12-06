# local_similarity.py
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

def build_user_item_matrix(ratings, users=None, items=None):
    """
    ratings: DataFrame with columns [userId, movieId, rating]
    Returns: csr_matrix (n_users x n_items), user_index, item_index
    """
    unique_users = users if users is not None else sorted(ratings['userId'].unique())
    unique_items = items if items is not None else sorted(ratings['movieId'].unique())
    user_index = {u: i for i, u in enumerate(unique_users)}
    item_index = {i_: j for j, i_ in enumerate(unique_items)}
    rows, cols, data = [], [], []
    for r in ratings.itertuples():
        rows.append(user_index[r.userId])
        cols.append(item_index[r.movieId])
        data.append(r.rating)
    M = csr_matrix((data, (rows, cols)), shape=(len(unique_users), len(unique_items)))
    return M, user_index, item_index

def compute_cosine_similarity_sparse(matrix, axis=0):
    """
    axis=0: compute similarity between columns (items)
    axis=1: compute similarity between rows (users)
    Uses sklearn cosine_similarity on dense blocks for small datasets.
    Returns a dense numpy array of similarities.
    """
    if axis == 0:
        mat = matrix.toarray() if hasattr(matrix, "toarray") else matrix
        vecs = mat
        # items are columns -> transpose
        vecs = vecs.T
    else:
        vecs = matrix.toarray()
    sim = cosine_similarity(vecs)
    return sim
