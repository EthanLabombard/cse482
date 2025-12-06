# spark_similarity.py
"""
PySpark-based similarity calculator for large-scale data (Hadoop-friendly).
This uses Spark's MLlib approach (IndexedRowMatrix / columnSimilarities) for item-item similarities.
Run this with spark-submit or in a Spark cluster.
"""
from pyspark.sql import SparkSession
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry, IndexedRowMatrix, IndexedRow
from pyspark.mllib.linalg import Vectors
import pyspark.sql.functions as F

def build_indexed_rows_from_ratings(spark, ratings_df, users_list=None, items_list=None, axis='item'):
    """
    ratings_df: spark DataFrame with columns userId, movieId, rating
    axis: 'item' -> we want item vectors across users (columns -> users), 'user' -> user vectors across items
    Returns: IndexedRowMatrix
    """
    # We'll convert to (index, vector) representation. For item-item similarity, we want each item as a dense vector of user ratings.
    if axis == 'item':
        # create mapping from userId to index
        users = users_list if users_list is not None else ratings_df.select("userId").distinct().rdd.map(lambda r: r[0]).collect()
        user_index = {u: i for i, u in enumerate(sorted(users))}
        items = items_list if items_list is not None else ratings_df.select("movieId").distinct().rdd.map(lambda r: r[0]).collect()
        item_index = {m: i for i, m in enumerate(sorted(items))}
        # build per-item sparse vectors
        entries = ratings_df.rdd.map(lambda r: (item_index[r.movieId], (user_index[r.userId], r.rating)))
        grouped = entries.groupByKey().mapValues(list)
        indexed_rows = grouped.map(lambda x: IndexedRow(x[0], Vectors.sparse(len(user_index), [(u_idx, float(r)) for u_idx, r in x[1]])))
        mat = IndexedRowMatrix(indexed_rows)
        return mat
    else:
        # user axis -> analogous
        items = items_list if items_list is not None else ratings_df.select("movieId").distinct().rdd.map(lambda r: r[0]).collect()
        item_index = {m: i for i, m in enumerate(sorted(items))}
        users = users_list if users_list is not None else ratings_df.select("userId").distinct().rdd.map(lambda r: r[0]).collect()
        user_index = {u: i for i, u in enumerate(sorted(users))}
        entries = ratings_df.rdd.map(lambda r: (user_index[r.userId], (item_index[r.movieId], r.rating)))
        grouped = entries.groupByKey().mapValues(list)
        indexed_rows = grouped.map(lambda x: IndexedRow(x[0], Vectors.sparse(len(item_index), [(i_idx, float(r)) for i_idx, r in x[1]])))
        mat = IndexedRowMatrix(indexed_rows)
        return mat

def compute_item_item_similarity(spark, ratings_df, threshold=0.1):
    """
    ratings_df: spark dataframe with columns userId,movieId,rating
    Returns (i, j, sim) RDD of similarities between item pairs above threshold.
    """
    # Build indexed row matrix with items as rows and users as columns (to use columnSimilarities we want row matrix and then transpose; MLlib has columnSimilarities on CoordinateMatrix)
    # Simpler approach: create item vectors as IndexedRowMatrix and convert to CoordinateMatrix then use columnSimilarities after transpose.
    irm = build_indexed_rows_from_ratings(spark, ratings_df, axis='item')
    # Convert to CoordinateMatrix via IndexedRowMatrix.toCoordinateMatrix()
    coord = irm.toCoordinateMatrix()
    # columnSimilarities returns similarities between columns. We want item-item similarities: if our matrix is items x users, its transpose (users x items) has items as columns.
    # So compute columnSimilarities on the coordinate matrix (which treats rows as indices and columns as vector dims).
    # We'll transpose by creating the transposed coordinate matrix (swap i,j)
    entries = coord.entries.map(lambda me: MatrixEntry(me.j, me.i, me.value))
    coord_t = CoordinateMatrix(entries)
    sims = coord_t.toRowMatrix().columnSimilarities(threshold)  # returns CoordinateMatrix of similarities
    # Return as RDD of triples
    return sims.entries.map(lambda me: (int(me.i), int(me.j), float(me.value)))
