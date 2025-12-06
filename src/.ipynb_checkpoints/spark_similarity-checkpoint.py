from pyspark.sql import SparkSession
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry, IndexedRowMatrix, IndexedRow
from pyspark.mllib.linalg import Vectors
import pyspark.sql.functions as F

def build_indexed_rows_from_ratings(spark, ratings_df, users_list=None, items_list=None, axis='item'):
    if axis == 'item':
        users = users_list if users_list is not None else ratings_df.select("userId").distinct().rdd.map(lambda r: r[0]).collect()
        user_index = {u: i for i, u in enumerate(sorted(users))}
        items = items_list if items_list is not None else ratings_df.select("movieId").distinct().rdd.map(lambda r: r[0]).collect()
        item_index = {m: i for i, m in enumerate(sorted(items))}
        entries = ratings_df.rdd.map(lambda r: (item_index[r.movieId], (user_index[r.userId], r.rating)))
        grouped = entries.groupByKey().mapValues(list)
        indexed_rows = grouped.map(lambda x: IndexedRow(x[0], Vectors.sparse(len(user_index), [(u_idx, float(r)) for u_idx, r in x[1]])))
        mat = IndexedRowMatrix(indexed_rows)
        return mat
    else:
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
    irm = build_indexed_rows_from_ratings(spark, ratings_df, axis='item')
    coord = irm.toCoordinateMatrix()
    entries = coord.entries.map(lambda me: MatrixEntry(me.j, me.i, me.value))
    coord_t = CoordinateMatrix(entries)
    sims = coord_t.toRowMatrix().columnSimilarities(threshold)
    return sims.entries.map(lambda me: (int(me.i), int(me.j), float(me.value)))
