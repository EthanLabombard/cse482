# data_prep.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

MOVIELENS_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def load_local(data_dir="data"):
    ratings_path = os.path.join(data_dir, "ratings.csv")
    movies_path = os.path.join(data_dir, "movies.csv")
    if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
        raise FileNotFoundError("Please download MovieLens ml-latest-small and place ratings.csv and movies.csv in ./data/")
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    return ratings, movies

def train_test_split_by_user(ratings, test_size=0.2, seed=42):
    # For each user, hold out a fraction of their ratings for test (scikit-learn stratified by user)
    train_list = []
    test_list = []
    grouped = ratings.groupby('userId')
    for user, grp in grouped:
        if len(grp) < 2:
            train_list.append(grp)
            continue
        tr, te = train_test_split(grp, test_size=test_size, random_state=seed)
        train_list.append(tr)
        test_list.append(te)
    train = pd.concat(train_list).reset_index(drop=True)
    test = pd.concat(test_list).reset_index(drop=True) if test_list else pd.DataFrame(columns=ratings.columns)
    return train, test

if __name__ == "__main__":
    ratings, movies = load_local()
    train, test = train_test_split_by_user(ratings, test_size=0.2)
    print("Train size:", len(train), "Test size:", len(test))


