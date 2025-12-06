# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from local_similarity import build_user_item_matrix, compute_cosine_similarity_sparse
from recommender import recommend_for_user_userbased, recommend_for_user_itembased
from data_prep import load_local, train_test_split_by_user

st.set_page_config(page_title="Recommender Demo", layout="wide")
st.title("Movie Recommender — User-based vs Item-based")

@st.cache_data
def load_data():
    ratings, movies = load_local()
    return ratings, movies

ratings, movies = load_data()
train, test = train_test_split_by_user(ratings, test_size=0.2)
st.sidebar.write(f"Ratings: {len(ratings)}, Users: {ratings.userId.nunique()}, Movies: {ratings.movieId.nunique()}")
users = sorted(ratings.userId.unique())
movies_map = movies.set_index('movieId')['title'].to_dict()

st.sidebar.header("Pick a user")
user_id = st.sidebar.selectbox("User ID", users)

if st.button("Compute similarities (local)"):
    M, user_index, item_index = build_user_item_matrix(train)
    st.session_state['M'] = M
    st.session_state['user_index'] = user_index
    st.session_state['item_index'] = item_index
    st.session_state['user_sim'] = compute_cosine_similarity_sparse(M, axis=1)
    st.session_state['item_sim'] = compute_cosine_similarity_sparse(M, axis=0)
    st.success("Similarity matrices computed (in-memory).")

if 'M' not in st.session_state:
    st.info("Click 'Compute similarities (local)' to build recommendations.")
else:
    st.header(f"Recommendations for user {user_id}")
    user_index = st.session_state['user_index']
    item_index = st.session_state['item_index']
    rating_matrix = st.session_state['M']
    user_sim = st.session_state['user_sim']
    item_sim = st.session_state['item_sim']

    top_n = st.sidebar.slider("Top N", 5, 50, 10)
    ub = recommend_for_user_userbased(user_id, user_index, item_index, rating_matrix, user_sim, top_n)
    ib = recommend_for_user_itembased(user_id, user_index, item_index, rating_matrix, item_sim, top_n)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("User-based CF")
        for mid, score in ub:
            st.write(f"{movies_map.get(mid,'?')} — predicted {score:.2f}")
    with col2:
        st.subheader("Item-based CF")
        for mid, score in ib:
            st.write(f"{movies_map.get(mid,'?')} — predicted {score:.2f}")

    st.markdown("---")
    st.subheader("User's actual recent ratings (sample)")
    user_r = ratings[ratings.userId==user_id].sort_values('timestamp', ascending=False).head(20)
    st.dataframe(user_r.merge(movies, on='movieId')[['title','rating','timestamp']])
