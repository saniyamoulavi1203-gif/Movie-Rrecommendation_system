"""Streamlit interface for the Movie Recommendation System."""
from pathlib import Path
import pickle
import streamlit as st
from predict_movies import recommend_for_user, recommend_similar

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "movie_recommender.pkl"

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")
st.title("🎬 Movie Recommendation System")
st.write("Discover movies using content similarity and rating-based personalization.")

if not MODEL_PATH.exists():
    st.warning("Model files are missing. Run `python train_model.py` in the project folder first.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    artifacts = pickle.load(f)

mode = st.radio("Recommendation type", ["For a user", "Similar to a movie"], horizontal=True)
top_n = st.slider("Number of recommendations", 3, 10, 5)

if mode == "For a user":
    user_ids = sorted(artifacts["user_item"].index.astype(int).tolist())
    user_id = st.selectbox("Choose a sample user ID", user_ids)
    if st.button("Recommend movies"):
        result = recommend_for_user(user_id, artifacts, top_n)
        st.subheader(f"Recommendations for user {user_id}")
        st.dataframe(result[["title", "year", "genres", "score"]], use_container_width=True)
else:
    titles = artifacts["movies"]["title"].tolist()
    title = st.selectbox("Choose a movie", titles)
    if st.button("Find similar movies"):
        result = recommend_similar(title, artifacts, top_n)
        st.subheader(f"Movies similar to {title}")
        st.dataframe(result[["title", "year", "genres", "similarity"]], use_container_width=True)

st.caption("Educational demo using the included sample dataset. Recommendations are not endorsements.")
