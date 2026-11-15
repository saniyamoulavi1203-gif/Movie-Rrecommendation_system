"""
Command-line movie recommender.
Examples:
  python predict_movies.py --user 1 --top 5
  python predict_movies.py --movie "The Matrix" --top 5
"""
from pathlib import Path
import argparse, pickle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "movie_recommender.pkl"

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python train_model.py")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def recommend_for_user(user_id, artifacts, top_n=5):
    movies = artifacts["movies"].copy()
    user_item = artifacts["user_item"]
    if user_id not in user_item.index:
        raise ValueError(f"Unknown user {user_id}. Choose a user from ratings.csv.")
    seen = user_item.loc[user_id].dropna().index.tolist()
    scores = {}
    for mid in movies["movieId"]:
        if mid in seen:
            continue
        # Weighted average of ratings from similar users is approximated by
        # movie mean; content similarity to this user's highly rated movies
        # personalizes the ranking.
        base = artifacts["movie_means"].get(mid, artifacts["global_mean"])
        liked = user_item.loc[user_id].dropna()
        liked = liked[liked >= 4.0]
        idx_map = {m:i for i,m in enumerate(movies["movieId"])}
        if len(liked):
            target_idx = idx_map[mid]
            liked_idx = [idx_map[m] for m in liked.index if m in idx_map]
            sims = artifacts["content_similarity"][target_idx, liked_idx]
            personalization = float(np.average(liked.values, weights=sims + 1e-6))
            scores[mid] = 0.55 * base + 0.45 * personalization
        else:
            scores[mid] = base
    result = movies[movies["movieId"].isin(scores)].copy()
    result["score"] = result["movieId"].map(scores)
    return result.sort_values("score", ascending=False).head(top_n)

def recommend_similar(movie_title, artifacts, top_n=5):
    movies = artifacts["movies"].copy()
    matches = movies[movies["title"].str.contains(movie_title, case=False, regex=False)]
    if matches.empty:
        raise ValueError(f"No movie found matching: {movie_title}")
    idx = movies.index[movies["movieId"] == matches.iloc[0]["movieId"]][0]
    sims = artifacts["content_similarity"][idx]
    order = np.argsort(sims)[::-1]
    order = [i for i in order if i != idx][:top_n]
    result = movies.iloc[order].copy()
    result["similarity"] = sims[order]
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommend movies to a user or based on a movie title.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--user", type=int, help="User ID from ratings.csv")
    group.add_argument("--movie", type=str, help="Movie title to find similar movies")
    parser.add_argument("--top", type=int, default=5, help="Number of recommendations")
    args = parser.parse_args()
    model = load_model()
    output = recommend_for_user(args.user, model, args.top) if args.user else recommend_similar(args.movie, model, args.top)
    print(output[["title", "genres"] + (["score"] if "score" in output else ["similarity"])].to_string(index=False))
