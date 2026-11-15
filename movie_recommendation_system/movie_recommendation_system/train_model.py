"""
Train the Movie Recommendation System.
Run: python train_model.py
"""
from pathlib import Path
import json, pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_DIR = ROOT / "models"
METRICS_DIR = ROOT / "outputs" / "metrics"
GRAPH_DIR = ROOT / "outputs" / "graphs"
for folder in (MODEL_DIR, METRICS_DIR, GRAPH_DIR):
    folder.mkdir(parents=True, exist_ok=True)

movies = pd.read_csv(DATA / "movies.csv").fillna("")
ratings = pd.read_csv(DATA / "ratings.csv")

# Combine genres and plot text to create movie vectors.
movies["content"] = movies["genres"].str.replace("|", " ", regex=False) + " " + movies["overview"]
vectorizer = TfidfVectorizer(stop_words="english")
tfidf = vectorizer.fit_transform(movies["content"])
content_similarity = cosine_similarity(tfidf)

# Build a user-item matrix for a simple collaborative filtering component.
user_item = ratings.pivot_table(index="userId", columns="movieId", values="rating")
movie_means = ratings.groupby("movieId")["rating"].mean()
global_mean = float(ratings["rating"].mean())

# Hold-out evaluation for a transparent popularity/mean-rating baseline.
train, test = train_test_split(ratings, test_size=0.20, random_state=42)
train_means = train.groupby("movieId")["rating"].mean()
pred = test["movieId"].map(train_means).fillna(train["rating"].mean()).clip(1, 5)
mae = float(mean_absolute_error(test["rating"], pred))
rmse = float(np.sqrt(mean_squared_error(test["rating"], pred)))

artifacts = {
    "movies": movies.drop(columns=["content"]),
    "vectorizer": vectorizer,
    "tfidf": tfidf,
    "content_similarity": content_similarity,
    "user_item": user_item,
    "movie_means": movie_means,
    "global_mean": global_mean
}
with open(MODEL_DIR / "movie_recommender.pkl", "wb") as f:
    pickle.dump(artifacts, f)

metrics = {
    "evaluation": "80/20 random holdout; movie mean-rating baseline",
    "mae": mae,
    "rmse": rmse,
    "movies": int(len(movies)),
    "users": int(ratings["userId"].nunique()),
    "ratings": int(len(ratings))
}
(METRICS_DIR / "evaluation.json").write_text(json.dumps(metrics, indent=2))

# Rating distribution
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 4))
ratings["rating"].value_counts().sort_index().plot(kind="bar")
plt.title("Distribution of User Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "rating_distribution.png", dpi=150)
plt.close()

# Most frequently rated movies
plt.figure(figsize=(8, 5))
ratings["movieId"].value_counts().head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Most Rated Movies")
plt.xlabel("Number of ratings")
plt.ylabel("Movie ID")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "most_rated_movies.png", dpi=150)
plt.close()

print("Training complete.")
print(json.dumps(metrics, indent=2))
