# Movie Recommendation System

A beginner-friendly, end-to-end movie recommender built with Python, pandas, scikit-learn and Streamlit.

## Project structure

```text
movie_recommendation_system/
├── app.py
├── train_model.py
├── predict_movies.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.pdf
├── PROJECT_PRESENTATION.pptx
├── data/
│   ├── movies.csv
│   └── ratings.csv
├── models/
│   └── movie_recommender.pkl
└── outputs/
    ├── graphs/
    │   ├── rating_distribution.png
    │   └── most_rated_movies.png
    └── metrics/
        └── evaluation.json
```

## Setup

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py
```

This creates the serialized model, evaluation metrics and graphs.

## Get recommendations from the terminal

```bash
python predict_movies.py --user 1 --top 5
python predict_movies.py --movie "The Matrix" --top 5
```

## Run the web app

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit in your browser.

## Method

* **Content-based filtering:** TF-IDF vectors are created from movie genres and plot summaries. Cosine similarity finds movies with similar descriptions.
* **Rating-based personalization:** user ratings and movie-average ratings provide a simple preference signal for ranking unseen titles.
* **Evaluation:** an 80/20 random holdout split evaluates a movie-mean rating baseline with MAE and RMSE. These metrics evaluate the rating baseline, not the content similarity ranking.

## Dataset note

The included CSV files are a small, generated educational sample. They are not an official MovieLens release and should not be presented as real-world benchmark data. For a formal research project, replace them with a properly licensed, documented dataset and rerun training.

## Limitations and future improvements

The sample dataset is small; new users and movies have limited history; synthetic ratings do not represent real audience preferences. Future work could use MovieLens, matrix factorization, ranking metrics such as Precision@K/Recall@K, cold-start questionnaires and user feedback.
