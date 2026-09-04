import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Load Saved Files
# -----------------------------

df = joblib.load("df.pkl")
tfidf = joblib.load("tfidf.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")
indices = joblib.load("indices.pkl")

# -----------------------------
# Recommendation Function
# -----------------------------

def recommend(title, n=10):

    if title not in indices:
        return []

    idx = indices[title]

    sim_score = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    similar_idx = sim_score.argsort()[::-1][1:n+1]

    return df["title"].iloc[similar_idx].tolist()


# -----------------------------
# Streamlit Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title("🎬 Movie Recommendation System")

st.write(
    "Select a movie and get recommendations based on its similarity "
    "with other movies."
)


# -----------------------------
# Movie Selection
# -----------------------------

movie_list = sorted(
    df["title"].dropna().unique()
)

selected_movie = st.selectbox(
    "Select a movie",
    movie_list
)


# -----------------------------
# Number of Recommendations
# -----------------------------

number_of_movies = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)


# -----------------------------
# Recommend Button
# -----------------------------

if st.button("Recommend Movies"):

    recommendations = recommend(
        selected_movie,
        number_of_movies
    )

    if recommendations:

        st.subheader(
            f"Movies similar to {selected_movie}"
        )

        for i, movie in enumerate(
            recommendations,
            start=1
        ):
            st.write(f"### {i}. {movie}")

    else:

        st.error("Movie not found.")