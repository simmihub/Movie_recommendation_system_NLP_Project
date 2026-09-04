# 🎬 Movie Recommendation System

A **Content-Based Movie Recommendation System** built using Python and Natural Language Processing (NLP) techniques. The system recommends movies similar to a selected movie by analyzing its genres, overview, and tagline.

---

## 📌 Project Overview

The Movie Recommendation System recommends similar movies based on the textual information associated with each movie.

The project uses the **TMDB movie metadata dataset** containing information about movies such as title, genres, overview, tagline, ratings, and popularity.

A combined text feature called **`tag`** is created using:

* Movie Overview
* Genres
* Tagline

The text is then processed using NLP techniques and converted into numerical vectors using **TF-IDF Vectorization**. Finally, **Cosine Similarity** is used to find movies that are most similar to the selected movie.

---

## 🎯 Objectives

* Build a content-based movie recommendation system.
* Clean and preprocess movie metadata.
* Apply NLP techniques to movie text data.
* Convert text into numerical features using TF-IDF.
* Calculate similarity between movies using Cosine Similarity.
* Recommend the top similar movies for a selected movie.
* Save the trained components for future use.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Data loading and data manipulation
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Seaborn** – Data visualization
* **NLTK** – Natural Language Processing
* **Regular Expressions** – Text cleaning
* **Scikit-learn** – TF-IDF Vectorization and Cosine Similarity
* **Joblib** – Saving and loading project components
* **Jupyter Notebook** – Development environment

---

## 📂 Dataset

The project uses the **`movies_metadata.csv`** dataset.

The original dataset contains **45,466 records and 24 columns**.

Important columns used in this project include:

* `title`
* `genres`
* `overview`
* `tagline`
* `vote_average`
* `popularity`

The dataset was reduced to these relevant columns during preprocessing.

---

## 🧹 Data Cleaning & Preprocessing

The following preprocessing steps were performed:

1. Loaded the movie dataset using Pandas.
2. Checked the dataset structure and data types.
3. Checked missing values.
4. Checked duplicate records.
5. Removed duplicate records.
6. Selected relevant columns.
7. Removed rows with missing movie titles.
8. Filled missing overview values with an empty space.
9. Converted the genres data from its original structured format into readable genre names.
10. Filled missing tagline values.
11. Combined overview, genres, and tagline into a single `tag` column.

---

## 🧠 NLP Text Preprocessing

The `tag` column is processed using Natural Language Processing.

The following steps are applied:

* Convert text to lowercase
* Remove punctuation
* Remove English stopwords
* Apply lemmatization

This produces cleaner text that can be used for feature extraction.

### Example

Before preprocessing:

```text
Led by Woody, Andy's toys live happily in his room...
Animation Comedy Family
```

After preprocessing:

```text
led woody andys toy live happily room andys birthday brings buzz lightyear...
animation comedy family
```

---

## 🔢 TF-IDF Vectorization

After NLP preprocessing, the movie `tag` text is converted into numerical vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

The project uses:

```python
TfidfVectorizer(
    max_features=50000,
    ngram_range=(1,2),
    stop_words='english'
)
```

The resulting TF-IDF matrix contains **45,447 movies and 50,000 features**.

---

## 🔍 Cosine Similarity

**Cosine Similarity** is used to measure the similarity between movies.

When a movie is selected:

1. Its index is identified.
2. Its TF-IDF vector is compared with all movie vectors.
3. Similarity scores are calculated.
4. Movies are sorted according to their similarity scores.
5. The top similar movies are returned.

The recommendation function returns the top **10 similar movies by default**.

---

## 🎥 Recommendation Function

The main recommendation function is:

```python
recommend(title, n=10)
```

For example:

```python
recommend('Avatar', 5)
```

The system returns movies similar to the selected movie based on their textual content.

---

## 💾 Saved Project Components

The following components are saved using **Joblib**:

```text
tfidf.pkl
tfidf_matrix.pkl
indices.pkl
df.pkl
```

These files contain the trained TF-IDF vectorizer, TF-IDF matrix, movie indices, and processed dataframe respectively.

---

## 📊 Project Workflow

```text
Movie Dataset
      ↓
Data Cleaning
      ↓
Feature Selection
      ↓
Combine Overview + Genres + Tagline
      ↓
NLP Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
Find Similar Movies
      ↓
Movie Recommendations
```

---

## 🔮 Future Improvements

The project can be enhanced by adding:

* Movie posters
* Movie ratings and reviews
* Genre-based filtering
* Multiple recommendation criteria
* Personalized recommendations
* Interactive web interface
* Deployment using Streamlit
* Integration with a movie API for additional movie information

---

## 🎓 Key Concepts Demonstrated

This project demonstrates practical implementation of:

* Data Cleaning
* Exploratory Data Processing
* Feature Engineering
* Natural Language Processing
* Text Preprocessing
* Lemmatization
* TF-IDF Vectorization
* Cosine Similarity
* Content-Based Recommendation Systems
* Model/Data Serialization using Joblib

---

## 👩‍💻 Author

**Simran**

B.Tech CSE (AIML) Student

---

## ⭐ Conclusion

This project demonstrates how **Machine Learning and NLP techniques** can be used to build a content-based recommendation system. By converting movie information into TF-IDF vectors and measuring similarity using Cosine Similarity, the system can recommend movies with similar content to the user's selected movie.
