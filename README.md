# 🎬 Netflix Recommendation System

A Netflix-style movie recommendation system built using **Machine Learning** and **Natural Language Processing (NLP)** techniques.  
The system suggests similar movies based on content such as genres and descriptions.

---

##🎬 App Preview
<img width="1902" height="867" alt="ui" src="https://github.com/user-attachments/assets/433d0592-ddb8-46ba-a214-154f725cdadc" />

---

## 🚀 Features

- 🔍 Search for any movie
- 🎯 Get top similar recommendations
- 🧠 Content-based filtering (not user-based)
- 🎬 Netflix-style UI using Streamlit
- 🖼️ Movie posters integration (TMDB API)

---

## 🧠 Tech Stack

- Python
- Streamlit
- Pandas & NumPy
- Scikit-learn
- Gensim (Word2Vec)
- Cosine Similarity
- TMDB API

---

## ⚙️ How It Works

1. Data preprocessing (cleaning & feature engineering)
2. Text representation using:
   - TF-IDF
   - Word2Vec embeddings
3. Dimensionality reduction using Truncated SVD
4. Clustering using K-Means (K = 12)
5. Recommendation using cosine similarity within clusters

---

## 📊 Dataset

- Source: Netflix Dataset (Kaggle)
- Features:
  - Title
  - Genre
  - Description
  - Release Year
- Type: Unsupervised Learning (No target variable)

---

## 🎯 Model Insights

- Word2Vec captured semantic similarity better than TF-IDF
- Clustering improved recommendation efficiency
- The system provides relevant and meaningful suggestions

---

## ⚠️ Limitations

- No user personalization (content-based only)
- Some descriptions are short or incomplete
- Cold start problem for new content

---

## 🔮 Future Work

- Integrate collaborative filtering
- Use advanced models like BERT
- Improve recommendation diversity
- Deploy full web application

---
🚀 **Live Demo:** [Click here to try the app](https://netflix-recommender-system29.streamlit.app/)
