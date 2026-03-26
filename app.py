import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import requests
from PIL import Image
import os

# ============================================
# 🎨 PAGE CONFIGURATION - Netflix Style
# ============================================
st.set_page_config(
    page_title="Netflix Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# 🎨 CUSTOM CSS - Netflix Theme
# ============================================
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(180deg, #141414 0%, #000000 100%);
        color: #ffffff;
    }
    
    /* Netflix Red Color */
    .netflix-red {
        color: #E50914;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #E50914;
        color: white;
        border: none;
        border-radius: 10px;
        # padding: 6px 16px ;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.5s ease;
        display: flex;
        justify-content: center;
        align-items: center;
        
    }
    
   
            
    .block-container {
    padding-top: 0.5rem !important;
    }
            
    /* Search Box */
    .stTextInput > div > div > input {
        background-color: #333333;
        color: white;
        border: 1px solid #333333;
        border-radius: 4px;
        
        align-self: center;
       
    }
    
    /* Movie Cards */
    .movie-card {
    
        padding: 15px;
        margin: 10px 0;
       
    }
    

    
    /* Movie Title */
    .movie-title {
        color: #ffffff;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 8px;
    }
    
    /* Movie Info */
    .movie-info {
        color: #b3b3b3;
        font-size: 14px;
        margin-bottom: 5px;
    }
    
    /* Genres Tags */
    .genre-tag {
        display: inline-block;
        background-color: #333333;
        color: #ffffff;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        margin: 2px;
    }
    
    /* Keywords */
    .keyword-tag {
        display: inline-block;
        background-color: #E50914;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 11px;
        margin: 2px;
    }
    
    /* Header */
    .main-header {
        font-size: 48px;
        font-weight: bold;
        color: #E50914;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Subtitle */
    .subtitle {
        font-size: 20px;
        color: #b3b3b3;
        text-align: center;
        margin-bottom: 40px;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
        margin: 10px 0 10px 0;
        border-left: 4px solid #E50914;
        padding-left: 15px;
    }
    
    /* Similarity Score */
    .similarity-score {
        color: #46d369;
        font-weight: bold;
        font-size: 14px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #b3b3b3;
        font-size: 14px;
        margin-top: 50px;
        border-top: 1px solid #333;
    }
    
    div[data-baseweb="slider"] {
    transform: scale(0.75);
    margin-top: -10px;
            }
            
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================
# 💾 LOAD MODELS AND DATA
# ============================================
@st.cache_resource
def load_models():
    """Load trained models from files"""
    try:
       from gensim.models import Word2Vec
       w2v_model = Word2Vec.load('models/w2v_model.model')
       with open('models/X_embed.pkl', 'rb') as f:
            X_embed = pickle.load(f)
       with open('models/df_processed.pkl', 'rb') as f:
            df = pickle.load(f)
       return w2v_model, X_embed, df
    except FileNotFoundError:
        st.error("❌ Model files not found! Please ensure models are saved in 'models/' folder")
        return None, None, None

# ============================================
# 🎬 RECOMMENDATION FUNCTION
# ============================================
def recommend_movies_w2v(movie_title, df, X_embed, top_n=10):
    """Get movie recommendations based on Word2Vec embeddings"""
    
    if movie_title not in df['title'].values:
        return None, "Movie not found!"
    
    movie_idx = df[df['title'] == movie_title].index[0]
    movie_cluster = df.loc[movie_idx, 'cluster_embed']
    
    # Get movies from same cluster
    cluster_indices = df[df['cluster_embed'] == movie_cluster].index
    cluster_vectors = X_embed[cluster_indices]
    movie_vector = X_embed[movie_idx].reshape(1, -1)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(movie_vector, cluster_vectors).flatten()
    
    # Sort by similarity
    sorted_idx = np.argsort(similarities)[::-1]
    similar_indices = cluster_indices[sorted_idx]
    
    # Remove the movie itself
    similar_indices = similar_indices[similar_indices != movie_idx]
    
    # Get top N recommendations
    top_movies = df.loc[similar_indices].drop_duplicates(subset=['title']).head(top_n).copy()
    top_movies['similarity_score'] = similarities[sorted_idx[1:top_n+1]]
    
    return top_movies, None

# ============================================
# 🖼️ GET MOVIE POSTER (TMDB API)
# ============================================
@st.cache_data(show_spinner=False) 
def get_movie_poster(movie_title):
    """Fetch movie poster from TMDB API"""
    try:
        api_key = st.secrets["TMDB_API_KEY"]
        # api_key = "91b63d68cc6bed2e23dec869267c81e8"  # Replace with your TMDB API key
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"
        response = requests.get(search_url, timeout=3)
        data = response.json()
        
        if data['results']:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
        return "https://image.tmdb.org/t/p/w500_and_h282_face_filter(blur)/x2LSRK2Cm7MZhjluni1msVJ3wDF.jpg"
    except:
        return "https://image.tmdb.org/t/p/w500_and_h282_face_filter(blur)/x2LSRK2Cm7MZhjluni1msVJ3wDF.jpg"

# ============================================
# 🎬 DISPLAY MOVIE CARD
# ============================================
def display_movie_card(movie_row, is_selected=False, similarity=None):
    """Display a movie card with poster and info"""
    st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Movie Poster
        poster_url = get_movie_poster(movie_row['title'])
        st.image(poster_url,width=200)
    
    with col2:
        # Movie Title
        st.markdown(f"<div class='movie-title'>{movie_row['title']} ({movie_row.get('release_year', 'N/A')})</div>", 
                   unsafe_allow_html=True)
        
        # Similarity Score (for recommendations)
        if similarity is not None:
            st.markdown(f"<div class='similarity-score'>🎯 Match: {similarity*100:.1f}%</div>", 
                       unsafe_allow_html=True)
        
        # Genres
        if 'listed_in' in movie_row:
            genres = str(movie_row['listed_in']).split(',')[:3]
            genres_html = ''.join([f"<span class='genre-tag'>{g.strip()}</span>" for g in genres])
            st.markdown(f"<div class='movie-info'>📁 Genres: {genres_html}</div>", 
                       unsafe_allow_html=True)
        
        # Key Words (from tokens)
        if 'tokens' in movie_row and movie_row['tokens']:
            keywords = movie_row['tokens'][:5]
            keywords_html = ''.join([f"<span class='keyword-tag'>{k}</span>" for k in keywords])
            st.markdown(f"<div class='movie-info'>🔑 Key Words: {keywords_html}</div>", 
                       unsafe_allow_html=True)
        
        # Description
        if 'description' in movie_row:
            st.markdown(f"<div class='movie-info'>📝 {movie_row['description'][:200]}...</div>", 
                       unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# 🏠 MAIN APP
# ============================================
def main():
    # Header
    st.markdown("<div class='main-header' style='font-size:50px;'>🎬 Netflix Recommender</div>", unsafe_allow_html=True)               
        
    # ✅ Load models
    with st.spinner('🔄 Loading models...'):
        w2v_model, X_embed, df = load_models()

    if df is None:
        st.stop()

    # إزالة التكرار
    df = df.drop_duplicates(subset=['title']).reset_index(drop=True)

    # 🔍 Search in center
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3,4,3])

    with col2:
        search_query = st.text_input("", placeholder="🔍 Search for a movie...")

        movie_list = df['title'].sort_values().tolist()
        selected_movie = st.selectbox("choose a movie", movie_list)

        top_n = st.slider("Number of recommendations", 5, 20, 10)

        search_button = st.button("🎬 Recommend", use_container_width=True)



    
    # Main Content Area
    if search_button:
        movie_to_search = search_query if search_query else selected_movie
        
        if movie_to_search:
            # Get recommendations
            recommendations, error = recommend_movies_w2v(movie_to_search, df, X_embed, top_n)
            
            if error:
                st.error(f"❌ {error}")
            else:
                # Get selected movie info
                selected_movie_data = df[df['title'] == movie_to_search].iloc[0]
                
                # Display selected movie
                st.markdown("<div class='section-header'>🎬 Selected Movie</div>", unsafe_allow_html=True)
                display_movie_card(selected_movie_data, is_selected=True)
                
                # Display recommendations
                st.markdown("<div class='section-header'>💡 Because you watched this...</div>", unsafe_allow_html=True)
                
                # Display recommendations in grid
                for idx, row in recommendations.iterrows():
                    with st.container():
                        display_movie_card(row, similarity=row.similarity_score)
    
    else:
        # Welcome message
        st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h2 style='color: #ffffff;'>🍿 Ready to discover your next favorite movie?</h2>
            <p style='color: #b3b3b3; font-size: 18px;'>
                Select a movie from the sidebar and let our AI recommend similar movies for you!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show trending movies (random sample)
        st.markdown("<div class='section-header'>🔥 Trending Now</div>", unsafe_allow_html=True)
        
        trending_movies = df.sample(6)
        cols = st.columns(3)
        
        for idx, row in enumerate(trending_movies.itertuples()):
            with cols[idx % 3]:
                poster_url = get_movie_poster(row.title)
                st.image(poster_url, use_container_width=True)
                st.markdown(f"**{row.title}**")
                st.caption(f"📅 {getattr(row, 'release_year', 'N/A')}")
    
    # Footer
    st.markdown("""
    <div class='footer'>
        <p>Powered by Machine Learning & NLP | Word2Vec + Cosine Similarity</p>
        <p>© 2024 Netflix Recommender System</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# 🚀 RUN APP
# ============================================
if __name__ == "__main__":
    main()

