"""
Streamlit Web App - Emotion-Based Book Recommender

Interactive web interface for recommending books based on user emotions.
Supports both TF-IDF baseline and SBERT advanced models.

Author: Emotion-Based Book Recommender Project
Date: December 26, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')


# Page Configuration
st.set_page_config(
    page_title="Emotion-Based Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        background: #E3F2FD;
        border-radius: 20px;
        margin: 0.25rem;
        font-weight: 500;
    }
    .book-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1E88E5;
    }
    .metric-card {
        background: #E8F5E9;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
        import os
        import subprocess
        x_train_tfidf_path = features_path / "X_train_tfidf.npz"
        if not x_train_tfidf_path.exists():
            subprocess.run(["python", "src/features/build_features.py"], check=True)
    """Load both TF-IDF and SBERT models."""
    base_path = Path("data/processed")
    models_path = base_path / "models"
    features_path = base_path / "features"
    
    # Load train data
    train_df = pd.read_csv(base_path / "books_train.csv")
    
    # TF-IDF Model
    with open(features_path / 'tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open(features_path / 'emotion_vectors.pkl', 'rb') as f:
        tfidf_emotion_vectors = pickle.load(f)
    X_train_tfidf = np.load(features_path / 'X_train_tfidf.npz')['X']
    tfidf_knn = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute')
    tfidf_knn.fit(X_train_tfidf)
    
    # SBERT Model
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    sbert_embeddings = np.load(models_path / 'sbert_lite_embeddings.npy')
    with open(models_path / 'sbert_lite_emotion_embeddings.pkl', 'rb') as f:
        sbert_emotion_embeddings = pickle.load(f)
    sbert_books_df = pd.read_csv(models_path / 'sbert_lite_books.csv')
    
    return {
        'train_df': train_df,
        'tfidf_vectorizer': tfidf_vectorizer,
        'tfidf_emotion_vectors': tfidf_emotion_vectors,
        'X_train_tfidf': X_train_tfidf,
        'tfidf_knn': tfidf_knn,
        'sbert_model': sbert_model,
        'sbert_embeddings': sbert_embeddings,
        'sbert_emotion_embeddings': sbert_emotion_embeddings,
        'sbert_books_df': sbert_books_df
    }


def recommend_tfidf(emotion, models, n_recommendations=10, min_rating=3.5):
    """Get recommendations using TF-IDF baseline model."""
    if emotion.lower() not in models['tfidf_emotion_vectors']:
        return pd.DataFrame()
    
    emotion_vector = models['tfidf_emotion_vectors'][emotion.lower()]
    similarities = cosine_similarity(emotion_vector, models['X_train_tfidf'])[0]
    
    top_indices = similarities.argsort()[::-1]
    
    recommendations = []
    for idx in top_indices:
        book = models['train_df'].iloc[idx]
        if book['average_rating'] >= min_rating:
            recommendations.append({
                'title': book['title'],
                'authors': book['authors'],
                'average_rating': book['average_rating'],
                'ratings_count': book.get('ratings_count', 0),
                'tags': book.get('all_tags_clean', '')[:80],
                'similarity_score': similarities[idx]
            })
            if len(recommendations) >= n_recommendations:
                break
    
    return pd.DataFrame(recommendations)


def recommend_sbert(emotion, models, n_recommendations=10, min_rating=3.5):
    """Get recommendations using SBERT advanced model."""
    if emotion.lower() not in models['sbert_emotion_embeddings']:
        return pd.DataFrame()
    
    emotion_embedding = models['sbert_emotion_embeddings'][emotion.lower()].reshape(1, -1)
    similarities = cosine_similarity(emotion_embedding, models['sbert_embeddings'])[0]
    
    top_indices = similarities.argsort()[::-1]
    
    recommendations = []
    for idx in top_indices:
        book = models['sbert_books_df'].iloc[idx]
        if book['average_rating'] >= min_rating:
            recommendations.append({
                'title': book['title'],
                'authors': book['authors'],
                'average_rating': book['average_rating'],
                'ratings_count': book.get('ratings_count', 0),
                'tags': book.get('all_tags_clean', '')[:80],
                'similarity_score': similarities[idx]
            })
            if len(recommendations) >= n_recommendations:
                break
    
    return pd.DataFrame(recommendations)


def display_book_card(book, rank):
    """Display a single book recommendation card."""
    st.markdown(f"""
    <div class="book-card">
        <h3>#{rank} {book['title']}</h3>
        <p><strong>Author:</strong> {book['authors']}</p>
        <p><strong>Rating:</strong> ⭐ {book['average_rating']:.2f} ({int(book['ratings_count'])} ratings)</p>
        <p><strong>Match Score:</strong> {book['similarity_score']:.3f}</p>
        <p><strong>Tags:</strong> <em>{book['tags']}</em></p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">📚 Emotion-Based Book Recommender</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Find the perfect book for your emotional state</p>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading models... (this may take a moment on first run)"):
        models = load_models()
    
    # Sidebar - User Input
    st.sidebar.header("🎯 How are you feeling?")
    
    # Emotion selection
    emotions = {
        'Anxious 😰': 'anxious',
        'Sad 😢': 'sad',
        'Lonely 🥺': 'lonely',
        'Stressed 😫': 'stressed',
        'Hopeless 😔': 'hopeless',
        'Angry 😠': 'angry',
        'Grief 💔': 'grief',
        'Overwhelmed 🤯': 'overwhelmed',
        'Happy 😊': 'happy',
        'Excited 🎉': 'excited'
    }
    
    selected_emotion = st.sidebar.selectbox(
        "Select your current emotion:",
        list(emotions.keys())
    )
    
    emotion_key = emotions[selected_emotion]
    
    # Model selection
    st.sidebar.header("⚙️ Model Settings")
    model_type = st.sidebar.radio(
        "Choose recommendation model:",
        ["SBERT (Advanced) - Best Quality", "TF-IDF (Baseline) - Fast"],
        help="SBERT provides better semantic understanding, TF-IDF is faster"
    )
    
    # Number of recommendations
    n_recs = st.sidebar.slider(
        "Number of recommendations:",
        min_value=3,
        max_value=20,
        value=10,
        step=1
    )
    
    # Minimum rating filter
    min_rating = st.sidebar.slider(
        "Minimum book rating:",
        min_value=3.0,
        max_value=5.0,
        value=3.5,
        step=0.1
    )
    
    # Get Recommendations Button
    if st.sidebar.button("🔍 Get Recommendations", type="primary"):
        
        # Display selected emotion
        st.markdown(f"### Finding books for: {selected_emotion}")
        
        # Get recommendations based on model type
        with st.spinner("Finding the perfect books for you..."):
            if "SBERT" in model_type:
                recommendations = recommend_sbert(emotion_key, models, n_recs, min_rating)
                model_name = "SBERT Advanced"
            else:
                recommendations = recommend_tfidf(emotion_key, models, n_recs, min_rating)
                model_name = "TF-IDF Baseline"
        
        if recommendations.empty:
            st.warning("No recommendations found. Try adjusting your filters.")
            return
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(recommendations)}</h3>
                <p>Books Found</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            avg_rating = recommendations['average_rating'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>{avg_rating:.2f} ⭐</h3>
                <p>Avg Rating</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            avg_match = recommendations['similarity_score'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>{avg_match:.3f}</h3>
                <p>Avg Match Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display recommendations
        st.markdown(f"### 📖 Recommended Books ({model_name})")
        
        for idx, book in recommendations.iterrows():
            display_book_card(book, idx + 1)
        
        # Download results
        st.markdown("---")
        csv = recommendations.to_csv(index=False)
        st.download_button(
            label="📥 Download Recommendations (CSV)",
            data=csv,
            file_name=f"recommendations_{emotion_key}_{model_name}.csv",
            mime="text/csv"
        )
    
    # About section
    with st.sidebar.expander("ℹ️ About This App"):
        st.markdown("""
        This app recommends books based on your emotional state using machine learning.
        
        **Models:**
        - **SBERT**: Semantic embeddings for nuanced understanding
        - **TF-IDF**: Keyword-based matching (faster)
        
        **Dataset:** GoodBooks-10k (10,000 books with user ratings and tags)
        
        **Built with:** Python, Streamlit, scikit-learn, sentence-transformers
        """)
    
    # Model comparison
    with st.sidebar.expander("📊 Model Performance"):
        st.markdown("""
        **Evaluation Results (NDCG@10):**
        - SBERT: 0.817 ⭐
        - TF-IDF: 0.383
        
        SBERT provides **2x better ranking quality** and **15% more diverse** recommendations.
        """)


if __name__ == "__main__":
    main()
