"""
Baseline TF-IDF + KNN Recommender Model

This script implements a content-based recommendation system using:
- TF-IDF features from book tags and titles
- Cosine similarity for finding similar books
- Emotion-to-book mapping using predefined emotion vectors

Author: Emotion-Based Book Recommender Project
Date: December 26, 2025
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


class TFIDFEmotionRecommender:
    """Baseline recommender using TF-IDF and cosine similarity."""
    
    def __init__(self, features_path="data/processed/features", processed_data_path="data/processed"):
        self.features_path = Path(features_path)
        self.processed_data_path = Path(processed_data_path)
        self.model = None
        self.tfidf_vectorizer = None
        self.emotion_vectors = None
        self.train_df = None
        self.X_train_tfidf = None
    
    def load_data(self):
        """Load features and train data."""
        print("📂 Loading data and features...")
        
        # Load train dataframe
        self.train_df = pd.read_csv(self.processed_data_path / "books_train.csv")
        print(f"✅ Train data: {self.train_df.shape}")
        
        # Load TF-IDF features
        self.X_train_tfidf = np.load(self.features_path / "X_train_tfidf.npy")
        print(f"✅ TF-IDF features: {self.X_train_tfidf.shape}")
        
        # Load vectorizer
        with open(self.features_path / 'tfidf_vectorizer.pkl', 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        print(f"✅ Loaded TF-IDF vectorizer")
        
        # Load emotion vectors
        with open(self.features_path / 'emotion_vectors.pkl', 'rb') as f:
            self.emotion_vectors = pickle.load(f)
        print(f"✅ Loaded {len(self.emotion_vectors)} emotion vectors")
        
        return self
    
    def build_model(self, n_neighbors=20, metric='cosine'):
        """
        Build KNN model for finding similar books.
        
        Args:
            n_neighbors: Number of neighbors to consider
            metric: Distance metric ('cosine', 'euclidean', etc.)
        """
        print(f"\n🔨 Building KNN model...")
        print(f"  - Number of neighbors: {n_neighbors}")
        print(f"  - Distance metric: {metric}")
        
        self.model = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=metric,
            algorithm='brute'  # Better for high-dimensional sparse data
        )
        
        self.model.fit(self.X_train_tfidf)
        
        print(f"✅ Model trained on {self.X_train_tfidf.shape[0]} books")
        
        return self
    
    def recommend_by_emotion(self, emotion, n_recommendations=10, min_rating=3.5):
        """
        Get book recommendations based on user's emotion.
        
        Args:
            emotion: User's emotional state (e.g., 'anxious', 'sad', 'happy')
            n_recommendations: Number of books to recommend
            min_rating: Minimum average rating threshold
            
        Returns:
            DataFrame with recommended books
        """
        print(f"\n🎯 Getting recommendations for emotion: '{emotion}'")
        
        # Get emotion vector
        if emotion.lower() not in self.emotion_vectors:
            available = ', '.join(self.emotion_vectors.keys())
            raise ValueError(f"Emotion '{emotion}' not found. Available: {available}")
        
        emotion_vector = self.emotion_vectors[emotion.lower()].toarray()
        
        # Find similar books using cosine similarity
        similarities = cosine_similarity(emotion_vector, self.X_train_tfidf)[0]
        
        # Get top indices
        top_indices = similarities.argsort()[::-1]
        
        # Filter by minimum rating and get recommendations
        recommendations = []
        for idx in top_indices:
            book = self.train_df.iloc[idx]
            
            # Apply quality filters
            if book['average_rating'] >= min_rating:
                recommendations.append({
                    'title': book['title'],
                    'authors': book['authors'],
                    'average_rating': book['average_rating'],
                    'ratings_count': book.get('ratings_count', 0),
                    'tags': book.get('all_tags_clean', '')[:100],  # First 100 chars
                    'similarity_score': similarities[idx]
                })
                
                if len(recommendations) >= n_recommendations:
                    break
        
        recommendations_df = pd.DataFrame(recommendations)
        
        print(f"✅ Found {len(recommendations_df)} recommendations")
        
        return recommendations_df
    
    def recommend_by_book(self, book_title, n_recommendations=10):
        """
        Get similar book recommendations based on a given book.
        
        Args:
            book_title: Title of the reference book
            n_recommendations: Number of similar books to recommend
            
        Returns:
            DataFrame with recommended books
        """
        print(f"\n📚 Finding books similar to: '{book_title}'")
        
        # Find the book in train set
        matching_books = self.train_df[
            self.train_df['title'].str.contains(book_title, case=False, na=False)
        ]
        
        if len(matching_books) == 0:
            print(f"⚠️ Book '{book_title}' not found in training set")
            return pd.DataFrame()
        
        # Use the first match
        book_idx = matching_books.index[0]
        book_vector = self.X_train_tfidf[book_idx].reshape(1, -1)
        
        print(f"  Found: {matching_books.iloc[0]['title']}")
        
        # Find nearest neighbors
        distances, indices = self.model.kneighbors(book_vector, n_neighbors=n_recommendations+1)
        
        # Skip the first one (itself)
        recommendations = []
        for i, idx in enumerate(indices[0][1:]):
            book = self.train_df.iloc[idx]
            recommendations.append({
                'title': book['title'],
                'authors': book['authors'],
                'average_rating': book['average_rating'],
                'ratings_count': book.get('ratings_count', 0),
                'tags': book.get('all_tags_clean', '')[:100],
                'similarity_score': 1 - distances[0][i+1]  # Convert distance to similarity
            })
        
        recommendations_df = pd.DataFrame(recommendations)
        
        print(f"✅ Found {len(recommendations_df)} similar books")
        
        return recommendations_df
    
    def save_model(self, output_path="data/processed/models"):
        """Save the trained model."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving model...")
        
        # Save KNN model
        with open(output_path / 'tfidf_knn_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"  ✅ Saved KNN model")
        print(f"📂 Model saved to: {output_path}")
        
        return self


def demo_recommendations():
    """Demonstrate the recommender system."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Emotion-Based Recommendations")
    print("=" * 60)
    
    # Initialize recommender
    recommender = TFIDFEmotionRecommender()
    recommender.load_data()
    recommender.build_model(n_neighbors=20)
    
    # Demo 1: Recommend for "anxious" emotion
    print("\n" + "-" * 60)
    print("Demo 1: Books for someone feeling anxious")
    print("-" * 60)
    anxious_recs = recommender.recommend_by_emotion('anxious', n_recommendations=5)
    print("\n", anxious_recs[['title', 'authors', 'average_rating', 'similarity_score']])
    
    # Demo 2: Recommend for "sad" emotion
    print("\n" + "-" * 60)
    print("Demo 2: Books for someone feeling sad")
    print("-" * 60)
    sad_recs = recommender.recommend_by_emotion('sad', n_recommendations=5)
    print("\n", sad_recs[['title', 'authors', 'average_rating', 'similarity_score']])
    
    # Demo 3: Find similar books
    print("\n" + "-" * 60)
    print("Demo 3: Books similar to 'Harry Potter'")
    print("-" * 60)
    similar_books = recommender.recommend_by_book('Harry Potter', n_recommendations=5)
    if not similar_books.empty:
        print("\n", similar_books[['title', 'authors', 'average_rating', 'similarity_score']])
    
    # Save model
    recommender.save_model()
    
    return recommender


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("BASELINE MODEL: TF-IDF + KNN RECOMMENDER")
    print("=" * 60)
    
    # Initialize and train
    recommender = TFIDFEmotionRecommender()
    recommender.load_data()
    recommender.build_model(n_neighbors=20)
    recommender.save_model()
    
    print("\n" + "=" * 60)
    print("✅ BASELINE MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print("\nModel capabilities:")
    print("  - Emotion-based recommendations (10 emotions)")
    print("  - Book-to-book similarity search")
    print("  - Quality filtering by ratings")
    print("\nNext step: Run demo or build advanced model")
    print("  - Demo: python -c 'from src.models.train_tfidf_knn import demo_recommendations; demo_recommendations()'")
    print("  - Advanced: python src/models/train_embedding_model.py")


if __name__ == "__main__":
    # Run main training
    main()
    
    # Optionally run demo
    print("\n" + "=" * 60)
    print("Running demonstration...")
    print("=" * 60)
    demo_recommendations()
