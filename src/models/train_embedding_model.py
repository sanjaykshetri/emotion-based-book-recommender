"""
Advanced Sentence-BERT Embedding Recommender Model

This script implements a semantic similarity-based recommendation system using:
- Sentence-BERT (all-MiniLM-L6-v2) for semantic embeddings
- Cosine similarity for finding semantically similar books
- Richer emotion-to-book mapping using sentence-level semantics

Author: Emotion-Based Book Recommender Project
Date: December 26, 2025
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class SBERTEmotionRecommender:
    """Advanced recommender using Sentence-BERT embeddings."""
    
    def __init__(self, processed_data_path="data/processed", model_name='all-MiniLM-L6-v2'):
        self.processed_data_path = Path(processed_data_path)
        self.model_name = model_name
        self.sbert_model = None
        self.train_df = None
        self.train_embeddings = None
        self.emotion_embeddings = None
    
    def load_model(self):
        """Load pre-trained Sentence-BERT model."""
        print(f"📦 Loading Sentence-BERT model: {self.model_name}")
        print("   (This may take a moment on first run...)")
        
        self.sbert_model = SentenceTransformer(self.model_name)
        
        print(f"✅ Model loaded: {self.model_name}")
        print(f"   Embedding dimension: {self.sbert_model.get_sentence_embedding_dimension()}")
        
        return self
    
    def load_data(self):
        """Load preprocessed train data."""
        print("\n📂 Loading train data...")
        
        self.train_df = pd.read_csv(self.processed_data_path / "books_train.csv")
        print(f"✅ Train data: {self.train_df.shape}")
        
        return self
    
    def create_embeddings(self, batch_size=16):
        """
        Create Sentence-BERT embeddings for all books.
        
        Args:
            batch_size: Number of texts to encode at once
        """
        print(f"\n🔮 Creating embeddings for {len(self.train_df)} books...")
        print(f"   Batch size: {batch_size}")
        print(f"   Device: {self.sbert_model.device}")
        
        # Prepare text (use combined_text which has title + tags)
        texts = self.train_df['combined_text'].fillna('').tolist()
        
        # Limit text length to avoid memory issues
        texts = [text[:512] if len(text) > 512 else text for text in texts]
        
        # Generate embeddings
        print("   Encoding texts (this may take a few minutes)...")
        self.train_embeddings = self.sbert_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for better cosine similarity
        )
        
        print(f"✅ Embeddings created: {self.train_embeddings.shape}")
        
        return self
    
    def create_emotion_embeddings(self):
        """Create embeddings for emotion descriptions."""
        print("\n💭 Creating emotion embeddings...")
        
        # Define richer emotion descriptions (more context than keywords)
        emotion_descriptions = {
            'anxious': 'calming peaceful soothing comforting hopeful inspiring reassuring gentle',
            'sad': 'uplifting heartwarming joyful hopeful inspiring comforting healing light',
            'lonely': 'connection friendship love companionship heartwarming community belonging',
            'stressed': 'relaxing calming peaceful escape simple lighthearted soothing comfort',
            'hopeless': 'hopeful inspiring uplifting motivational triumphant encouraging empowering',
            'angry': 'calming peaceful understanding cathartic healing soothing gentle',
            'grief': 'comforting healing hopeful understanding peaceful moving forward acceptance',
            'overwhelmed': 'simple calming peaceful gentle escape comforting straightforward easy',
            'happy': 'joyful fun entertaining lighthearted delightful uplifting cheerful',
            'excited': 'adventure thrilling exciting fast-paced action engaging dynamic'
        }
        
        # Create embeddings for each emotion
        self.emotion_embeddings = {}
        
        for emotion, description in emotion_descriptions.items():
            embedding = self.sbert_model.encode([description], convert_to_numpy=True)
            self.emotion_embeddings[emotion] = embedding[0]
        
        print(f"✅ Created embeddings for {len(self.emotion_embeddings)} emotions")
        
        return self
    
    def recommend_by_emotion(self, emotion, n_recommendations=10, min_rating=3.5):
        """
        Get book recommendations based on user's emotion using semantic similarity.
        
        Args:
            emotion: User's emotional state
            n_recommendations: Number of books to recommend
            min_rating: Minimum average rating threshold
            
        Returns:
            DataFrame with recommended books
        """
        print(f"\n🎯 Getting recommendations for emotion: '{emotion}'")
        
        # Get emotion embedding
        if emotion.lower() not in self.emotion_embeddings:
            available = ', '.join(self.emotion_embeddings.keys())
            raise ValueError(f"Emotion '{emotion}' not found. Available: {available}")
        
        emotion_embedding = self.emotion_embeddings[emotion.lower()].reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(emotion_embedding, self.train_embeddings)[0]
        
        # Get top indices
        top_indices = similarities.argsort()[::-1]
        
        # Filter and collect recommendations
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
                    'tags': book.get('all_tags_clean', '')[:100],
                    'similarity_score': similarities[idx]
                })
                
                if len(recommendations) >= n_recommendations:
                    break
        
        recommendations_df = pd.DataFrame(recommendations)
        
        print(f"✅ Found {len(recommendations_df)} recommendations")
        
        return recommendations_df
    
    def recommend_by_book(self, book_title, n_recommendations=10):
        """
        Get similar book recommendations based on semantic similarity.
        
        Args:
            book_title: Title of the reference book
            n_recommendations: Number of similar books to recommend
            
        Returns:
            DataFrame with recommended books
        """
        print(f"\n📚 Finding books similar to: '{book_title}'")
        
        # Find the book
        matching_books = self.train_df[
            self.train_df['title'].str.contains(book_title, case=False, na=False)
        ]
        
        if len(matching_books) == 0:
            print(f"⚠️ Book '{book_title}' not found")
            return pd.DataFrame()
        
        # Use first match
        book_idx = matching_books.index[0]
        book_embedding = self.train_embeddings[book_idx].reshape(1, -1)
        
        print(f"  Found: {matching_books.iloc[0]['title']}")
        
        # Calculate similarities
        similarities = cosine_similarity(book_embedding, self.train_embeddings)[0]
        
        # Get top indices (excluding the book itself)
        top_indices = similarities.argsort()[::-1][1:n_recommendations+1]
        
        # Collect recommendations
        recommendations = []
        for idx in top_indices:
            book = self.train_df.iloc[idx]
            recommendations.append({
                'title': book['title'],
                'authors': book['authors'],
                'average_rating': book['average_rating'],
                'ratings_count': book.get('ratings_count', 0),
                'tags': book.get('all_tags_clean', '')[:100],
                'similarity_score': similarities[idx]
            })
        
        recommendations_df = pd.DataFrame(recommendations)
        
        print(f"✅ Found {len(recommendations_df)} similar books")
        
        return recommendations_df
    
    def save_model(self, output_path="data/processed/models"):
        """Save embeddings and model info."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving embeddings and model...")
        
        # Save embeddings
        np.save(output_path / 'sbert_train_embeddings.npy', self.train_embeddings)
        print(f"  ✅ Saved train embeddings")
        
        # Save emotion embeddings
        with open(output_path / 'sbert_emotion_embeddings.pkl', 'wb') as f:
            pickle.dump(self.emotion_embeddings, f)
        print(f"  ✅ Saved emotion embeddings")
        
        # Save model info
        model_info = {
            'model_name': self.model_name,
            'embedding_dim': self.sbert_model.get_sentence_embedding_dimension(),
            'n_books': len(self.train_df),
            'emotions': list(self.emotion_embeddings.keys())
        }
        
        with open(output_path / 'sbert_model_info.pkl', 'wb') as f:
            pickle.dump(model_info, f)
        print(f"  ✅ Saved model info")
        
        print(f"📂 Model saved to: {output_path}")
        
        return self


def demo_sbert_recommendations():
    """Demonstrate the SBERT recommender."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: SBERT Emotion-Based Recommendations")
    print("=" * 60)
    
    # Initialize recommender
    recommender = SBERTEmotionRecommender()
    recommender.load_model()
    recommender.load_data()
    recommender.create_embeddings(batch_size=16)  # Smaller batch
    recommender.create_emotion_embeddings()
    
    # Demo 1: Anxious emotion
    print("\n" + "-" * 60)
    print("Demo 1: Books for someone feeling anxious")
    print("-" * 60)
    anxious_recs = recommender.recommend_by_emotion('anxious', n_recommendations=5)
    print("\n", anxious_recs[['title', 'authors', 'average_rating', 'similarity_score']])
    
    # Demo 2: Happy emotion
    print("\n" + "-" * 60)
    print("Demo 2: Books for someone feeling happy")
    print("-" * 60)
    happy_recs = recommender.recommend_by_emotion('happy', n_recommendations=5)
    print("\n", happy_recs[['title', 'authors', 'average_rating', 'similarity_score']])
    
    # Demo 3: Similar books
    print("\n" + "-" * 60)
    print("Demo 3: Books similar to 'Pride and Prejudice'")
    print("-" * 60)
    similar = recommender.recommend_by_book('Pride and Prejudice', n_recommendations=5)
    if not similar.empty:
        print("\n", similar[['title', 'authors', 'average_rating', 'similarity_score']])
    
    # Save model
    recommender.save_model()
    
    return recommender


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("ADVANCED MODEL: SENTENCE-BERT RECOMMENDER")
    print("=" * 60)
    
    # Initialize and train
    recommender = SBERTEmotionRecommender()
    recommender.load_model()
    recommender.load_data()
    recommender.create_embeddings(batch_size=16)  # Smaller batch for stability
    recommender.create_emotion_embeddings()
    recommender.save_model()
    
    print("\n" + "=" * 60)
    print("✅ ADVANCED MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print("\nModel capabilities:")
    print("  - Semantic emotion-based recommendations")
    print("  - Deeper understanding of emotional context")
    print("  - Better handling of nuanced emotions")
    print("\nNext step: Run demo or deploy to Streamlit")


if __name__ == "__main__":
    # Run main training
    main()
    
    # Optionally run demo
    print("\n" + "=" * 60)
    print("Running demonstration...")
    print("=" * 60)
    demo_sbert_recommendations()
