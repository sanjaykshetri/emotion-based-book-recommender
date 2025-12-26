"""
Lightweight Sentence-BERT Embedding Recommender (Lite Version)

This script implements a lightweight version that works on smaller subsets
for demonstration and development purposes.

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


class SBERTEmotionRecommenderLite:
    """Lightweight SBERT recommender for testing and development."""
    
    def __init__(self, processed_data_path="data/processed", model_name='all-MiniLM-L6-v2', sample_size=2000):
        self.processed_data_path = Path(processed_data_path)
        self.model_name = model_name
        self.sample_size = sample_size
        self.sbert_model = None
        self.train_df = None
        self.train_embeddings = None
        self.emotion_embeddings = None
    
    def load_model(self):
        """Load pre-trained Sentence-BERT model."""
        print(f"📦 Loading Sentence-BERT model: {self.model_name}")
        self.sbert_model = SentenceTransformer(self.model_name)
        print(f"✅ Model loaded (dimension: {self.sbert_model.get_sentence_embedding_dimension()})")
        return self
    
    def load_data(self):
        """Load preprocessed train data (subset for speed)."""
        print(f"\n📂 Loading train data (sample size: {self.sample_size})...")
        
        full_df = pd.read_csv(self.processed_data_path / "books_train.csv")
        # Sample high-rated books for better recommendations
        self.train_df = full_df[full_df['average_rating'] >= 3.5].sample(
            n=min(self.sample_size, len(full_df)), 
            random_state=42
        ).reset_index(drop=True)
        
        print(f"✅ Train data: {self.train_df.shape}")
        return self
    
    def create_embeddings(self, batch_size=16):
        """Create Sentence-BERT embeddings for sampled books."""
        print(f"\n🔮 Creating embeddings for {len(self.train_df)} books...")
        
        # Prepare shortened texts
        texts = self.train_df['combined_text'].fillna('').apply(lambda x: x[:300]).tolist()
        
        print("   Encoding... (this will take a few minutes)")
        self.train_embeddings = self.sbert_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        print(f"✅ Embeddings: {self.train_embeddings.shape}")
        return self
    
    def create_emotion_embeddings(self):
        """Create embeddings for emotion descriptions."""
        print("\n💭 Creating emotion embeddings...")
        
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
        
        self.emotion_embeddings = {}
        for emotion, description in emotion_descriptions.items():
            embedding = self.sbert_model.encode([description], convert_to_numpy=True, normalize_embeddings=True)
            self.emotion_embeddings[emotion] = embedding[0]
        
        print(f"✅ Created embeddings for {len(self.emotion_embeddings)} emotions")
        return self
    
    def recommend_by_emotion(self, emotion, n_recommendations=10, min_rating=3.5):
        """Get book recommendations based on emotion."""
        if emotion.lower() not in self.emotion_embeddings:
            available = ', '.join(self.emotion_embeddings.keys())
            raise ValueError(f"Emotion '{emotion}' not found. Available: {available}")
        
        emotion_embedding = self.emotion_embeddings[emotion.lower()].reshape(1, -1)
        similarities = cosine_similarity(emotion_embedding, self.train_embeddings)[0]
        
        top_indices = similarities.argsort()[::-1]
        
        recommendations = []
        for idx in top_indices:
            book = self.train_df.iloc[idx]
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
    
    def recommend_by_book(self, book_title, n_recommendations=10):
        """Get similar book recommendations."""
        matching_books = self.train_df[
            self.train_df['title'].str.contains(book_title, case=False, na=False)
        ]
        
        if len(matching_books) == 0:
            print(f"⚠️ Book '{book_title}' not found")
            return pd.DataFrame()
        
        book_idx = matching_books.index[0]
        book_embedding = self.train_embeddings[book_idx].reshape(1, -1)
        
        similarities = cosine_similarity(book_embedding, self.train_embeddings)[0]
        top_indices = similarities.argsort()[::-1][1:n_recommendations+1]
        
        recommendations = []
        for idx in top_indices:
            book = self.train_df.iloc[idx]
            recommendations.append({
                'title': book['title'],
                'authors': book['authors'],
                'average_rating': book['average_rating'],
                'ratings_count': book.get('ratings_count', 0),
                'tags': book.get('all_tags_clean', '')[:80],
                'similarity_score': similarities[idx]
            })
        
        return pd.DataFrame(recommendations)
    
    def save_model(self, output_path="data/processed/models"):
        """Save embeddings and model info."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving model...")
        
        np.save(output_path / 'sbert_lite_embeddings.npy', self.train_embeddings)
        self.train_df.to_csv(output_path / 'sbert_lite_books.csv', index=False)
        
        with open(output_path / 'sbert_lite_emotion_embeddings.pkl', 'wb') as f:
            pickle.dump(self.emotion_embeddings, f)
        
        model_info = {
            'model_name': self.model_name,
            'embedding_dim': self.sbert_model.get_sentence_embedding_dimension(),
            'n_books': len(self.train_df),
            'emotions': list(self.emotion_embeddings.keys())
        }
        
        with open(output_path / 'sbert_lite_model_info.pkl', 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"✅ Model saved to: {output_path}")
        return self


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("LIGHTWEIGHT SBERT RECOMMENDER (2000 book sample for speed)")
    print("=" * 70)
    
    recommender = SBERTEmotionRecommenderLite(sample_size=2000)
    recommender.load_model()
    recommender.load_data()
    recommender.create_embeddings(batch_size=16)
    recommender.create_emotion_embeddings()
    
    # Demo 1: Anxious
    print("\n" + "-" * 70)
    print("📖 DEMO 1: Books for feeling ANXIOUS")
    print("-" * 70)
    anxious_recs = recommender.recommend_by_emotion('anxious', n_recommendations=5)
    print(anxious_recs[['title', 'authors', 'average_rating', 'similarity_score']].to_string(index=False))
    
    # Demo 2: Happy
    print("\n" + "-" * 70)
    print("📖 DEMO 2: Books for feeling HAPPY")
    print("-" * 70)
    happy_recs = recommender.recommend_by_emotion('happy', n_recommendations=5)
    print(happy_recs[['title', 'authors', 'average_rating', 'similarity_score']].to_string(index=False))
    
    # Demo 3: Similar books
    print("\n" + "-" * 70)
    print("📖 DEMO 3: Books similar to 'Harry Potter'")
    print("-" * 70)
    similar = recommender.recommend_by_book('Harry Potter', n_recommendations=5)
    if not similar.empty:
        print(similar[['title', 'authors', 'average_rating', 'similarity_score']].to_string(index=False))
    
    # Save
    recommender.save_model()
    
    print("\n" + "=" * 70)
    print("✅ LIGHTWEIGHT SBERT MODEL COMPLETE!")
    print("=" * 70)
    print("\nThis lite version uses 2000 books for faster training.")
    print("For production, use the full train_embedding_model.py with all 8000 books.")


if __name__ == "__main__":
    main()
