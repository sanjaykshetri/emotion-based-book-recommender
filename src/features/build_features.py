"""
Feature Engineering for Emotion-Based Book Recommender

This script handles:
1. TF-IDF vectorization of book text (tags + titles)
2. Creating emotion-to-book feature mappings
3. Generating embeddings (optional, for advanced model)
4. Saving feature matrices for modeling

Author: Emotion-Based Book Recommender Project
Date: December 26, 2025
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """Build features for emotion-based book recommendations."""
    
    def __init__(self, processed_data_path="data/processed", features_path="data/processed/features"):
        self.processed_data_path = Path(processed_data_path)
        self.features_path = Path(features_path)
        self.features_path.mkdir(parents=True, exist_ok=True)
        
        self.tfidf_vectorizer = None
        self.scaler = None
    
    def load_data(self):
        """Load preprocessed train and test data."""
        print("📂 Loading preprocessed data...")
        
        train_path = self.processed_data_path / "books_train.csv"
        test_path = self.processed_data_path / "books_test.csv"
        
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        print(f"✅ Train set: {self.train_df.shape}")
        print(f"✅ Test set: {self.test_df.shape}")
        
        return self
    
    def build_tfidf_features(self, max_features=5000, ngram_range=(1, 2)):
        """
        Build TF-IDF features from combined text (tags + titles).
        
        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: N-gram range for TF-IDF (unigrams + bigrams)
        """
        print(f"\n🔤 Building TF-IDF features...")
        print(f"  - Max features: {max_features}")
        print(f"  - N-gram range: {ngram_range}")
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8,  # Ignore terms that appear in more than 80% of documents
            stop_words='english',
            sublinear_tf=True  # Apply sublinear tf scaling
        )
        
        # Fit on training data
        train_text = self.train_df['combined_text'].fillna('')
        self.X_train_tfidf = self.tfidf_vectorizer.fit_transform(train_text)
        
        # Transform test data
        test_text = self.test_df['combined_text'].fillna('')
        self.X_test_tfidf = self.tfidf_vectorizer.transform(test_text)
        
        print(f"✅ TF-IDF train shape: {self.X_train_tfidf.shape}")
        print(f"✅ TF-IDF test shape: {self.X_test_tfidf.shape}")
        print(f"  - Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        
        # Show top features
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        print(f"\n  Sample features: {list(feature_names[:20])}")
        
        return self
    
    def build_metadata_features(self):
        """Extract numerical metadata features (ratings, counts, etc.)."""
        print("\n📊 Building metadata features...")
        
        # Select numerical features
        metadata_cols = ['average_rating', 'ratings_count', 'work_ratings_count', 
                        'work_text_reviews_count', 'books_count']
        
        # Keep only available columns
        available_cols = [col for col in metadata_cols if col in self.train_df.columns]
        
        if available_cols:
            X_train_meta = self.train_df[available_cols].fillna(0)
            X_test_meta = self.test_df[available_cols].fillna(0)
            
            # Standardize
            self.scaler = StandardScaler()
            self.X_train_meta = self.scaler.fit_transform(X_train_meta)
            self.X_test_meta = self.scaler.transform(X_test_meta)
            
            print(f"✅ Metadata features: {available_cols}")
            print(f"  - Train shape: {self.X_train_meta.shape}")
            print(f"  - Test shape: {self.X_test_meta.shape}")
        else:
            print("⚠️ No metadata features available")
            self.X_train_meta = None
            self.X_test_meta = None
        
        return self
    
    def create_emotion_mappings(self):
        """
        Create emotion-to-feature mappings using predefined emotion keywords.
        This helps convert user emotions into searchable feature vectors.
        """
        print("\n💭 Creating emotion mappings...")
        
        # Define emotion categories and their keywords
        emotion_mappings = {
            'anxious': ['calm', 'peace', 'comfort', 'hope', 'uplifting', 'inspiring'],
            'sad': ['uplifting', 'hope', 'inspiring', 'happy', 'joy', 'comfort'],
            'lonely': ['love', 'friendship', 'comfort', 'heartwarming', 'connection'],
            'stressed': ['calm', 'peace', 'relaxing', 'escape', 'comfort'],
            'hopeless': ['hope', 'inspiring', 'uplifting', 'motivational', 'triumph'],
            'angry': ['calm', 'peace', 'understanding', 'cathartic'],
            'grief': ['comfort', 'healing', 'hope', 'understanding', 'peace'],
            'overwhelmed': ['calm', 'simple', 'peaceful', 'escape', 'comfort'],
            'happy': ['joy', 'fun', 'lighthearted', 'entertaining', 'uplifting'],
            'excited': ['adventure', 'thrilling', 'fast-paced', 'exciting']
        }
        
        # Convert emotion keywords to TF-IDF vectors
        emotion_vectors = {}
        
        for emotion, keywords in emotion_mappings.items():
            # Create a query string from keywords
            query = ' '.join(keywords)
            
            # Transform to TF-IDF space
            if self.tfidf_vectorizer:
                emotion_vec = self.tfidf_vectorizer.transform([query])
                emotion_vectors[emotion] = emotion_vec
        
        self.emotion_vectors = emotion_vectors
        
        print(f"✅ Created {len(emotion_vectors)} emotion mappings:")
        for emotion in emotion_mappings.keys():
            print(f"  - {emotion}")
        
        return self
    
    def save_features(self):
        """Save feature matrices and vectorizers."""
        print("\n💾 Saving features...")
        
        # Save TF-IDF features
        np.save(self.features_path / 'X_train_tfidf.npy', self.X_train_tfidf.toarray())
        np.save(self.features_path / 'X_test_tfidf.npy', self.X_test_tfidf.toarray())
        print(f"  ✅ Saved TF-IDF features")
        
        # Save metadata features if available
        if self.X_train_meta is not None:
            np.save(self.features_path / 'X_train_meta.npy', self.X_train_meta)
            np.save(self.features_path / 'X_test_meta.npy', self.X_test_meta)
            print(f"  ✅ Saved metadata features")
        
        # Save vectorizer
        with open(self.features_path / 'tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        print(f"  ✅ Saved TF-IDF vectorizer")
        
        # Save scaler if available
        if self.scaler:
            with open(self.features_path / 'scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"  ✅ Saved scaler")
        
        # Save emotion mappings
        with open(self.features_path / 'emotion_vectors.pkl', 'wb') as f:
            pickle.dump(self.emotion_vectors, f)
        print(f"  ✅ Saved emotion vectors")
        
        # Save feature info
        feature_info = {
            'tfidf_shape': self.X_train_tfidf.shape,
            'vocab_size': len(self.tfidf_vectorizer.vocabulary_),
            'max_features': self.tfidf_vectorizer.max_features,
            'emotions': list(self.emotion_vectors.keys())
        }
        
        if self.X_train_meta is not None:
            feature_info['metadata_shape'] = self.X_train_meta.shape
        
        with open(self.features_path / 'feature_info.pkl', 'wb') as f:
            pickle.dump(feature_info, f)
        print(f"  ✅ Saved feature info")
        
        print(f"\n📂 All features saved to: {self.features_path}")
        
        return self


def main():
    """Main feature engineering pipeline."""
    print("=" * 60)
    print("EMOTION-BASED BOOK RECOMMENDER - FEATURE ENGINEERING")
    print("=" * 60)
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Run pipeline
    fe.load_data()
    fe.build_tfidf_features(max_features=5000, ngram_range=(1, 2))
    fe.build_metadata_features()
    fe.create_emotion_mappings()
    fe.save_features()
    
    print("\n" + "=" * 60)
    print("✅ FEATURE ENGINEERING COMPLETE!")
    print("=" * 60)
    print("\nNext step: Train models (src/models/train_tfidf_knn.py)")


if __name__ == "__main__":
    main()
