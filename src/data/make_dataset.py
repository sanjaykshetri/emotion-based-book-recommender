"""
Data Preprocessing Pipeline for Emotion-Based Book Recommender

This script handles:
1. Loading raw and processed data
2. Text cleaning and normalization
3. Handling missing values
4. Creating train/test splits
5. Saving processed datasets for modeling

Author: Emotion-Based Book Recommender Project
Date: December 26, 2025
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """Preprocess book data for emotion-based recommendations."""
    
    def __init__(self, raw_data_path="data/raw", processed_data_path="data/processed"):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load the enriched books dataset."""
        print("📂 Loading processed data...")
        books_enriched_path = self.processed_data_path / "books_enriched.csv"
        
        if books_enriched_path.exists():
            self.df = pd.read_csv(books_enriched_path)
            print(f"✅ Loaded books_enriched.csv: {self.df.shape}")
        else:
            raise FileNotFoundError(
                "books_enriched.csv not found. Run the EDA notebook first to create it."
            )
        
        return self
    
    def clean_text(self, text):
        """Clean and normalize text data."""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove special characters but keep spaces and commas
        text = re.sub(r'[^a-z0-9\s,]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def preprocess_features(self):
        """Clean and prepare features for modeling."""
        print("\n🧹 Preprocessing features...")
        
        # Clean tags
        print("  - Cleaning tags...")
        self.df['all_tags'] = self.df['all_tags'].fillna('')
        self.df['all_tags_clean'] = self.df['all_tags'].apply(self.clean_text)
        
        # Clean titles
        print("  - Cleaning titles...")
        self.df['title_clean'] = self.df['title'].apply(self.clean_text)
        
        # Clean authors
        print("  - Cleaning authors...")
        self.df['authors_clean'] = self.df['authors'].apply(self.clean_text)
        
        # Create combined text feature (title + tags for richer representation)
        print("  - Creating combined text features...")
        self.df['combined_text'] = (
            self.df['title_clean'] + ' ' + 
            self.df['all_tags_clean']
        ).str.strip()
        
        # Handle missing ratings
        self.df['average_rating'] = self.df['average_rating'].fillna(
            self.df['average_rating'].median()
        )
        
        # Create quality flag (books with sufficient ratings)
        if 'ratings_count' in self.df.columns:
            self.df['is_popular'] = (self.df['ratings_count'] >= 100).astype(int)
        
        print(f"✅ Preprocessing complete. Dataset shape: {self.df.shape}")
        
        return self
    
    def filter_quality_books(self, min_tags=1, min_rating_count=50):
        """Filter for books with sufficient data quality."""
        print(f"\n🔍 Filtering for quality books...")
        print(f"  - Minimum tags: {min_tags}")
        print(f"  - Minimum rating count: {min_rating_count}")
        
        original_size = len(self.df)
        
        # Keep books with at least some tags
        self.df = self.df[self.df['all_tags_clean'].str.len() > 0]
        
        # Keep books with sufficient ratings (if column exists)
        if 'ratings_count' in self.df.columns:
            self.df = self.df[self.df['ratings_count'] >= min_rating_count]
        
        filtered_size = len(self.df)
        removed = original_size - filtered_size
        
        print(f"✅ Filtered: {filtered_size} books retained, {removed} removed")
        
        return self
    
    def create_train_test_split(self, test_size=0.2, random_state=42):
        """Split data into train and test sets."""
        print(f"\n✂️ Creating train/test split ({int((1-test_size)*100)}/{int(test_size*100)})...")
        
        train_df, test_df = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
        
        print(f"  - Train set: {len(train_df)} books")
        print(f"  - Test set: {len(test_df)} books")
        
        return train_df, test_df
    
    def save_processed_data(self, train_df, test_df):
        """Save processed datasets."""
        print("\n💾 Saving processed datasets...")
        
        # Save full processed dataset
        full_path = self.processed_data_path / "books_processed.csv"
        self.df.to_csv(full_path, index=False)
        print(f"  ✅ Saved: {full_path}")
        
        # Save train set
        train_path = self.processed_data_path / "books_train.csv"
        train_df.to_csv(train_path, index=False)
        print(f"  ✅ Saved: {train_path}")
        
        # Save test set
        test_path = self.processed_data_path / "books_test.csv"
        test_df.to_csv(test_path, index=False)
        print(f"  ✅ Saved: {test_path}")
        
        # Print summary statistics
        print("\n📊 Processing Summary:")
        print(f"  - Total books processed: {len(self.df)}")
        print(f"  - Books with tags: {(self.df['all_tags_clean'].str.len() > 0).sum()}")
        print(f"  - Average rating: {self.df['average_rating'].mean():.2f}")
        print(f"  - Features created: combined_text, *_clean columns")
        
        return self


def main():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("EMOTION-BASED BOOK RECOMMENDER - DATA PREPROCESSING")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Run pipeline
    preprocessor.load_data()
    preprocessor.preprocess_features()
    preprocessor.filter_quality_books(min_tags=1, min_rating_count=50)
    
    # Create splits
    train_df, test_df = preprocessor.create_train_test_split(test_size=0.2)
    
    # Save results
    preprocessor.save_processed_data(train_df, test_df)
    
    print("\n" + "=" * 60)
    print("✅ DATA PREPROCESSING COMPLETE!")
    print("=" * 60)
    print("\nNext step: Run feature engineering (src/features/build_features.py)")


if __name__ == "__main__":
    main()
