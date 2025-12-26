"""
Model Evaluation Script

Compares TF-IDF+KNN baseline vs SBERT advanced models using:
- Precision@K
- Recall@K  
- NDCG@K (Normalized Discounted Cumulative Gain)
- Diversity metrics
- Coverage analysis

Author: Emotion-Based Book Recommender Project
Date: December 26, 2025
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Evaluate and compare recommendation models."""
    
    def __init__(self, processed_data_path="data/processed"):
        self.processed_data_path = Path(processed_data_path)
        self.models_path = self.processed_data_path / "models"
        self.features_path = self.processed_data_path / "features"
        
        # Data
        self.train_df = None
        self.test_df = None
        
        # TF-IDF Model
        self.tfidf_vectorizer = None
        self.tfidf_knn = None
        self.tfidf_emotion_vectors = None
        self.X_train_tfidf = None
        self.X_test_tfidf = None
        
        # SBERT Model
        self.sbert_model = None
        self.sbert_train_embeddings = None
        self.sbert_emotion_embeddings = None
        self.sbert_books_df = None
        
    def load_data(self):
        """Load train and test datasets."""
        print("[*] Loading datasets...")
        self.train_df = pd.read_csv(self.processed_data_path / "books_train.csv")
        self.test_df = pd.read_csv(self.processed_data_path / "books_test.csv")
        print(f"  [OK] Train: {self.train_df.shape}, Test: {self.test_df.shape}")
        return self
    
    def load_tfidf_model(self):
        """Load TF-IDF baseline model."""
        print("\n[*] Loading TF-IDF+KNN model...")
        
        # Load vectorizer
        with open(self.features_path / 'tfidf_vectorizer.pkl', 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        
        # Load emotion vectors
        with open(self.features_path / 'emotion_vectors.pkl', 'rb') as f:
            self.tfidf_emotion_vectors = pickle.load(f)
        
        # Load features
        self.X_train_tfidf = np.load(self.features_path / 'X_train_tfidf.npy')
        
        # Build KNN model
        self.tfidf_knn = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute')
        self.tfidf_knn.fit(self.X_train_tfidf)
        
        print(f"  [OK] TF-IDF model loaded ({self.X_train_tfidf.shape})")
        return self
    
    def load_sbert_model(self):
        """Load SBERT advanced model (lite version)."""
        print("\n[*] Loading SBERT model...")
        
        # Load SBERT
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load embeddings
        self.sbert_train_embeddings = np.load(self.models_path / 'sbert_lite_embeddings.npy')
        
        # Load emotion embeddings
        with open(self.models_path / 'sbert_lite_emotion_embeddings.pkl', 'rb') as f:
            self.sbert_emotion_embeddings = pickle.load(f)
        
        # Load books dataframe (subset used in lite model)
        self.sbert_books_df = pd.read_csv(self.models_path / 'sbert_lite_books.csv')
        
        print(f"  [OK] SBERT model loaded ({self.sbert_train_embeddings.shape})")
        return self
    
    def recommend_tfidf(self, emotion, n_recommendations=10):
        """Get TF-IDF recommendations for emotion."""
        if emotion.lower() not in self.tfidf_emotion_vectors:
            return []
        
        emotion_vector = self.tfidf_emotion_vectors[emotion.lower()]
        similarities = cosine_similarity(emotion_vector, self.X_train_tfidf)[0]
        
        top_indices = similarities.argsort()[::-1][:n_recommendations]
        return [self.train_df.iloc[idx]['book_id'] for idx in top_indices]
    
    def recommend_sbert(self, emotion, n_recommendations=10):
        """Get SBERT recommendations for emotion."""
        if emotion.lower() not in self.sbert_emotion_embeddings:
            return []
        
        emotion_embedding = self.sbert_emotion_embeddings[emotion.lower()].reshape(1, -1)
        similarities = cosine_similarity(emotion_embedding, self.sbert_train_embeddings)[0]
        
        top_indices = similarities.argsort()[::-1][:n_recommendations]
        return [self.sbert_books_df.iloc[idx]['book_id'] for idx in top_indices]
    
    def precision_at_k(self, recommended_ids, relevant_ids, k=10):
        """Calculate Precision@K."""
        recommended_k = recommended_ids[:k]
        relevant_set = set(relevant_ids)
        hits = len([item for item in recommended_k if item in relevant_set])
        return hits / k if k > 0 else 0
    
    def recall_at_k(self, recommended_ids, relevant_ids, k=10):
        """Calculate Recall@K."""
        recommended_k = recommended_ids[:k]
        relevant_set = set(relevant_ids)
        hits = len([item for item in recommended_k if item in relevant_set])
        return hits / len(relevant_set) if len(relevant_set) > 0 else 0
    
    def ndcg_at_k(self, recommended_ids, relevant_ids, k=10):
        """Calculate NDCG@K (Normalized Discounted Cumulative Gain)."""
        recommended_k = recommended_ids[:k]
        relevant_set = set(relevant_ids)
        
        # DCG: sum of (relevance / log2(position+1))
        dcg = sum([1.0 / np.log2(idx + 2) if item in relevant_set else 0 
                   for idx, item in enumerate(recommended_k)])
        
        # IDCG: perfect ranking
        idcg = sum([1.0 / np.log2(idx + 2) for idx in range(min(len(relevant_set), k))])
        
        return dcg / idcg if idcg > 0 else 0
    
    def diversity_score(self, recommended_ids):
        """Calculate diversity (unique genres/tags coverage)."""
        recommended_books = self.train_df[self.train_df['book_id'].isin(recommended_ids)]
        
        # Count unique tags
        all_tags = set()
        for tags_str in recommended_books['all_tags_clean'].fillna(''):
            tags = tags_str.split()[:5]  # Top 5 tags per book
            all_tags.update(tags)
        
        return len(all_tags)
    
    def evaluate_emotion_recommendations(self, emotions_to_test=['anxious', 'happy', 'sad'], k_values=[5, 10]):
        """Evaluate emotion-based recommendations."""
        print("\n" + "=" * 80)
        print("EVALUATING EMOTION-BASED RECOMMENDATIONS")
        print("=" * 80)
        
        results = []
        
        for emotion in emotions_to_test:
            print(f"\n[EVAL] Evaluating emotion: {emotion.upper()}")
            
            # Get recommendations from both models
            tfidf_recs = self.recommend_tfidf(emotion, n_recommendations=20)
            sbert_recs = self.recommend_sbert(emotion, n_recommendations=20)
            
            # For emotion-based recs, we use high-rated books as "relevant"
            relevant_ids = self.train_df[
                (self.train_df['average_rating'] >= 4.0) & 
                (self.train_df['ratings_count'] >= 100)
            ]['book_id'].tolist()
            
            for k in k_values:
                # TF-IDF metrics
                tfidf_precision = self.precision_at_k(tfidf_recs, relevant_ids, k)
                tfidf_recall = self.recall_at_k(tfidf_recs, relevant_ids, k)
                tfidf_ndcg = self.ndcg_at_k(tfidf_recs, relevant_ids, k)
                tfidf_diversity = self.diversity_score(tfidf_recs[:k])
                
                # SBERT metrics
                sbert_precision = self.precision_at_k(sbert_recs, relevant_ids, k)
                sbert_recall = self.recall_at_k(sbert_recs, relevant_ids, k)
                sbert_ndcg = self.ndcg_at_k(sbert_recs, relevant_ids, k)
                sbert_diversity = self.diversity_score(sbert_recs[:k])
                
                results.append({
                    'emotion': emotion,
                    'k': k,
                    'tfidf_precision': tfidf_precision,
                    'sbert_precision': sbert_precision,
                    'tfidf_recall': tfidf_recall,
                    'sbert_recall': sbert_recall,
                    'tfidf_ndcg': tfidf_ndcg,
                    'sbert_ndcg': sbert_ndcg,
                    'tfidf_diversity': tfidf_diversity,
                    'sbert_diversity': sbert_diversity
                })
                
                print(f"\n  K={k}:")
                print(f"    TF-IDF  → P:{tfidf_precision:.3f} R:{tfidf_recall:.3f} NDCG:{tfidf_ndcg:.3f} Div:{tfidf_diversity}")
                print(f"    SBERT   → P:{sbert_precision:.3f} R:{sbert_recall:.3f} NDCG:{sbert_ndcg:.3f} Div:{sbert_diversity}")
        
        return pd.DataFrame(results)
    
    def evaluate_rating_prediction(self, sample_size=100):
        """Evaluate how well recommendations match user preferences."""
        print("\n" + "=" * 80)
        print("EVALUATING RATING ALIGNMENT")
        print("=" * 80)
        
        emotions = ['anxious', 'happy', 'sad', 'excited', 'stressed']
        
        tfidf_avg_ratings = []
        sbert_avg_ratings = []
        
        for emotion in emotions:
            # Get top 10 recommendations
            tfidf_recs = self.recommend_tfidf(emotion, n_recommendations=10)
            sbert_recs = self.recommend_sbert(emotion, n_recommendations=10)
            
            # Get average ratings
            tfidf_books = self.train_df[self.train_df['book_id'].isin(tfidf_recs)]
            sbert_books = self.sbert_books_df[self.sbert_books_df['book_id'].isin(sbert_recs)]
            
            tfidf_avg = tfidf_books['average_rating'].mean()
            sbert_avg = sbert_books['average_rating'].mean()
            
            tfidf_avg_ratings.append(tfidf_avg)
            sbert_avg_ratings.append(sbert_avg)
            
            print(f"\n  {emotion.upper()}: TF-IDF avg rating: {tfidf_avg:.2f} | SBERT avg rating: {sbert_avg:.2f}")
        
        print(f"\n  Overall TF-IDF: {np.mean(tfidf_avg_ratings):.3f}")
        print(f"  Overall SBERT:  {np.mean(sbert_avg_ratings):.3f}")
        
        return {
            'tfidf_avg': np.mean(tfidf_avg_ratings),
            'sbert_avg': np.mean(sbert_avg_ratings)
        }
    
    def generate_comparison_report(self, results_df, rating_results):
        """Generate final comparison report."""
        print("\n" + "=" * 80)
        print("FINAL MODEL COMPARISON REPORT")
        print("=" * 80)
        
        # Aggregate metrics
        tfidf_metrics = {
            'Precision@10': results_df[results_df['k']==10]['tfidf_precision'].mean(),
            'Recall@10': results_df[results_df['k']==10]['tfidf_recall'].mean(),
            'NDCG@10': results_df[results_df['k']==10]['tfidf_ndcg'].mean(),
            'Diversity@10': results_df[results_df['k']==10]['tfidf_diversity'].mean(),
            'Avg Rating': rating_results['tfidf_avg']
        }
        
        sbert_metrics = {
            'Precision@10': results_df[results_df['k']==10]['sbert_precision'].mean(),
            'Recall@10': results_df[results_df['k']==10]['sbert_recall'].mean(),
            'NDCG@10': results_df[results_df['k']==10]['sbert_ndcg'].mean(),
            'Diversity@10': results_df[results_df['k']==10]['sbert_diversity'].mean(),
            'Avg Rating': rating_results['sbert_avg']
        }
        
        comparison_df = pd.DataFrame({
            'TF-IDF+KNN': tfidf_metrics,
            'SBERT': sbert_metrics
        })
        
        print("\n[RESULTS] Aggregated Metrics:")
        print(comparison_df.round(3))
        
        # Determine winner
        print("\n[WINNER] By Metric:")
        for metric in comparison_df.index:
            tfidf_val = comparison_df.loc[metric, 'TF-IDF+KNN']
            sbert_val = comparison_df.loc[metric, 'SBERT']
            winner = 'SBERT' if sbert_val > tfidf_val else 'TF-IDF+KNN'
            print(f"  {metric:15} → {winner}")
        
        # Save results
        results_df.to_csv(self.models_path / 'evaluation_results.csv', index=False)
        comparison_df.to_csv(self.models_path / 'model_comparison.csv')
        
        print(f"\n[SAVED] Results saved to: {self.models_path}")
        
        return comparison_df


def main():
    """Main evaluation pipeline."""
    print("=" * 80)
    print("MODEL EVALUATION PIPELINE")
    print("=" * 80)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load data and models
    evaluator.load_data()
    evaluator.load_tfidf_model()
    evaluator.load_sbert_model()
    
    # Run evaluations
    results_df = evaluator.evaluate_emotion_recommendations(
        emotions_to_test=['anxious', 'happy', 'sad', 'excited', 'stressed'],
        k_values=[5, 10]
    )
    
    rating_results = evaluator.evaluate_rating_prediction()
    
    # Generate report
    comparison_df = evaluator.generate_comparison_report(results_df, rating_results)
    
    print("\n" + "=" * 80)
    print("[OK] EVALUATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
