# 🚀 Quick Start Guide - Emotion-Based Book Recommender

## Running the Streamlit App

### Option 1: Run Locally

```bash
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Launch Streamlit app
streamlit run src\app\app_streamlit.py
```

The app will open in your browser at: **http://localhost:8501**

### Option 2: Using Python Module

```bash
python -m streamlit run src\app\app_streamlit.py
```

## Using the App

1. **Select Your Emotion** 🎯
   - Choose from 10 emotions: anxious, sad, lonely, stressed, hopeless, angry, grief, overwhelmed, happy, excited

2. **Choose Model** ⚙️
   - **SBERT (Advanced)**: Best quality, semantic understanding (recommended)
   - **TF-IDF (Baseline)**: Faster, keyword-based matching

3. **Adjust Settings** 📊
   - Number of recommendations (3-20)
   - Minimum book rating (3.0-5.0)

4. **Get Recommendations** 🔍
   - Click "Get Recommendations" button
   - Browse results with ratings, match scores, and tags

5. **Download Results** 📥
   - Export recommendations as CSV file

## Features

### 🎨 Interactive UI
- Clean, modern interface
- Emoji-based emotion selection
- Real-time model comparison

### 📚 Book Cards
- Book title and author
- Average rating (with review count)
- Match score (similarity to emotion)
- Related tags

### 📊 Metrics Dashboard
- Total books found
- Average rating
- Average match score

### 🔄 Model Comparison
- Side-by-side performance stats
- NDCG@10 scores
- Quality indicators

## Performance

**SBERT Model:**
- NDCG@10: 0.817
- Precision@10: 0.82
- Avg Rating: 4.16★
- 2x better than baseline

**TF-IDF Model:**
- NDCG@10: 0.383
- Precision@10: 0.48
- Avg Rating: 3.98★
- Faster inference

## Troubleshooting

### Port Already in Use
```bash
streamlit run src\app\app_streamlit.py --server.port 8502
```

### Models Not Loading
- Ensure you've run the training scripts:
  ```bash
  python src\models\train_tfidf_knn.py
  python src\models\train_embedding_lite.py
  ```

### Missing Dependencies
```bash
pip install streamlit sentence-transformers scikit-learn pandas numpy
```

## Customization

### Change Theme
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1E88E5"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Add More Emotions
Edit emotion mappings in `app_streamlit.py`:
```python
emotions = {
    'Your Emotion 😊': 'emotion_key',
    # Add more...
}
```

### Adjust Filters
Modify default values in sidebar:
```python
min_rating = st.sidebar.slider("Minimum rating:", 3.0, 5.0, 3.5)
```

## Production Deployment

### Streamlit Cloud (Free)
1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect repository
4. Deploy with one click

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "src/app/app_streamlit.py"]
```

### Heroku
```bash
# Add Procfile
echo "web: streamlit run src/app/app_streamlit.py --server.port $PORT" > Procfile
git push heroku main
```

## Next Steps

- [ ] Add book cover images (Goodreads API)
- [ ] User feedback collection (thumbs up/down)
- [ ] Save recommendation history
- [ ] Multi-emotion selection
- [ ] Book-to-book similarity search
- [ ] Integration with library APIs

## Support

For issues or questions:
- Check the README.md
- Review model evaluation results
- Verify data files in `data/processed/`

**Happy Reading! 📖✨**
