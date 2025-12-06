# Recommender with precomputed neighbors

1. Install dependencies:
   pip install -r requirements.txt

2. Download MovieLens 100k and preprocess:
   python src/download_movielens.py
   python src/preprocess.py

3. Start Streamlit:
   streamlit run src/app.py

The preprocess step computes top-K user neighbors and item neighbors and saves them to artifacts/.
The Streamlit app loads these artifacts for instant recommendations.
