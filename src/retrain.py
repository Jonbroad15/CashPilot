"""
Retrain the SpendingClassifier on master.csv using the sentence-transformer
embedding model. Run this after switching from OpenAI embeddings.

Usage:
    python src/retrain.py
    python src/retrain.py --master master.csv --cache embedding_cache.db
"""
import argparse
import datetime
import os
import sys
import pandas as pd

# Allow running as `python src/retrain.py` from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.model import SpendingClassifier


def retrain(master_csv='master.csv', cache_db='embedding_cache.db'):
    print(f"Loading training data from {master_csv}...")
    df = pd.read_csv(master_csv)
    df = df.dropna(subset=['category', 'description'])
    df['debit'] = pd.to_numeric(df['debit'], errors='coerce').fillna(0)
    df['credit'] = pd.to_numeric(df['credit'], errors='coerce').fillna(0)
    print(f"  {len(df)} labeled transactions found.")

    clf = SpendingClassifier(
        cache_db=cache_db,
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function='MultiClass'
    )
    print("Training classifier...")
    clf.train(df)

    os.makedirs('models', exist_ok=True)
    model_path = os.path.join(
        'models',
        f"spending_classifier_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    )
    clf.save_model(model_path)
    print(f"Model saved to {model_path}")
    return model_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Retrain the SpendingClassifier.")
    parser.add_argument('--master', default='master.csv', help='Path to master CSV')
    parser.add_argument('--cache', default='embedding_cache.db', help='Path to embedding cache DB')
    args = parser.parse_args()
    retrain(master_csv=args.master, cache_db=args.cache)
