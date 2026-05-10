import subprocess
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sentence_transformers import SentenceTransformer
import math
import io
import sqlite3
import json
import os
import hashlib

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
_sentence_model = None


def _get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer(EMBEDDING_MODEL)
    return _sentence_model


class DataCleaner:
    def clean(self, df, account_name):
        """
        Cleans and reformats a DataFrame into the common format:
        date, description, debit, credit, category, account
        """
        return self.query_llm(df, account_name)

    def query_llm(self, df, account_name):
        """
        Sends the DataFrame to Claude via the CLI and returns a cleaned DataFrame
        in the standardized format.
        """
        csv_data = df.to_csv(index=False)

        prompt = f"""You are a data wrangling assistant. Given a CSV of raw financial transaction data, convert it into a clean, standardized CSV with the following columns:

date, description, debit, credit, category, account

Input CSV:
{csv_data}

Output the cleaned CSV only, no explanation.
ensure that:
- The 'date' column is in YYYY-MM-DD format.
- The account name is set to '{account_name}'.
- The 'category' column is left blank for now, as it will be filled later by a classifier.
- The 'debit' and 'credit' columns are numeric, with 'debit' for expenses and 'credit' for income.
- Output the CSV with all text fields properly quoted using double quotes if needed, especially when fields contain commas."""

        if 'amex' in account_name.lower():
            prompt += "\nNote: This is an Amex account, so the amount column contains debits. If the amount is negative, it should be treated as a credit."

        result = subprocess.run(
            ['claude', '-p', prompt],
            capture_output=True,
            text=True,
            check=True
        )
        csv_output = result.stdout.strip()

        if csv_output.startswith("```"):
            csv_output = csv_output.strip("```").strip()
            if csv_output.lower().startswith("csv"):
                csv_output = csv_output[3:].strip()

        expected_columns = {'date', 'description', 'debit', 'credit', 'category', 'account'}
        try:
            cleaned_df = pd.read_csv(io.StringIO(csv_output), quotechar='"')
        except pd.errors.ParserError as e:
            print("Error parsing CSV output from LLM:", e)
            print("LLM output was:")
            print(csv_output)
            raise ValueError("Claude did not return a valid CSV format.")
        missing = expected_columns - set(cleaned_df.columns)
        if missing:
            raise ValueError(f"Claude response is missing expected columns: {missing}\nOutput was:\n{csv_output}")
        return cleaned_df


class EmbeddingCache:
    def __init__(self, db_path="embedding_cache.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                hash TEXT PRIMARY KEY,
                text TEXT,
                embedding TEXT
            )
        ''')
        self.conn.commit()

    def _hash(self, text):
        # Include model name in hash so cached OpenAI embeddings are not reused
        return hashlib.sha256(f"{EMBEDDING_MODEL}:{text}".encode('utf-8')).hexdigest()

    def get(self, text):
        h = self._hash(text)
        cur = self.conn.execute('SELECT embedding FROM embeddings WHERE hash = ?', (h,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def set(self, text, embedding):
        h = self._hash(text)
        self.conn.execute(
            'INSERT OR REPLACE INTO embeddings (hash, text, embedding) VALUES (?, ?, ?)',
            (h, text, json.dumps(embedding))
        )
        self.conn.commit()


def get_embeddings(texts, cache=None, batch_size=100):
    """
    Generate sentence-transformer embeddings for a list of texts.
    Results are cached in an EmbeddingCache if provided.
    """
    if isinstance(texts, str):
        texts = [texts]

    texts = [str(t).strip() if pd.notnull(t) else "" for t in texts]

    cached = []
    to_query = []
    index_map = []

    for idx, text in enumerate(texts):
        emb = cache.get(text) if cache else None
        if emb is not None:
            cached.append((idx, emb))
        else:
            to_query.append(text)
            index_map.append(idx)

    new_embeddings = [None] * len(to_query)

    if to_query:
        model = _get_sentence_model()
    for i in range(0, len(to_query), batch_size):
        batch = to_query[i:i + batch_size]
        print(f"→ Embedding batch {i // batch_size + 1} / {math.ceil(len(to_query) / batch_size)}...")
        batch_embeddings = model.encode(batch).tolist()
        for j, emb in enumerate(batch_embeddings):
            global_index = i + j
            new_embeddings[global_index] = emb
            if cache:
                cache.set(to_query[global_index], emb)

    full_embeddings = [None] * len(texts)
    for idx, emb in cached:
        full_embeddings[idx] = emb
    for i, idx in enumerate(index_map):
        full_embeddings[idx] = new_embeddings[i]

    return full_embeddings


class SpendingClassifier(CatBoostClassifier):
    def __init__(self, cache_db="embedding_cache.db", **kwargs):
        self.cache = EmbeddingCache(cache_db)
        super().__init__(**kwargs)

    def embed_descriptions(self, descriptions):
        return np.array(get_embeddings(
            descriptions.tolist(),
            cache=self.cache
        ))

    def train(self, df_master):
        desc_embeddings = self.embed_descriptions(df_master["description"])
        other_features = df_master[["debit", "credit"]].values
        X = np.hstack([desc_embeddings, other_features])
        y = df_master["category"]
        self.fit(X, y)

    def predict(self, df):
        desc_embeddings = self.embed_descriptions(df["description"])
        other_features = df[["debit", "credit"]].values
        X = np.hstack([desc_embeddings, other_features])
        predictions = super().predict(X)
        if hasattr(predictions, "shape") and len(predictions.shape) > 1:
            predictions = predictions[:, 0]

        df = df.copy()
        df["category"] = predictions
        return df
