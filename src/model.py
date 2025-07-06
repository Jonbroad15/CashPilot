from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from openai import OpenAI
import math

import io
import sqlite3
import json
import os
import hashlib

class DataCleaner:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)

    def clean(self, df, account_name):
        """
        Cleans and reformats a DataFrame into the common format:
        date	description	debit	credit	category	Account
        """
        formatted_df = self.query_llm(df, account_name)
        return formatted_df

    def query_llm(self, df, account_name):
        """
        Sends the DataFrame to OpenAI and returns a cleaned DataFrame
        in the standardized format.
        """
        # Convert dataframe to CSV string for LLM context
        csv_data = df.to_csv(index=False)

        prompt = f"""
You are a data wrangling assistant. Given a CSV of raw financial transaction data, convert it into a clean, standardized CSV with the following columns:

date, description, debit, credit, category, account

Input CSV:
{csv_data}

Output the cleaned CSV only, no explanation.
ensure that:
- The 'date' column is in YYYY-MM-DD format.
- The account name is set to '{account_name}'.
- The 'category' column is left blank for now, as it will be filled later by a classifier.
- The 'debit' and 'credit' columns are numeric, with 'debit' for expenses and 'credit' for income.
- Output the CSV with all text fields properly quoted using double quotes if needed, especially when fields contain commas.
        """
        if 'amex' in account_name.lower():
            prompt += "\nNote: This is an Amex account, so the amount column contains debits. If the amount is negative, it should be treated as a credit."

        response = self.client.chat.completions.create(model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful data transformation assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1)

        csv_output = response.choices[0].message.content
        if csv_output.startswith("```"):
            csv_output = csv_output.strip("```").strip()
            if csv_output.lower().startswith("csv"):
                csv_output = csv_output[3:].strip()

        # Convert the LLM's response back to a DataFrame
        try:
            cleaned_df = pd.read_csv(io.StringIO(csv_output), quotechar='"')
        except pd.errors.ParserError as e:
            print("Error parsing CSV output from LLM:", e)
            print("LLM output was:")
            print(csv_output)
            cleaned_df = pd.read_csv(io.StringIO(csv_output), quotechar='"', error_bad_lines=False)
            breakpoint()
            raise ValueError("LLM did not return a valid CSV format.")
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
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

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


def get_openai_embedding(texts, api_key, model="text-embedding-3-small", cache=None, batch_size=100):
    client = OpenAI(api_key=api_key)

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

    # 🧠 Batch requests
    for i in range(0, len(to_query), batch_size):
        batch = to_query[i:i+batch_size]
        print(f"→ Querying batch {i // batch_size + 1} / {math.ceil(len(to_query)/batch_size)}...")

        response = client.embeddings.create(
            model=model,
            input=batch
        )
        batch_embeddings = [r.embedding for r in response.data]

        for j, emb in enumerate(batch_embeddings):
            global_index = i + j
            new_embeddings[global_index] = emb
            if cache:
                cache.set(to_query[global_index], emb)

    # Assemble final result in order
    full_embeddings = [None] * len(texts)
    for idx, emb in cached:
        full_embeddings[idx] = emb
    for i, idx in enumerate(index_map):
        full_embeddings[idx] = new_embeddings[i]

    return full_embeddings

class SpendingClassifier(CatBoostClassifier):
    def __init__(self, api_key, cache_db="embedding_cache.db", **kwargs):
        self.api_key = api_key
        self.cache = EmbeddingCache(cache_db)
        super().__init__(**kwargs)

    def embed_descriptions(self, descriptions):
        return np.array(get_openai_embedding(
            descriptions.tolist(),
            api_key=self.api_key,
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
        # Flatten predictions if needed
        if hasattr(predictions, "shape") and len(predictions.shape) > 1:
            predictions = predictions[:, 0]

        df = df.copy()
        df["category"] = predictions
        return df


if __name__ == "__main__":


    # Example
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    if os.path.exists('data/amex_aeroplan_cleaned.csv'):
        print("Cleaned data already exists. Skipping cleaning step.")
        cleaned_df = pd.read_csv('data/amex_aeroplan_cleaned.csv')
    else:
        cleaner = DataCleaner(api_key)
        df = pd.read_csv('data/amex_aeroplan.jun25.csv')
        cleaned_df = cleaner.clean(df, account_name="amex_aeroplan")
        cleaned_df.to_csv('data/amex_aeroplan_cleaned.csv', index=False)
        print("Cleaned data saved to 'data/amex_aeroplan_cleaned.csv'.")


    # Train the classifier
    df_master = pd.read_csv('master.csv')
    classifier = SpendingClassifier(api_key=api_key, iterations=1000, learning_rate=0.1, depth=6, loss_function='MultiClass')
    classifier.train(df_master)
    print("Classifier trained on master data.")
    # Predict categories for cleaned data
    predictions_df = classifier.predict(cleaned_df)
    predictions_df.to_csv('data/amex_aeroplan_predictions.csv', index=False)
    print("Predictions saved to 'data/amex_aeroplan_predictions.csv'.")

    print(predictions_df.head(10))


