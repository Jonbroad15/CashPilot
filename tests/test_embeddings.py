"""
Unit tests for EmbeddingCache and get_embeddings.
Uses real sentence-transformers (no mocking) to catch model integration issues,
but operates on a tiny in-memory cache to stay fast.
"""
import numpy as np
import pytest

from src.model import EMBEDDING_MODEL, EmbeddingCache, get_embeddings

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension


class TestEmbeddingCache:
    def test_miss_returns_none(self):
        cache = EmbeddingCache(":memory:")
        assert cache.get("nonexistent text") is None

    def test_set_and_get_roundtrip(self):
        cache = EmbeddingCache(":memory:")
        embedding = [0.1, 0.2, 0.3]
        cache.set("hello world", embedding)
        result = cache.get("hello world")
        assert result == embedding

    def test_different_texts_have_different_hashes(self):
        cache = EmbeddingCache(":memory:")
        cache.set("coffee", [1.0])
        cache.set("rent", [2.0])
        assert cache.get("coffee") == [1.0]
        assert cache.get("rent") == [2.0]

    def test_overwrite_existing_key(self):
        cache = EmbeddingCache(":memory:")
        cache.set("text", [1.0, 2.0])
        cache.set("text", [9.0, 8.0])
        assert cache.get("text") == [9.0, 8.0]

    def test_hash_includes_model_name(self):
        """Ensures old OpenAI-keyed embeddings won't collide with new ones."""
        cache = EmbeddingCache(":memory:")
        # Manually insert with a hash that does NOT include the model name
        import hashlib, json
        old_hash = hashlib.sha256("some text".encode()).hexdigest()
        cache.conn.execute(
            'INSERT INTO embeddings (hash, text, embedding) VALUES (?, ?, ?)',
            (old_hash, "some text", json.dumps([0.5]))
        )
        cache.conn.commit()
        # The new model-prefixed hash should not find the old entry
        assert cache.get("some text") is None


class TestGetEmbeddings:
    def test_returns_list_of_vectors(self):
        result = get_embeddings(["coffee shop", "grocery store"])
        assert len(result) == 2
        assert len(result[0]) == EMBEDDING_DIM

    def test_single_string_input(self):
        result = get_embeddings("single transaction")
        assert len(result) == 1
        assert len(result[0]) == EMBEDDING_DIM

    def test_empty_string_handled(self):
        result = get_embeddings([""])
        assert len(result) == 1
        assert len(result[0]) == EMBEDDING_DIM

    def test_null_values_handled(self):
        import pandas as pd
        result = get_embeddings([pd.NA, None, "valid"])
        assert len(result) == 3

    def test_caching_returns_same_values(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path / "test.db"))
        texts = ["starbucks latte"]
        first = get_embeddings(texts, cache=cache)
        second = get_embeddings(texts, cache=cache)
        np.testing.assert_array_almost_equal(first, second)

    def test_cached_results_skip_model_call(self, tmp_path):
        """Second call should hit cache and not call the model encoder."""
        from unittest.mock import patch
        cache = EmbeddingCache(str(tmp_path / "test.db"))
        texts = ["cached transaction"]

        # First call — populate cache
        get_embeddings(texts, cache=cache)

        # Second call — should use cache, not the model
        with patch('src.model._get_sentence_model') as mock_model:
            get_embeddings(texts, cache=cache)
        mock_model.assert_not_called()

    def test_output_length_matches_input(self):
        texts = ["a", "b", "c", "d", "e"]
        result = get_embeddings(texts)
        assert len(result) == len(texts)
