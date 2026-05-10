"""
Unit and integration tests for SpendingClassifier.
embed_descriptions uses the real sentence-transformer model.
CatBoost is trained on a small synthetic dataset to keep tests fast.
"""
import numpy as np
import pandas as pd
import pytest

from src.model import SpendingClassifier

# Minimal synthetic training set — 5 categories × 6 examples each = 30 rows
_CATEGORIES = ['coffee', 'housing', 'groceries', 'entertainment', 'income']
_DESCRIPTIONS = {
    'coffee':        ['starbucks', 'tim hortons', 'second cup', 'espresso bar', 'coffee shop', 'latte'],
    'housing':       ['rent payment', 'mortgage', 'property tax', 'home insurance', 'condo fee', 'landlord'],
    'groceries':     ['loblaws', 'metro grocery', 'sobeys', 'whole foods', 'food basics', 'no frills'],
    'entertainment': ['netflix', 'spotify', 'cineplex', 'steam games', 'concert ticket', 'disney plus'],
    'income':        ['payroll deposit', 'direct deposit', 'salary transfer', 'employer pay', 'wage deposit', 'paycheck'],
}
_DEBITS = {
    'coffee': 5.0, 'housing': 1500.0, 'groceries': 80.0, 'entertainment': 15.0, 'income': 0.0,
}
_CREDITS = {
    'coffee': 0.0, 'housing': 0.0, 'groceries': 0.0, 'entertainment': 0.0, 'income': 3000.0,
}

TRAINING_DF = pd.DataFrame([
    {'description': desc, 'debit': _DEBITS[cat], 'credit': _CREDITS[cat], 'category': cat}
    for cat, descs in _DESCRIPTIONS.items()
    for desc in descs
])

CLASSIFIER_KWARGS = dict(iterations=50, learning_rate=0.1, depth=3, loss_function='MultiClass')


@pytest.fixture(scope='module')
def trained_classifier(tmp_path_factory):
    """Train once per test module to avoid repeated embedding computation."""
    cache_db = str(tmp_path_factory.mktemp('cache') / 'test.db')
    clf = SpendingClassifier(cache_db=cache_db, **CLASSIFIER_KWARGS)
    clf.train(TRAINING_DF)
    return clf


class TestSpendingClassifierInit:
    def test_no_api_key_required(self, tmp_path):
        clf = SpendingClassifier(cache_db=str(tmp_path / 'c.db'), **CLASSIFIER_KWARGS)
        assert clf is not None

    def test_cache_is_initialized(self, tmp_path):
        clf = SpendingClassifier(cache_db=str(tmp_path / 'c.db'), **CLASSIFIER_KWARGS)
        assert clf.cache is not None


class TestEmbedDescriptions:
    def test_output_shape(self, tmp_path):
        clf = SpendingClassifier(cache_db=str(tmp_path / 'c.db'), **CLASSIFIER_KWARGS)
        descriptions = pd.Series(['coffee', 'rent', 'groceries'])
        embeddings = clf.embed_descriptions(descriptions)
        assert embeddings.shape == (3, 384)

    def test_returns_numpy_array(self, tmp_path):
        clf = SpendingClassifier(cache_db=str(tmp_path / 'c.db'), **CLASSIFIER_KWARGS)
        result = clf.embed_descriptions(pd.Series(['test']))
        assert isinstance(result, np.ndarray)

    def test_embeddings_are_cached(self, tmp_path):
        cache_db = str(tmp_path / 'c.db')
        clf = SpendingClassifier(cache_db=cache_db, **CLASSIFIER_KWARGS)
        descriptions = pd.Series(['unique test phrase 12345'])
        clf.embed_descriptions(descriptions)
        # Cache should now have an entry for this text
        assert clf.cache.get('unique test phrase 12345') is not None


class TestTrain:
    def test_train_completes_without_error(self, tmp_path):
        clf = SpendingClassifier(cache_db=str(tmp_path / 'c.db'), **CLASSIFIER_KWARGS)
        clf.train(TRAINING_DF)  # should not raise

    def test_model_is_fitted_after_train(self, trained_classifier):
        assert trained_classifier.is_fitted()


class TestPredict:
    def test_predict_returns_dataframe(self, trained_classifier):
        test_df = pd.DataFrame({
            'description': ['starbucks coffee'],
            'debit': [4.50],
            'credit': [0.0],
        })
        result = trained_classifier.predict(test_df)
        assert isinstance(result, pd.DataFrame)

    def test_predict_adds_category_column(self, trained_classifier):
        test_df = pd.DataFrame({
            'description': ['netflix subscription'],
            'debit': [15.99],
            'credit': [0.0],
        })
        result = trained_classifier.predict(test_df)
        assert 'category' in result.columns

    def test_predict_category_is_known_value(self, trained_classifier):
        test_df = pd.DataFrame({
            'description': ['payroll deposit'],
            'debit': [0.0],
            'credit': [3000.0],
        })
        result = trained_classifier.predict(test_df)
        assert result['category'].iloc[0] in _CATEGORIES

    def test_predict_does_not_mutate_input(self, trained_classifier):
        test_df = pd.DataFrame({
            'description': ['loblaws groceries'],
            'debit': [75.00],
            'credit': [0.0],
        })
        original_columns = list(test_df.columns)
        trained_classifier.predict(test_df)
        assert list(test_df.columns) == original_columns

    def test_predict_multiple_rows(self, trained_classifier):
        test_df = pd.DataFrame({
            'description': ['starbucks', 'rent payment', 'loblaws'],
            'debit': [5.0, 1500.0, 80.0],
            'credit': [0.0, 0.0, 0.0],
        })
        result = trained_classifier.predict(test_df)
        assert len(result) == 3
