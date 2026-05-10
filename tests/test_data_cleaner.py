"""
Unit tests for DataCleaner.
subprocess.run is mocked so no actual Claude CLI call is made.
"""
import io
import textwrap
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.model import DataCleaner

STANDARD_COLUMNS = ['date', 'description', 'debit', 'credit', 'category', 'account']

RAW_DF = pd.DataFrame({
    'Date': ['Jan 1, 2024', 'Jan 2, 2024'],
    'Description': ['STARBUCKS', 'TRANSFER'],
    'Amount': [-5.50, 100.00],
})

CLEAN_CSV_RESPONSE = textwrap.dedent("""\
    date,description,debit,credit,category,account
    2024-01-01,STARBUCKS,5.50,0.0,,test_account
    2024-01-02,TRANSFER,0.0,100.0,,test_account
""")


def make_mock_result(stdout):
    mock = MagicMock()
    mock.stdout = stdout
    mock.returncode = 0
    return mock


class TestDataCleanerInit:
    def test_no_api_key_required(self):
        cleaner = DataCleaner()
        assert cleaner is not None


class TestDataCleanerQueryLlm:
    def test_returns_dataframe_with_standard_columns(self):
        cleaner = DataCleaner()
        with patch('subprocess.run', return_value=make_mock_result(CLEAN_CSV_RESPONSE)):
            result = cleaner.query_llm(RAW_DF, 'test_account')
        assert list(result.columns) == STANDARD_COLUMNS

    def test_returns_correct_row_count(self):
        cleaner = DataCleaner()
        with patch('subprocess.run', return_value=make_mock_result(CLEAN_CSV_RESPONSE)):
            result = cleaner.query_llm(RAW_DF, 'test_account')
        assert len(result) == 2

    def test_strips_markdown_code_fence(self):
        fenced = "```csv\n" + CLEAN_CSV_RESPONSE + "\n```"
        cleaner = DataCleaner()
        with patch('subprocess.run', return_value=make_mock_result(fenced)):
            result = cleaner.query_llm(RAW_DF, 'test_account')
        assert list(result.columns) == STANDARD_COLUMNS

    def test_amex_note_included_in_prompt(self):
        cleaner = DataCleaner()
        captured_prompt = []

        def capture_run(cmd, **kwargs):
            captured_prompt.append(cmd[-1])  # last arg is the prompt string
            return make_mock_result(CLEAN_CSV_RESPONSE)

        with patch('subprocess.run', side_effect=capture_run):
            cleaner.query_llm(RAW_DF, 'amex_cobalt')

        assert 'amex' in captured_prompt[0].lower()
        assert 'debit' in captured_prompt[0].lower()

    def test_non_amex_account_has_no_amex_note(self):
        cleaner = DataCleaner()
        captured_prompt = []

        def capture_run(cmd, **kwargs):
            captured_prompt.append(cmd[-1])
            return make_mock_result(CLEAN_CSV_RESPONSE)

        with patch('subprocess.run', side_effect=capture_run):
            cleaner.query_llm(RAW_DF, 'cibc_dividend')

        assert 'amex account' not in captured_prompt[0].lower()

    def test_raises_on_missing_columns(self):
        cleaner = DataCleaner()
        # Valid CSV but missing required columns
        bad_response = "col_a,col_b\nfoo,bar\n"
        with patch('subprocess.run', return_value=make_mock_result(bad_response)):
            with pytest.raises(ValueError, match="missing expected columns"):
                cleaner.query_llm(RAW_DF, 'test_account')

    def test_clean_delegates_to_query_llm(self):
        cleaner = DataCleaner()
        with patch.object(cleaner, 'query_llm', return_value=pd.DataFrame()) as mock_q:
            cleaner.clean(RAW_DF, 'test_account')
        mock_q.assert_called_once_with(RAW_DF, 'test_account')
