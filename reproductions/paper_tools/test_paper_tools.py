import datetime as dt
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prepare import acquire, relative_path, verify
from run_fintexts import validate_rows


class InputTests(unittest.TestCase):
    def test_hash_not_just_size(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data"
            p.write_bytes(b"bad")
            item = {"bytes": 3, "sha256": hashlib.sha256(b"yes").hexdigest()}
            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify(p, item)

    def test_corrupt_existing_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data"
            p.write_bytes(b"old")
            item = {"path": "data", "bytes": 3, "sha256": hashlib.sha256(b"new").hexdigest()}
            with patch("prepare.urllib.request.urlopen") as network:
                with self.assertRaises(ValueError):
                    acquire({}, item, d, True)
                network.assert_not_called()
            self.assertEqual(p.read_bytes(), b"old")

    def test_truncated_response_never_published(self):
        with tempfile.TemporaryDirectory() as d:
            item = {"path": "data", "bytes": 10, "sha256": "0" * 64}
            with patch("prepare.urllib.request.urlopen", return_value=io.BytesIO(b"short")):
                with self.assertRaisesRegex(ValueError, "truncated"):
                    acquire({"dataset_id": "a/b", "revision": "abc"}, item, d, True)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_verified_download_and_offline_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            item = {"path": "nested/data", "bytes": 3, "sha256": hashlib.sha256(b"yes").hexdigest()}
            with patch("prepare.urllib.request.urlopen", return_value=io.BytesIO(b"yes")):
                p = acquire({"dataset_id": "a/b", "revision": "abc"}, item, d, True)
            self.assertEqual(p.read_bytes(), b"yes")
            self.assertEqual(acquire({}, item, d), p)

    def test_path_escape(self):
        for path in ("../file", "/tmp/file", "C:/file", "..\\file"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                relative_path(path)


class ContractTests(unittest.TestCase):
    def rows(self):
        rows = []
        for year, count in ((2021, 100), (2022, 80), (2023, 20)):
            for offset in range(count):
                date = dt.date(year, 1, 1) + dt.timedelta(days=offset)
                row = dict(date=str(date), ticker="TEST", open=1., high=2., low=.5, close=1.5)
                row.update({f"macro_emb{i}": 0.1 for i in range(384)})
                rows.append(row)
        return rows

    def test_nonempty_author_windows(self):
        rows = self.rows()
        result = validate_rows(rows, rows[0], ["macro"])
        self.assertEqual(result["windows"], {"train": 34, "val": 78, "test": 18})

    def test_raw_text_cannot_masquerade_as_embeddings(self):
        with self.assertRaisesRegex(ValueError, "384 columns"):
            validate_rows([], ["date", "ticker", "open", "high", "low", "close"], ["macro"])

    def test_duplicate_date_and_mixed_tickers(self):
        for field, value in (("date", "2021-01-01"), ("ticker", "OTHER")):
            rows = self.rows()
            rows[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_rows(rows, rows[0], ["macro"])

    def test_no_validation_or_test_data_rejected(self):
        rows = self.rows()[:100]
        with self.assertRaises(ValueError):
            validate_rows(rows, rows[0], ["macro"])

    def test_missing_text_retains_author_semantics_but_infinity_rejected(self):
        rows = self.rows()
        rows[0]["macro_emb0"] = None
        result = validate_rows(rows, rows[0], ["macro"])
        self.assertEqual(result["missing_embedding_cells"], 1)
        rows[0]["macro_emb0"] = float("inf")
        with self.assertRaisesRegex(ValueError, "Infinite"):
            validate_rows(rows, rows[0], ["macro"])


if __name__ == "__main__":
    unittest.main()
