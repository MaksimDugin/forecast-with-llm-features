"""Checks of window boundaries, as-of joins, pooling, matrix and actual gradients."""
import unittest

import numpy as np
import pandas as pd
import torch

from embed_fintexts import pool_categories
from finmultitime import Forecast, MODELS, Windows, join_available_features, split_windows
from news_features import build_daily
from run_matrix import build_commands


class FakeEncoder:
    def encode(self, texts, **kwargs):
        return np.array([np.full(384, {"a": 1, "b": 3}[t]) for t in texts])


class WorkflowTests(unittest.TestCase):
    def test_pool_missing_and_equal_category_weights(self):
        df = pd.DataFrame({"first": ["a", None, "a"], "second": ["b", "", None]})
        out = pool_categories(df, list(df), FakeEncoder())
        np.testing.assert_equal(out[0], np.full(384, 2))
        self.assertTrue(np.isnan(out[1]).all())
        np.testing.assert_equal(out[2], np.ones(384))

    def test_targets_never_cross_split_boundary(self):
        dates = pd.Series(pd.date_range("2020-01-01", periods=400))
        split = split_windows(dates, 20, 24, "2020-06-01", "2020-10-01")
        self.assertTrue(all(dates[i+23] < pd.Timestamp("2020-06-01") for i in split['train']))
        self.assertTrue(all(dates[i] >= pd.Timestamp("2020-06-01") and dates[i+23] < pd.Timestamp("2020-10-01") for i in split['val']))
        self.assertTrue(all(dates[i] >= pd.Timestamp("2020-10-01") for i in split['test']))
        ds = Windows(np.arange(400)[:, None], np.arange(400)[:, None], [200], 20, 24)
        x, y = ds[0]
        self.assertEqual(x[-1].item(), 199)
        self.assertEqual(y[0].item(), 200)
        self.assertEqual(y[-1].item(), 223)

    def test_no_features_from_after_close_and_no_backfill(self):
        prices = pd.DataFrame({'date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])})
        features = pd.DataFrame({'available_at': ['2020-01-02T08:00:00Z'], 'score': [9.]})
        joined = join_available_features(prices, features, ['score'])
        self.assertEqual(joined.score.tolist(), [0, 0, 9])
        with self.assertRaises(ValueError):
            join_available_features(prices, features.drop(columns='available_at'), ['score'])

    def test_news_timezone_and_next_midnight(self):
        rows = [{'datetime': '2020-01-02 16:01:00', 'summary': '中文新闻'}]
        articles, daily = build_daily(rows, '000001.SZ')
        self.assertEqual(daily.news_count.tolist(), [1, 0])
        self.assertEqual(str(daily.available_at.iloc[0]), '2020-01-02 16:00:00+00:00')
        scores = pd.DataFrame({'article_id': articles.article_id, 'sentiment': [0.8]})
        _, daily = build_daily(rows, '000001.SZ', scores)
        self.assertEqual(daily.sentiment_mean.iloc[0], 0.8)

    def test_models_have_gradients_and_correct_shapes(self):
        torch.set_num_threads(2)
        for name in MODELS:
            with self.subTest(model=name):
                model = Forecast(name, 6, 96, 24, hidden=8)
                y = model(torch.randn(2, 96, 6))
                self.assertEqual(tuple(y.shape), (2, 24, 4))
                y.square().mean().backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))

    def test_aggregation_weights_tickers_equally(self):
        import json
        import tempfile
        from pathlib import Path
        from run_matrix import aggregate
        with tempfile.TemporaryDirectory() as tmp:
            cells = []
            for ticker, seed, value in [('A', 7, 1.), ('A', 17, 3.), ('B', 7, 10.)]:
                path = Path(tmp) / f'{ticker}-{seed}'
                path.mkdir()
                metrics = {'scaled_mse': value, 'scaled_mae': value, 'persistence_scaled_mse': value, 'smoke_only': True}
                (path / 'SUCCESS.json').write_text(json.dumps({'metrics': metrics}))
                cells.append({'id': path.name, 'ticker': ticker, 'seed': seed, 'model': 'DLinear', 'horizon': 3, 'output': str(path)})
            result = aggregate(cells)
            self.assertEqual(result['summaries'][0]['metrics']['scaled_mse']['mean_across_tickers_after_seed_average'], 6.)
            self.assertTrue(result['smoke_only'])
            cells.append({'id': 'missing', 'ticker': 'C', 'model': 'DLinear', 'horizon': 3, 'output': str(Path(tmp) / 'missing')})
            self.assertFalse(aggregate(cells)['complete'])

    def test_matrix_is_explicit_and_never_executes_by_construction(self):
        cfg = {'paper': 'fintexts', 'tickers': ['AAPL', 'MSFT'], 'models': ['DLinear', 'PatchTST'],
               'seeds': [7, 17, 27], 'epochs': 10, 'author': '/author', 'prepared': '/data', 'prefixes': ['macro']}
        cells = build_commands(cfg, '/outputs')
        self.assertEqual(len(cells), 12)
        self.assertTrue(all('--execute' not in c['command'] for c in cells))
        cfg['tickers'] = ['../escape']
        with self.assertRaises(ValueError):
            build_commands(cfg, '/outputs')


if __name__ == '__main__':
    unittest.main()
