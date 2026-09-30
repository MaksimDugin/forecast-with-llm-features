import math
import unittest

import torch

from author_compat import portable_method


def original(self, values, corr):
    index = torch.arange(values.shape[-1]).cuda()
    return values.index_select(-1, index)


class CompatTests(unittest.TestCase):
    def test_cuda_allocation_follows_values_device(self):
        fn = portable_method(original)
        x = torch.randn(2, 3, 4, 16)
        torch.testing.assert_close(fn(None, x, x), x)

    def test_unrecognized_source_is_rejected(self):
        with self.assertRaises(ValueError):
            portable_method(math.log)
