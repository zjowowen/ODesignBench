from __future__ import annotations

import unittest

import torch

from refold.chai1.chai1_distributed_inference import _masked_token_asym_id


class Chai1TokenAsymIdTests(unittest.TestCase):
    def test_masked_token_asym_id_removes_padding_tokens(self) -> None:
        token_asym_id = torch.tensor([[1, 2, 1, 0, 0]], dtype=torch.int32)
        token_mask_1d = torch.tensor([True, True, True, False, False])

        masked = _masked_token_asym_id(token_asym_id, token_mask_1d)

        self.assertEqual(tuple(masked.shape), (3,))
        self.assertEqual(masked.tolist(), [1, 2, 1])


if __name__ == "__main__":
    unittest.main()
