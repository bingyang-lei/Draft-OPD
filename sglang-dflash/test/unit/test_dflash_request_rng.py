import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.dflash_utils import (
    _dflash_hash_to_uniform,
    make_dflash_verify_uniform_samples,
)


class TestDFlashHashScaling(unittest.TestCase):
    def test_uint32_max_stays_below_one(self):
        uniform = _dflash_hash_to_uniform(
            torch.tensor(
                [0, torch.iinfo(torch.uint32).max],
                dtype=torch.uint32,
            )
        )

        self.assertEqual(uniform[0].item(), 0.0)
        self.assertGreaterEqual(uniform.min().item(), 0.0)
        self.assertLess(uniform.max().item(), 1.0)


@unittest.skipUnless(
    torch.cuda.is_available(), "MurmurHash implementation requires CUDA"
)
class TestDFlashRequestRNG(unittest.TestCase):
    def test_draws_follow_request_when_batch_is_reordered(self):
        device = torch.device("cuda")
        block_size = 4
        sampling_info = SimpleNamespace(
            sampling_seed=torch.tensor([11, 29], dtype=torch.int64, device=device)
        )
        positions = torch.tensor(
            [7, 8, 9, 10, 31, 32, 33, 34],
            dtype=torch.int64,
            device=device,
        )

        accept, final = make_dflash_verify_uniform_samples(
            sampling_info=sampling_info,
            positions=positions,
            draft_token_num=block_size,
        )
        reordered_accept, reordered_final = make_dflash_verify_uniform_samples(
            sampling_info=SimpleNamespace(
                sampling_seed=sampling_info.sampling_seed.flip(0)
            ),
            positions=positions.reshape(2, block_size).flip(0).reshape(-1),
            draft_token_num=block_size,
        )

        torch.testing.assert_close(reordered_accept, accept.flip(0), rtol=0, atol=0)
        torch.testing.assert_close(reordered_final, final.flip(0), rtol=0, atol=0)

    def test_draws_are_stable_when_request_is_run_alone(self):
        device = torch.device("cuda")
        block_size = 3
        seeds = torch.tensor([101, 202], dtype=torch.int64, device=device)
        positions = torch.tensor(
            [4, 5, 6, 40, 41, 42],
            dtype=torch.int64,
            device=device,
        )

        batched_accept, batched_final = make_dflash_verify_uniform_samples(
            sampling_info=SimpleNamespace(sampling_seed=seeds),
            positions=positions,
            draft_token_num=block_size,
        )
        alone_accept, alone_final = make_dflash_verify_uniform_samples(
            sampling_info=SimpleNamespace(sampling_seed=seeds[1:]),
            positions=positions[block_size:],
            draft_token_num=block_size,
        )

        torch.testing.assert_close(alone_accept[0], batched_accept[1], rtol=0, atol=0)
        torch.testing.assert_close(alone_final[0], batched_final[1], rtol=0, atol=0)

    def test_global_rng_is_not_consumed(self):
        device = torch.device("cuda")
        sampling_info = SimpleNamespace(
            sampling_seed=torch.tensor([77], dtype=torch.int64, device=device)
        )
        positions = torch.tensor([12, 13], dtype=torch.int64, device=device)

        torch.cuda.manual_seed(1234)
        expected = torch.rand((4,), device=device)
        torch.cuda.manual_seed(1234)
        make_dflash_verify_uniform_samples(
            sampling_info=sampling_info,
            positions=positions,
            draft_token_num=2,
        )
        actual = torch.rand((4,), device=device)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_nondeterministic_mode_delegates_to_existing_rng(self):
        accept, final = make_dflash_verify_uniform_samples(
            sampling_info=SimpleNamespace(sampling_seed=None),
            positions=torch.tensor([0, 1]),
            draft_token_num=2,
        )

        self.assertIsNone(accept)
        self.assertIsNone(final)


if __name__ == "__main__":
    unittest.main()
