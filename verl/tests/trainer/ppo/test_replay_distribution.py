import torch
import torch.nn.functional as F

try:
    from verl.trainer.ppo.replay_distribution import (
        ANCHOR_FULL_ACCEPT,
        ReplayDistributionTracker,
        replay_anchor_bucket,
        replay_block_counts,
        reverse_kl_from_logits,
    )
except ModuleNotFoundError as exc:
    if exc.name != "ray":
        raise
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).resolve().parents[3] / "verl" / "trainer" / "ppo" / "replay_distribution.py"
    spec = importlib.util.spec_from_file_location("replay_distribution", module_path)
    replay_distribution = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(replay_distribution)
    ANCHOR_FULL_ACCEPT = replay_distribution.ANCHOR_FULL_ACCEPT
    ReplayDistributionTracker = replay_distribution.ReplayDistributionTracker
    replay_anchor_bucket = replay_distribution.replay_anchor_bucket
    replay_block_counts = replay_distribution.replay_block_counts
    reverse_kl_from_logits = replay_distribution.reverse_kl_from_logits


def test_replay_block_counts_full_accept():
    assert replay_block_counts(accepted_length=16, drafted_length=16) == {
        "accepted": 16,
        "first_rejected": 0,
        "post_rejection_suffix": 0,
    }
    assert replay_anchor_bucket(accepted_length=16, drafted_length=16) == ANCHOR_FULL_ACCEPT


def test_replay_block_counts_reject_at_first_position():
    assert replay_block_counts(accepted_length=0, drafted_length=16) == {
        "accepted": 0,
        "first_rejected": 1,
        "post_rejection_suffix": 15,
    }
    assert replay_anchor_bucket(accepted_length=0, drafted_length=16) == "1"


def test_replay_block_counts_reject_after_three_tokens():
    assert replay_block_counts(accepted_length=3, drafted_length=16) == {
        "accepted": 3,
        "first_rejected": 1,
        "post_rejection_suffix": 12,
    }
    assert replay_anchor_bucket(accepted_length=3, drafted_length=16) == "4"


def test_replay_block_counts_truncated_block():
    assert replay_block_counts(accepted_length=2, drafted_length=5) == {
        "accepted": 2,
        "first_rejected": 1,
        "post_rejection_suffix": 2,
    }
    assert replay_anchor_bucket(accepted_length=2, drafted_length=5) == "3"


def test_reverse_kl_from_logits_matches_manual_log_softmax():
    q_logits = torch.tensor([[1.0, -1.0, 0.5], [0.2, 0.3, -0.7]], dtype=torch.float32)
    p_logits = torch.tensor([[0.1, 0.4, -0.2], [-0.4, 0.9, 0.0]], dtype=torch.float32)

    q_log_probs = F.log_softmax(q_logits, dim=-1)
    p_log_probs = F.log_softmax(p_logits, dim=-1)
    expected = (q_log_probs.exp() * (q_log_probs - p_log_probs)).sum(dim=-1)

    actual = reverse_kl_from_logits(q_logits, p_logits)
    torch.testing.assert_close(actual, expected)


def test_tracker_flattens_token_aligned_replay_block_metadata():
    class DummyGenOutput:
        non_tensor_batch = {
            "dflash_replay_block_accepted_lengths": [[[16], [], [0], [3]]],
            "dflash_replay_block_drafted_lengths": [[[16], [], [16], [16]]],
            "global_steps": [1],
        }

        def __len__(self):
            return 1

    tracker = ReplayDistributionTracker(total_training_steps=2)
    tracker.update_rollout(DummyGenOutput(), fallback_global_step=1)

    assert tracker.composition_counts == {
        "accepted": 19,
        "first_rejected": 2,
        "post_rejection_suffix": 27,
    }
    assert tracker.anchor_counts["first_half"]["1"] == 1
    assert tracker.anchor_counts["first_half"]["4"] == 1
    assert tracker.anchor_counts["first_half"][ANCHOR_FULL_ACCEPT] == 1
