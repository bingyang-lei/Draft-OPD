import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace

_UTILS_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "sglang"
    / "srt"
    / "managers"
    / "customized_info_utils.py"
)
_SPEC = importlib.util.spec_from_file_location("customized_info_utils", _UTILS_PATH)
_UTILS = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_UTILS)

append_dflash_reject_token_mask = _UTILS.append_dflash_reject_token_mask
append_dflash_rejected_draft_metadata = _UTILS.append_dflash_rejected_draft_metadata
update_customized_info_meta = _UTILS.update_customized_info_meta


class TestDFlashCustomizedInfo(unittest.TestCase):
    def test_dflash_mask_pads_prefill_token_before_verify_commit(self):
        req = SimpleNamespace(output_ids=[101], customized_info=None)

        append_dflash_reject_token_mask(
            req,
            3,
            output_ids_already_updated=False,
        )

        self.assertEqual(
            req.customized_info["dflash_reject_token_mask"],
            [False, False, False, True],
        )

    def test_dflash_mask_handles_already_updated_output_ids(self):
        req = SimpleNamespace(output_ids=[101, 102, 103], customized_info=None)

        append_dflash_reject_token_mask(
            req,
            2,
            output_ids_already_updated=True,
        )

        self.assertEqual(
            req.customized_info["dflash_reject_token_mask"],
            [False, False, True],
        )

    def test_dflash_mask_extends_existing_aligned_mask(self):
        req = SimpleNamespace(
            output_ids=[101, 102],
            customized_info={"dflash_reject_token_mask": [False, True]},
        )

        append_dflash_reject_token_mask(
            req,
            2,
            output_ids_already_updated=False,
        )

        self.assertEqual(
            req.customized_info["dflash_reject_token_mask"],
            [False, True, False, True],
        )

    def test_dflash_rejected_draft_partial_accept_flattens_suffix_metadata(self):
        req = SimpleNamespace(output_ids=[101, 102, 103], customized_info=None)

        append_dflash_rejected_draft_metadata(
            req,
            2,
            anchor_index=0,
            offsets=[2, 3],
            token_ids=[202, 203],
            teacher_logprobs=[-0.2, -0.3],
            output_ids_already_updated=True,
        )

        state = SimpleNamespace(customized_info={})
        meta_info = {}
        update_customized_info_meta(
            meta_info=meta_info,
            state=state,
            recv_customized_info={key: [value] for key, value in req.customized_info.items()},
            recv_index=0,
            use_stream_output=False,
        )

        self.assertEqual(meta_info["dflash_rejected_draft_anchor_indices"], [0, 0])
        self.assertEqual(meta_info["dflash_rejected_draft_offsets"], [2, 3])
        self.assertEqual(meta_info["dflash_rejected_draft_token_ids"], [202, 203])
        self.assertEqual(meta_info["dflash_rejected_draft_teacher_logprobs"], [-0.2, -0.3])

    def test_dflash_rejected_draft_all_accepted_stores_no_suffix(self):
        req = SimpleNamespace(output_ids=[101, 102, 103], customized_info=None)

        append_dflash_rejected_draft_metadata(
            req,
            2,
            anchor_index=0,
            offsets=[],
            token_ids=[],
            teacher_logprobs=[],
            output_ids_already_updated=True,
        )

        self.assertIsNone(req.customized_info)

    def test_dflash_rejected_draft_early_stop_before_rejection_stores_no_suffix(self):
        req = SimpleNamespace(output_ids=[101, 102], customized_info=None)

        append_dflash_rejected_draft_metadata(
            req,
            1,
            anchor_index=0,
            offsets=[],
            token_ids=[],
            teacher_logprobs=[],
            output_ids_already_updated=True,
        )

        self.assertIsNone(req.customized_info)

    def test_non_streaming_customized_info_accumulates_chunks(self):
        state = SimpleNamespace(customized_info={})

        meta_info = {}
        update_customized_info_meta(
            meta_info=meta_info,
            state=state,
            recv_customized_info={"dflash_reject_token_mask": [[False, True]]},
            recv_index=0,
            use_stream_output=False,
        )
        self.assertEqual(meta_info["dflash_reject_token_mask"], [False, True])

        meta_info = {}
        update_customized_info_meta(
            meta_info=meta_info,
            state=state,
            recv_customized_info={"dflash_reject_token_mask": [[False, False, True]]},
            recv_index=0,
            use_stream_output=False,
        )
        self.assertEqual(
            meta_info["dflash_reject_token_mask"],
            [False, True, False, False, True],
        )

        meta_info = {}
        update_customized_info_meta(
            meta_info=meta_info,
            state=state,
            recv_customized_info=None,
            recv_index=0,
            use_stream_output=False,
        )
        self.assertEqual(
            meta_info["dflash_reject_token_mask"],
            [False, True, False, False, True],
        )

    def test_non_streaming_customized_info_can_defer_meta_emit(self):
        state = SimpleNamespace(customized_info={})

        meta_info = {}
        update_customized_info_meta(
            meta_info=meta_info,
            state=state,
            recv_customized_info={"dflash_reject_token_mask": [[False, True]]},
            recv_index=0,
            use_stream_output=False,
            emit_accumulated=False,
        )

        self.assertNotIn("dflash_reject_token_mask", meta_info)
        self.assertEqual(state.customized_info["dflash_reject_token_mask"], [False, True])

        meta_info = {}
        update_customized_info_meta(
            meta_info=meta_info,
            state=state,
            recv_customized_info={"dflash_reject_token_mask": [[False, False, True]]},
            recv_index=0,
            use_stream_output=False,
            emit_accumulated=True,
        )
        self.assertEqual(
            meta_info["dflash_reject_token_mask"],
            [False, True, False, False, True],
        )

    def test_streaming_customized_info_remains_chunk_local(self):
        state = SimpleNamespace(customized_info={})

        meta_info = {}
        update_customized_info_meta(
            meta_info=meta_info,
            state=state,
            recv_customized_info={"dflash_reject_token_mask": [[False, True]]},
            recv_index=0,
            use_stream_output=True,
        )

        self.assertEqual(meta_info["dflash_reject_token_mask"], [False, True])
        self.assertEqual(state.customized_info, {})


if __name__ == "__main__":
    unittest.main()
