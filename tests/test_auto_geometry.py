import importlib
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfyui_gpt_image_size_calculator_auto_testpkg"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = package
    spec.loader.exec_module(package)
    return importlib.import_module(PACKAGE_NAME + ".auto_geometry")


geometry = _load_module()


class AutoGeometryTests(unittest.TestCase):
    def setUp(self):
        self.prepare_node = geometry.GptImageAutoPrepare()
        self.restore_node = geometry.GptImageAutoRestore()

    def _prepare(self, raw, mask=None):
        with mock.patch.object(geometry, "choose_size", return_value=(1024, 1024)):
            return self.prepare_node.prepare(raw, mask)

    def _restore(self, edited, raw, transform=None, mask=None):
        return self.restore_node.restore(edited, raw, transform, mask)

    def test_constant_roundtrip_preserves_original_geometry(self):
        raw = torch.full((1, 13, 9, 3), 0.37, dtype=torch.float32)
        prepared, prepared_mask, width, height, size, transform, info = self._prepare(raw)
        restored, restored_edit, raw_mask, restore_info = self._restore(prepared, raw, transform)
        self.assertEqual((width, height, size), (1024, 1024, "1024x1024"))
        self.assertEqual(tuple(prepared.shape), (1, 1024, 1024, 3))
        self.assertEqual(tuple(prepared_mask.shape), (1, 1024, 1024))
        self.assertEqual(tuple(restored.shape), tuple(raw.shape))
        self.assertTrue(torch.allclose(restored, raw, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(restored_edit, raw, atol=1e-4, rtol=1e-4))
        self.assertEqual(tuple(raw_mask.shape), (1, 13, 9))
        self.assertEqual(transform["version"], 1)
        self.assertIsInstance(info, str)
        self.assertIsInstance(restore_info, str)

    def test_prepare_full_image_pad_records_zero_mask_and_padding(self):
        raw = torch.ones((1, 10, 20, 1), dtype=torch.float32)
        prepared, prepared_mask, *_rest = self._prepare(raw)
        self.assertEqual(tuple(prepared.shape), (1, 1024, 1024, 1))
        self.assertEqual(tuple(prepared_mask.shape), (1, 1024, 1024))
        self.assertEqual(float(prepared_mask.min()), 0.0)
        self.assertGreater(float(prepared_mask.max()), 0.0)
        self.assertEqual(float(prepared[:, 0].max()), 0.0)

    def test_restore_gradient_has_original_size_and_spatial_landmarks(self):
        raw = torch.zeros((1, 9, 13, 1), dtype=torch.float32)
        raw[:, 2, 3] = 1.0
        raw[:, 6, 10] = 0.75
        prepared, _mask, _w, _h, _size, transform, _info = self._prepare(raw)
        restored, *_ = self._restore(prepared, raw, transform)
        self.assertEqual(tuple(restored.shape), tuple(raw.shape))
        self.assertGreater(float(restored[:, 2:4, 3:5].max()), 0.2)
        self.assertGreater(float(restored[:, 5:8, 9:12].max()), 0.1)

    def test_edit_mask_preserves_pixels_outside_mask(self):
        raw = torch.linspace(0, 1, 9 * 13).reshape(1, 9, 13, 1)
        edit_mask = torch.zeros((1, 9, 13), dtype=torch.float32)
        edit_mask[:, 3:6, 5:8] = 1.0
        prepared, _pmask, _w, _h, _size, transform, _info = self._prepare(raw, edit_mask)
        edited = prepared.clone()
        edited += 0.25
        restored, _restored_edit, _raw_mask, _info = self._restore(
            edited, raw, transform, edit_mask
        )
        outside = edit_mask == 0
        self.assertTrue(torch.equal(restored[outside], raw[outside]))
        self.assertGreater(float((restored[edit_mask == 1] - raw[edit_mask == 1]).abs().max()), 0)

    def test_zero_edit_mask_restores_raw_exactly(self):
        raw = torch.rand((1, 7, 11, 1), dtype=torch.float32)
        zero_mask = torch.zeros((1, 7, 11), dtype=torch.float32)
        prepared, _pmask, _w, _h, _size, transform, _info = self._prepare(raw, zero_mask)
        edited = prepared + 1
        restored, restored_edit, *_ = self._restore(edited, raw, transform, zero_mask)
        self.assertTrue(torch.equal(restored, raw))
        self.assertFalse(torch.equal(restored_edit, raw))

    def test_batch_and_singleton_broadcasting(self):
        raw = torch.rand((1, 8, 12, 1), dtype=torch.float32)
        prepared, pmask, _w, _h, _size, transform, _info = self._prepare(raw)
        edited = prepared.repeat(3, 1, 1, 1)
        restored, restored_edit, raw_mask, *_ = self._restore(edited, raw, transform)
        self.assertEqual(restored.shape[0], 3)
        self.assertEqual(restored_edit.shape[0], 3)
        self.assertEqual(raw_mask.shape[0], 3)
        self.assertEqual(pmask.shape[0], 1)

    def test_dtype_and_device_are_preserved(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                raw = torch.rand((1, 6, 10, 1), dtype=dtype)
                prepared, _pmask, _w, _h, _size, transform, _info = self._prepare(raw)
                restored, *_ = self._restore(prepared, raw, transform)
                self.assertEqual(restored.dtype, dtype)
                self.assertEqual(restored.device, raw.device)

    def test_malformed_metadata_and_dimensions_are_rejected(self):
        raw = torch.rand((1, 6, 10, 1))
        prepared, _pmask, _w, _h, _size, transform, _info = self._prepare(raw)
        bad = dict(transform)
        bad["version"] = 99
        with self.assertRaises((ValueError, KeyError, TypeError)):
            self._restore(prepared, raw, bad)
        with self.assertRaises((ValueError, RuntimeError)):
            self._restore(prepared[:, :-1], raw, transform)
        with self.assertRaises((ValueError, RuntimeError)):
            self._restore(prepared, torch.rand((2, 6, 10, 1)), transform)

    def test_restore_rejects_transform_raw_batch_mismatch(self):
        raw = torch.rand((2, 6, 10, 1))
        prepared, _pmask, _w, _h, _size, transform, _info = self._prepare(raw)
        bad = dict(transform)
        bad["raw_batch"] = 3
        with self.assertRaises(ValueError):
            self._restore(prepared, raw, bad)

    def test_restore_accepts_mixed_raw_and_edited_dtypes(self):
        raw = torch.rand((1, 7, 11, 1), dtype=torch.float32)
        prepared, _pmask, _w, _h, _size, transform, _info = self._prepare(raw)
        restored, restored_edit, *_ = self._restore(prepared.half(), raw, transform)
        self.assertEqual(restored.dtype, torch.float32)
        self.assertEqual(restored_edit.dtype, torch.float32)
        self.assertEqual(tuple(restored.shape), tuple(raw.shape))

    def test_default_portrait_size_and_transform_for_real_example(self):
        # Expand keeps the fixture compact while exercising the documented dimensions.
        raw = torch.ones((1, 1, 1, 1), dtype=torch.float32).expand(1, 4598, 3678, 1)
        prepared, _pmask, width, height, size, transform, _info = self.prepare_node.prepare(raw)
        self.assertEqual((width, height, size), (2576, 3216, "2576x3216"))
        self.assertEqual(tuple(prepared.shape), (1, 3216, 2576, 1))
        self.assertEqual((transform["raw_width"], transform["raw_height"]), (3678, 4598))
        self.assertGreater(transform["pad_left"], 0)

    def test_real_canvas_choice_is_stable_under_size_policy(self):
        raw = torch.ones((1, 1, 1, 1), dtype=torch.float32).expand(1, 4299, 8827, 1)
        prepared, _pmask, width, height, _size, transform, _info = self.prepare_node.prepare(raw)
        self.assertEqual((width, height), (transform["canvas_width"], transform["canvas_height"]))
        self.assertEqual((width, height), geometry.choose_size(width, height))
        self.assertEqual(tuple(prepared.shape), (1, height, width, 1))

    def test_invalid_channels_masks_and_nan_are_rejected(self):
        with self.assertRaises(ValueError):
            self._prepare(torch.rand((1, 6, 10, 2)))
        raw = torch.rand((1, 6, 10, 1))
        bad_mask = torch.ones((1, 5, 10))
        with self.assertRaises(ValueError):
            self._prepare(raw, bad_mask)
        nan_raw = raw.clone()
        nan_raw[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self._prepare(nan_raw)
        prepared, _pmask, _w, _h, _size, transform, _info = self._prepare(raw)
        with self.assertRaises(ValueError):
            self._restore(prepared, raw, transform, torch.ones((1, 5, 10)))


if __name__ == "__main__":
    unittest.main()
