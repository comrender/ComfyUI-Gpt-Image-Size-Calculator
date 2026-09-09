import importlib
import math
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
calculator = importlib.import_module("gpt_image_size_calculator")


class SizePolicyTests(unittest.TestCase):
    def test_observed_portrait_chooses_documented_canvas(self):
        self.assertEqual(calculator.choose_size(3678, 4598), (2576, 3216))

    def test_observed_portrait_heuristic_canvas(self):
        self.assertEqual(calculator.choose_size(3678, 4598, calculator.MODES[1]), (2592, 3200))

    def test_nanoseed_4k_verified_regression_table(self):
        cases = (
            ((3678, 4598), (2576, 3216)),
            ((3678, 4598), (2576, 3216)),
            ((4598, 3678), (3216, 2576)),
            ((3600, 4500), (2576, 3216)),
            ((2592, 3200), (2592, 3200)),
            ((2576, 3216), (2576, 3216)),
            ((1800, 2250), (2576, 3216)),
            ((4000, 4000), (2880, 2880)),
            ((3000, 4000), (2496, 3312)),
            ((3000, 4500), (2352, 3520)),
            ((2160, 3840), (2160, 3840)),
            ((4500, 1500), (3840, 1280)),
        )
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(calculator.choose_size(*source), expected)

    def test_all_modes_return_valid_sizes_over_budgets_and_aspects(self):
        for budget in (calculator.MIN_PIXELS, 1_000_000, 3_686_400, calculator.MAX_PIXELS):
            for width, height in ((320, 960), (960, 320), (1000, 1000), (3678, 4598), (4598, 3678)):
                for mode in calculator.MODES[1:3]:
                    result = calculator.choose_size(width, height, mode, budget, 2.0)
                    self.assertEqual(calculator.validate_size(*result), result)
                    self.assertLessEqual(math.prod(result), budget)

    def test_closest_aspect_prioritizes_ratio_within_near_ideal_area(self):
        result = calculator.choose_size(1000, 2000, calculator.MODES[2], 4_000_000)
        self.assertLess(abs(result[0] / result[1] - 0.5), 0.01)
        self.assertGreaterEqual(math.prod(result), 3_800_000)

    def test_zero_crop_tolerance_falls_back_to_closest_ratio(self):
        result = calculator.choose_size(3678, 4598, calculator.MODES[1], calculator.MAX_PIXELS, 0)
        candidates = list(calculator._grid_candidates(calculator.MAX_PIXELS))
        expected = min(candidates, key=lambda p: (abs(math.log((p[0] / p[1]) / (3678 / 4598))), -math.prod(p), p[0], p[1]))
        self.assertEqual(result, expected)

    def test_exact_dimensions_validation(self):
        self.assertEqual(calculator.choose_size(2592, 3200, calculator.MODES[3]), (2592, 3200))
        for pair in ((17, 3200), (16, 16), (3840, 3840), (4000, 1000), (1024, 4000)):
            with self.subTest(pair=pair):
                if pair == (16, 16):
                    with self.assertRaisesRegex(ValueError, "at least"):
                        calculator.choose_size(*pair, calculator.MODES[3])
                else:
                    with self.assertRaises(ValueError):
                        calculator.choose_size(*pair, calculator.MODES[3])

    def test_crop_geometry_anchors(self):
        self.assertEqual(calculator.crop_geometry(2000, 1000, 1024, 1024, 0, 0), (0, 0, 1000, 1000))
        self.assertEqual(calculator.crop_geometry(2000, 1000, 1024, 1024, 1, 1), (1000, 0, 1000, 1000))
        self.assertEqual(calculator.crop_geometry(1000, 2000, 1024, 1024, 0.5, 0.5), (0, 500, 1000, 1000))


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.node = calculator.GptImageSizeCalculator()

    def calculate(self, image, **kwargs):
        args = dict(sizing_mode=calculator.MODES[0], pixel_budget=calculator.MAX_PIXELS,
                    max_crop_percent=2.0, target_width=1024, target_height=1024,
                    fit="Crop", crop_x=0.5, crop_y=0.5)
        args.update(kwargs)
        return self.node.calculate(image, **args)

    def test_rgba_and_float64_preserve_channels_and_device(self):
        image = torch.full((2, 32, 48, 4), 0.25, dtype=torch.float64)
        image[..., 3] = 0.75
        output, *_ = self.calculate(image, sizing_mode="Exact dimensions")
        self.assertEqual(output.shape, (2, 1024, 1024, 4))
        self.assertEqual(output.dtype, image.dtype)
        self.assertEqual(output.device, image.device)
        self.assertTrue(torch.allclose(output[..., 3], torch.full_like(output[..., 3], 0.75)))
        self.assertTrue(torch.allclose(output[..., :3], torch.full_like(output[..., :3], 0.25)))

    def test_full_portrait_input_and_metadata(self):
        image = torch.linspace(0, 1, 3678 * 4598).reshape(1, 4598, 3678, 1)
        output, width, height, size, aspect, info = self.calculate(
            image, target_width=1024, target_height=1024)
        self.assertEqual(tuple(output.shape), (1, height, width, 1))
        self.assertEqual((width, height, size, aspect), (2576, 3216, "2576x3216", "161:201"))
        self.assertIn("input 3678x4598", info)

    def test_channels_batches_clamp_and_dtype(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                image = torch.tensor([[[[-1.0, 0.5, 2.0], [0.2, 0.3, 0.4]],
                                      [[0.7, 0.8, 0.9], [1.1, -0.2, 0.1]]]], dtype=dtype).repeat(2, 1, 1, 1)
                output, width, height, *_ = self.calculate(image, target_width=1024, target_height=1024)
                self.assertEqual(tuple(output.shape), (2, height, width, 3))
                self.assertEqual(output.dtype, dtype)
                self.assertGreaterEqual(float(output.min()), 0.0)
                self.assertLessEqual(float(output.max()), 1.0)

    def test_pad_uses_x_and_y_anchors(self):
        image = torch.ones((1, 1200, 800, 1), dtype=torch.float32)
        left = self.calculate(image, fit="Pad", crop_x=0.0, crop_y=0.0)[0]
        right = self.calculate(image, fit="Pad", crop_x=1.0, crop_y=0.0)[0]
        self.assertEqual(float(left[:, :, 0, :].max()), 1.0)
        self.assertEqual(float(left[:, :, -1, :].max()), 0.0)
        self.assertEqual(float(right[:, :, 0, :].max()), 0.0)
        self.assertEqual(float(right[:, :, -1, :].max()), 1.0)
        image = torch.ones((1, 800, 1200, 1), dtype=torch.float32)
        top = self.calculate(image, fit="Pad", crop_x=0.0, crop_y=0.0)[0]
        bottom = self.calculate(image, fit="Pad", crop_x=0.0, crop_y=1.0)[0]
        self.assertEqual(float(top[:, 0, :, :].max()), 1.0)
        self.assertEqual(float(top[:, -1, :, :].max()), 0.0)
        self.assertEqual(float(bottom[:, 0, :, :].max()), 0.0)
        self.assertEqual(float(bottom[:, -1, :, :].max()), 1.0)

    def test_malformed_inputs_raise(self):
        cases = [torch.ones((8, 8, 3)), torch.ones((1, 8, 8, 2)), torch.ones((1, 8, 8, 3), dtype=torch.int32)]
        for image in cases:
            with self.subTest(shape=tuple(image.shape), dtype=image.dtype), self.assertRaises(ValueError):
                self.calculate(image)
        image = torch.ones((1, 8, 8, 1))
        for kwargs in ({"fit": "Stretch"}, {"crop_x": -0.1}, {"crop_y": 1.1}, {"pixel_budget": calculator.MIN_PIXELS - 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.calculate(image, sizing_mode=calculator.MODES[1], **kwargs)


class ComfyMappingTests(unittest.TestCase):
    def test_mapping_imports(self):
        package_name = "comfyui_gpt_image_size_calculator_testpkg"
        spec = importlib.util.spec_from_file_location(package_name, ROOT / "__init__.py",
                                                      submodule_search_locations=[str(ROOT)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
        self.assertEqual(package.NODE_CLASS_MAPPINGS["GptImageSizeCalculator"].__name__, "GptImageSizeCalculator")
        self.assertEqual(package.NODE_DISPLAY_NAME_MAPPINGS["GptImageSizeCalculator"], "Gpt Image Size Calculator")


if __name__ == "__main__":
    unittest.main()
