"""ComfyUI node for choosing practical GPT Image 2.5 canvas sizes.

The sizing rules are an experimental estimate based on documented canvas
constraints. They do not predict the dimensions selected by an API or model.
"""
from __future__ import annotations
import math
from typing import Iterable

MIN_PIXELS = 655_360
MAX_PIXELS = 8_294_400
MAX_EDGE = 3_840
GRID = 16
MODES = ("NanoSeed 4K (verified)", "Max area (experimental estimate)", "Closest aspect", "Exact dimensions")


def validate_size(width: int, height: int, *, allow_min: bool = True) -> tuple[int, int]:
    """Validate and return a GPT Image-compatible width/height pair."""
    if (isinstance(width, bool) or isinstance(height, bool)
            or int(width) != width or int(height) != height):
        raise ValueError("Width and height must be integers")
    width, height = int(width), int(height)
    if width <= 0 or height <= 0 or width > MAX_EDGE or height > MAX_EDGE:
        raise ValueError(f"Dimensions must be between 1 and {MAX_EDGE}")
    if width % GRID or height % GRID:
        raise ValueError("Dimensions must be multiples of 16")
    pixels = width * height
    if allow_min and pixels < MIN_PIXELS:
        raise ValueError(f"Image area must be at least {MIN_PIXELS} pixels")
    if pixels > MAX_PIXELS:
        raise ValueError(f"Image area must not exceed {MAX_PIXELS} pixels")
    if not 1 / 3 <= width / height <= 3:
        raise ValueError("Aspect ratio must be between 1:3 and 3:1")
    return width, height

def _grid_candidates(pixel_budget: int) -> Iterable[tuple[int, int]]:
    budget = min(int(pixel_budget), MAX_PIXELS)
    for width in range(GRID, MAX_EDGE + 1, GRID):
        for height in range(GRID, min(MAX_EDGE, budget // width) + 1, GRID):
            if height * width >= MIN_PIXELS and 1 / 3 <= width / height <= 3:
                yield width, height

def choose_size(
    width: int,
    height: int,
    mode: str = MODES[0],
    pixel_budget: int = MAX_PIXELS,
    max_crop_percent: float = 2.0,
) -> tuple[int, int]:
    """Choose a valid size using the selected sizing policy."""
    if width <= 0 or height <= 0 or mode not in MODES:
        raise ValueError("Invalid source dimensions or sizing mode")
    if mode == MODES[0]:
        return _choose_nanoseed_4k_size(width, height)
    if int(pixel_budget) != pixel_budget or not MIN_PIXELS <= int(pixel_budget) <= MAX_PIXELS:
        raise ValueError(f"pixel_budget must be between {MIN_PIXELS} and {MAX_PIXELS}")
    if not 0 <= float(max_crop_percent) <= 25:
        raise ValueError("max_crop_percent must be between 0 and 25")
    if mode == "Exact dimensions":
        return validate_size(width, height)
    source_ratio = max(1 / 3, min(3.0, float(width) / float(height)))
    candidates = list(_grid_candidates(int(pixel_budget)))
    if not candidates:
        raise ValueError("No valid size fits the requested pixel budget")
    def ratio_error(pair):
        return abs(math.log((pair[0] / pair[1]) / source_ratio))
    if mode.startswith("Max area"):
        allowed = float(max_crop_percent) / 100.0
        suitable = [p for p in candidates if 1 - min(p[0] / p[1] / source_ratio, source_ratio / (p[0] / p[1])) <= allowed + 1e-12]
        pool = suitable or candidates
        if suitable:
            return max(pool, key=lambda p: (p[0] * p[1], -ratio_error(p), -p[0], -p[1]))
        return min(pool, key=lambda p: (ratio_error(p), -p[0] * p[1], p[0], p[1]))
    ideal_width = math.sqrt(int(pixel_budget) * source_ratio)
    ideal_height = math.sqrt(int(pixel_budget) / source_ratio)
    scale = min(1.0, MAX_EDGE / max(ideal_width, ideal_height))
    ideal_area = ideal_width * ideal_height * scale * scale
    near = [p for p in candidates if p[0] * p[1] >= ideal_area * 0.95]
    pool = near or candidates
    return min(pool, key=lambda p: (ratio_error(p), -p[0] * p[1], p[0], p[1]))


def _choose_nanoseed_4k_size(width: int, height: int) -> tuple[int, int]:
    """Match NanoSeed's local GPT 4K auto-size calculation without importing it."""
    max_edge = MAX_EDGE
    source_long = max(width, height)
    source_short = min(width, height)
    source_ratio = min(source_long / source_short, 3.0)
    ideal_long = min(max_edge, math.sqrt(MAX_PIXELS * source_ratio))
    ideal_short = ideal_long / source_ratio
    long_candidates = sorted({
        max(GRID, math.floor(ideal_long / GRID) * GRID),
        max(GRID, math.ceil(ideal_long / GRID) * GRID),
    })
    short_candidates = sorted({
        max(GRID, math.floor(ideal_short / GRID) * GRID),
        max(GRID, math.ceil(ideal_short / GRID) * GRID),
    })
    valid_sizes = [
        (long_edge, short_edge)
        for long_edge in long_candidates
        for short_edge in short_candidates
        if long_edge <= max_edge
        and short_edge <= max_edge
        and long_edge * short_edge <= MAX_PIXELS
        and long_edge / short_edge <= 3.0
    ]
    if not valid_sizes:
        raise ValueError("Unable to calculate a valid GPT image size")
    output_long, output_short = max(
        valid_sizes,
        key=lambda pair: (
            pair[0] * pair[1],
            -abs((pair[0] / pair[1]) - source_ratio),
            pair[0],
            pair[1],
        ),
    )
    if width >= height:
        return output_long, output_short
    return output_short, output_long

def crop_geometry(source_width: int, source_height: int, target_width: int, target_height: int,
                  crop_x: float = 0.5, crop_y: float = 0.5) -> tuple[int, int, int, int]:
    """Return (left, top, width, height) for the largest centered ratio crop."""
    if (source_width <= 0 or source_height <= 0 or target_width <= 0
            or target_height <= 0):
        raise ValueError("Source and target dimensions must be positive")
    if not 0 <= crop_x <= 1 or not 0 <= crop_y <= 1:
        raise ValueError("crop_x and crop_y must be between 0 and 1")
    target_ratio = target_width / target_height
    if source_width / source_height > target_ratio:
        crop_h, crop_w = source_height, max(1, min(source_width, int(round(source_height * target_ratio))))
    else:
        crop_w, crop_h = source_width, max(1, min(source_height, int(round(source_width / target_ratio))))
    return (int(round((source_width - crop_w) * crop_x)), int(round((source_height - crop_h) * crop_y)), crop_w, crop_h)

class GptImageSizeCalculator:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "sizing_mode": (list(MODES), {"default": MODES[0],
                                             "tooltip": "NanoSeed 4K (verified) matches the local NanoSeed auto-size algorithm; its fixed 4K budget ignores pixel_budget and max_crop_percent."}),
            "pixel_budget": ("INT", {"default": MAX_PIXELS, "min": MIN_PIXELS,
                                        "max": MAX_PIXELS, "step": GRID}),
            "max_crop_percent": ("FLOAT", {"default": 2.0, "min": 0.0,
                                              "max": 25.0, "step": 0.1}),
            "target_width": ("INT", {"default": 2592, "min": GRID,
                                       "max": MAX_EDGE, "step": GRID}),
            "target_height": ("INT", {"default": 3200, "min": GRID,
                                        "max": MAX_EDGE, "step": GRID}),
            "fit": (["Crop", "Pad"], {"default": "Crop"}),
            "crop_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,
                                    "step": 0.01}),
            "crop_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,
                                    "step": 0.01}),
        }}
    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "width", "height", "size", "aspect_ratio", "info")
    FUNCTION = "calculate"
    CATEGORY = "image"

    def calculate(
        self, image, sizing_mode, pixel_budget, max_crop_percent,
        target_width, target_height, fit, crop_x, crop_y,
    ):
        import torch
        import torch.nn.functional as F
        if (image.ndim != 4 or not image.is_floating_point()
                or image.shape[-1] not in (1, 3, 4)
                or min(image.shape[:3]) <= 0):
            raise ValueError("IMAGE must be a rank-4 floating tensor in NHWC format with 1, 3, or 4 channels")
        if not 0 <= crop_x <= 1 or not 0 <= crop_y <= 1:
            raise ValueError("crop_x and crop_y must be between 0 and 1")
        src_h, src_w = int(image.shape[1]), int(image.shape[2])
        out_w, out_h = (validate_size(target_width, target_height) if sizing_mode == "Exact dimensions" else choose_size(src_w, src_h, sizing_mode, pixel_budget, max_crop_percent))
        nchw = image.permute(0, 3, 1, 2)
        interpolation_dtype = nchw.dtype
        if interpolation_dtype in (torch.float16, torch.bfloat16):
            nchw = nchw.float()

        def resize(tensor, size):
            resized = F.interpolate(tensor, size=size, mode="bicubic",
                                    align_corners=False, antialias=True)
            return resized.clamp(0, 1)

        if fit == "Crop":
            left, top, crop_w, crop_h = crop_geometry(src_w, src_h, out_w, out_h, crop_x, crop_y)
            result = resize(nchw[:, :, top:top + crop_h, left:left + crop_w], (out_h, out_w))
            crop_loss = 1 - crop_w * crop_h / (src_w * src_h)
            detail = f"crop rect left={left}, top={top}, width={crop_w}, height={crop_h}"
        elif fit == "Pad":
            scale = min(out_w / src_w, out_h / src_h)
            rw, rh = max(1, round(src_w * scale)), max(1, round(src_h * scale))
            resized = resize(nchw, (rh, rw))
            result = torch.zeros((image.shape[0], image.shape[-1], out_h, out_w), dtype=image.dtype, device=image.device)
            off_x = round((out_w - rw) * crop_x)
            off_y = round((out_h - rh) * crop_y)
            result[:, :, off_y:off_y + rh, off_x:off_x + rw] = resized.to(result.dtype)
            crop_loss, detail = 0.0, f"pad offsets left={off_x}, top={off_y}, width={rw}, height={rh}"
        else:
            raise ValueError("fit must be Crop or Pad")
        result = result.to(interpolation_dtype).permute(0, 2, 3, 1).contiguous()
        ratio = math.gcd(out_w, out_h)
        warning = "; experimental estimate for larger than 3686400 pixels" if out_w * out_h > 3_686_400 else ""
        if sizing_mode == MODES[0]:
            sizing_info = "local NanoSeed 4K algorithm (verified against 12 provided cases)"
        elif sizing_mode == MODES[3]:
            sizing_info = "exact dimensions"
        else:
            sizing_info = "heuristic; cannot predict automatic model sizing"
        info = f"input {src_w}x{src_h} ({src_w/src_h:.4f}), output {out_w}x{out_h} ({out_w/out_h:.4f}); {detail}; estimated crop loss {crop_loss * 100:.2f}%{warning}; {sizing_info}"
        return result, out_w, out_h, f"{out_w}x{out_h}", f"{out_w//ratio}:{out_h//ratio}", info

NODE_CLASS_MAPPINGS = {"GptImageSizeCalculator": GptImageSizeCalculator}
NODE_DISPLAY_NAME_MAPPINGS = {"GptImageSizeCalculator": "Gpt Image Size Calculator"}
