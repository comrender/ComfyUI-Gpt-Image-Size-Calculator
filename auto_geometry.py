"""Lossless geometry bookkeeping for GPT Image edit workflows."""
from __future__ import annotations

from .gpt_image_size_calculator import choose_size, validate_size


_TRANSFORM_TYPE = "GPT_IMAGE_TRANSFORM"
_VERSION = 1


def _image_shape(image, name: str):
    if (getattr(image, "ndim", None) != 4 or not image.is_floating_point()
            or image.shape[-1] not in (1, 3, 4)
            or min(image.shape[:3]) <= 0):
        raise ValueError(f"{name} must be a rank-4 floating tensor in NHWC format with 1, 3, or 4 channels")
    import torch
    if not torch.isfinite(image).all().item():
        raise ValueError(f"{name} contains NaN or infinite values")
    return int(image.shape[0]), int(image.shape[1]), int(image.shape[2]), int(image.shape[3])


def _mask_shape(mask, name: str):
    if getattr(mask, "ndim", None) == 2:
        mask = mask.unsqueeze(0)
    if getattr(mask, "ndim", None) != 3 or not mask.is_floating_point() or min(mask.shape) <= 0:
        raise ValueError(f"{name} must be a rank-2 or rank-3 floating MASK tensor")
    import torch
    if not torch.isfinite(mask).all().item():
        raise ValueError(f"{name} contains NaN or infinite values")
    if ((mask < 0) | (mask > 1)).any().item():
        raise ValueError(f"{name} values must be between 0 and 1")
    return mask


def _resize(tensor, size, mode: str):
    import torch
    import torch.nn.functional as F
    source_dtype = tensor.dtype
    work = tensor
    if source_dtype in (torch.float16, torch.bfloat16):
        work = tensor.float()
    kwargs = {"size": size, "mode": mode, "antialias": True}
    if mode in ("bicubic", "bilinear"):
        kwargs["align_corners"] = False
    return F.interpolate(work, **kwargs).clamp(0, 1).to(source_dtype)


def _mapping(mapping):
    if not isinstance(mapping, dict) or mapping.get("version") != _VERSION:
        raise ValueError("GPT_IMAGE_TRANSFORM must be a version 1 transform mapping")
    fields = ("raw_width", "raw_height", "canvas_width", "canvas_height",
              "resized_width", "resized_height", "pad_left", "pad_top", "raw_batch")
    values = {}
    for field in fields:
        value = mapping.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0 and field not in ("pad_left", "pad_top"):
            raise ValueError(f"GPT_IMAGE_TRANSFORM field {field} must be a positive integer")
        if field in ("pad_left", "pad_top") and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"GPT_IMAGE_TRANSFORM field {field} must be a non-negative integer")
        values[field] = value
    validate_size(values["canvas_width"], values["canvas_height"])
    if values["resized_width"] > values["canvas_width"] or values["resized_height"] > values["canvas_height"]:
        raise ValueError("GPT_IMAGE_TRANSFORM resized dimensions exceed canvas")
    if values["pad_left"] + values["resized_width"] > values["canvas_width"] or values["pad_top"] + values["resized_height"] > values["canvas_height"]:
        raise ValueError("GPT_IMAGE_TRANSFORM padding is outside canvas")
    expected = _fit_geometry(values["raw_width"], values["raw_height"], values["canvas_width"], values["canvas_height"])
    if (values["resized_width"], values["resized_height"], values["pad_left"], values["pad_top"]) != expected:
        raise ValueError("GPT_IMAGE_TRANSFORM geometry does not describe centered uniform fit")
    return values


def _fit_geometry(raw_w, raw_h, canvas_w, canvas_h):
    scale = min(canvas_w / raw_w, canvas_h / raw_h)
    resized_w, resized_h = max(1, round(raw_w * scale)), max(1, round(raw_h * scale))
    return resized_w, resized_h, round((canvas_w - resized_w) / 2), round((canvas_h - resized_h) / 2)


def _stable_canvas(raw_w, raw_h):
    """Stabilize the automatic size against a downstream auto-sizing pass."""
    canvas = choose_size(raw_w, raw_h)
    seen = set()
    for _ in range(256):
        if canvas in seen:
            raise ValueError("Automatic GPT image sizing did not converge")
        seen.add(canvas)
        next_canvas = choose_size(*canvas)
        if next_canvas == canvas:
            return canvas
        canvas = next_canvas
    raise ValueError("Automatic GPT image sizing did not converge")


class GptImageAutoPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"raw_image": ("IMAGE",)},
                "optional": {"edit_mask": ("MASK", {"tooltip": "Optional raw-image mask; white pixels are editable."})}}

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING", _TRANSFORM_TYPE, "STRING")
    RETURN_NAMES = ("image", "mask", "width", "height", "size", "transform", "info")
    FUNCTION = "prepare"
    CATEGORY = "image"

    def prepare(self, raw_image, edit_mask=None):
        import torch
        b, raw_h, raw_w, channels = _image_shape(raw_image, "raw_image")
        canvas_w, canvas_h = _stable_canvas(raw_w, raw_h)
        resized_w, resized_h, pad_left, pad_top = _fit_geometry(raw_w, raw_h, canvas_w, canvas_h)
        nchw = raw_image.permute(0, 3, 1, 2)
        resized = _resize(nchw, (resized_h, resized_w), "bicubic")
        result = torch.zeros((b, channels, canvas_h, canvas_w), dtype=raw_image.dtype, device=raw_image.device)
        result[:, :, pad_top:pad_top + resized_h, pad_left:pad_left + resized_w] = resized
        result = result.permute(0, 2, 3, 1).contiguous()
        if edit_mask is None:
            prepared_mask = torch.zeros((b, canvas_h, canvas_w), dtype=raw_image.dtype, device=raw_image.device)
            prepared_mask[:, pad_top:pad_top + resized_h, pad_left:pad_left + resized_w] = 1
        else:
            mask = _mask_shape(edit_mask, "edit_mask")
            if mask.shape[1:] != (raw_h, raw_w):
                raise ValueError("edit_mask spatial dimensions must match raw_image")
            if mask.shape[0] not in (1, b):
                raise ValueError("edit_mask batch must be 1 or match raw_image batch")
            mask = mask.to(device=raw_image.device)
            if mask.shape[0] == 1 and b != 1:
                mask = mask.expand(b, -1, -1)
            prepared_mask = torch.zeros((b, canvas_h, canvas_w), dtype=mask.dtype, device=raw_image.device)
            resized_mask = _resize(mask.unsqueeze(1), (resized_h, resized_w), "bilinear").squeeze(1)
            prepared_mask[:, pad_top:pad_top + resized_h, pad_left:pad_left + resized_w] = resized_mask
        transform = {"version": _VERSION, "raw_width": raw_w, "raw_height": raw_h,
                     "canvas_width": canvas_w, "canvas_height": canvas_h,
                     "resized_width": resized_w, "resized_height": resized_h,
                     "pad_left": pad_left, "pad_top": pad_top, "raw_batch": b}
        info = (f"prepared {raw_w}x{raw_h} into {canvas_w}x{canvas_h}; "
                f"content {resized_w}x{resized_h}, centered pad left={pad_left}, top={pad_top}; "
                "automatic size stabilized; geometry only; use the same original image when restoring")
        return result, prepared_mask, canvas_w, canvas_h, f"{canvas_w}x{canvas_h}", transform, info


class GptImageAutoRestore:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"edited_image": ("IMAGE",), "raw_image": ("IMAGE",), "transform": (_TRANSFORM_TYPE,)},
                "optional": {"edit_mask": ("MASK", {"tooltip": "Optional raw-image mask; white pixels are replaced."})}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "restored_edit", "edit_mask", "info")
    FUNCTION = "restore"
    CATEGORY = "image"

    def restore(self, edited_image, raw_image, transform, edit_mask=None):
        import torch
        import torch.nn.functional as F
        raw_b, raw_h, raw_w, raw_c = _image_shape(raw_image, "raw_image")
        edit_b, edit_h, edit_w, edit_c = _image_shape(edited_image, "edited_image")
        m = _mapping(transform)
        if (raw_w, raw_h) != (m["raw_width"], m["raw_height"]):
            raise ValueError("raw_image dimensions do not match GPT_IMAGE_TRANSFORM")
        if raw_b != m["raw_batch"]:
            raise ValueError("raw_image batch does not match GPT_IMAGE_TRANSFORM")
        if (edit_w, edit_h) != (m["canvas_width"], m["canvas_height"]):
            raise ValueError("edited_image dimensions must exactly match the transform canvas")
        if raw_c != edit_c:
            raise ValueError("raw_image and edited_image must have the same channel count")
        batch = max(raw_b, edit_b)
        if raw_b not in (1, batch) or edit_b not in (1, batch):
            raise ValueError("raw_image and edited_image batches must match or be singleton")
        raw = raw_image if raw_b == batch else raw_image.expand(batch, -1, -1, -1)
        edited = edited_image if edit_b == batch else edited_image.expand(batch, -1, -1, -1)
        edited = edited.to(device=raw.device, dtype=raw.dtype)
        content = edited[:, m["pad_top"]:m["pad_top"] + m["resized_height"], m["pad_left"]:m["pad_left"] + m["resized_width"]]
        nchw = content.permute(0, 3, 1, 2)
        source_dtype = raw.dtype
        work = nchw.float() if source_dtype in (torch.float16, torch.bfloat16) else nchw
        restored = F.interpolate(work, size=(raw_h, raw_w), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1).to(source_dtype)
        restored = restored.permute(0, 2, 3, 1).contiguous()
        if edit_mask is None:
            mask = torch.ones((batch, raw_h, raw_w), dtype=raw.dtype, device=raw.device)
            composite = restored
        else:
            mask = _mask_shape(edit_mask, "edit_mask")
            if mask.shape[1:] != (raw_h, raw_w) or mask.shape[0] not in (1, batch):
                raise ValueError("edit_mask must have raw_image spatial dimensions and batch 1 or matching output")
            mask = mask.to(device=raw.device, dtype=raw.dtype)
            if mask.shape[0] == 1 and batch != 1:
                mask = mask.expand(batch, -1, -1)
            alpha = mask.unsqueeze(-1)
            composite = torch.where(alpha == 0, raw, torch.where(alpha == 1, restored, raw * (1 - alpha) + restored * alpha))
        info = f"restored {m['canvas_width']}x{m['canvas_height']} edit to {raw_w}x{raw_h}; source dtype/device preserved"
        return composite, restored, mask, info


AUTO_NODE_CLASS_MAPPINGS = {"GptImageAutoPrepare": GptImageAutoPrepare, "GptImageAutoRestore": GptImageAutoRestore}
AUTO_NODE_DISPLAY_NAME_MAPPINGS = {"GptImageAutoPrepare": "Gpt Image Auto Prepare", "GptImageAutoRestore": "Gpt Image Auto Restore"}
