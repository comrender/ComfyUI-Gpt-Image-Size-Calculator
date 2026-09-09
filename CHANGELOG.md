# Changelog

## 1.1.0

- Added automatic Prepare and Restore companion nodes with an original-image reference.
- Preserve the full frame with automatic fit/padding and recorded geometry.
- Restore generated edits to the original dimensions without manual sizing controls.
- Optional original-resolution edit masks preserve untouched pixels exactly.
- Existing calculator inputs and outputs remain compatible.

## 1.0.0

- Standalone Gpt Image Size Calculator node.
- Default NanoSeed 4K sizing verified against 12 supplied Sunburst output sizes.
- Crop and pad with adjustable positioning and batch support.
- Exact dimensions, closest-aspect sizing, and an optional experimental area-first policy.
- Dimension, aspect ratio, API size string, and operation details as outputs.
