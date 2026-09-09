# Gpt Image Size Calculator

A standalone ComfyUI node for preparing images for **GPT Image 2.5 Sunburst and Flare**. Calculates a valid output canvas, crops or pads without deliberate stretching, and returns the prepared image plus dimensions. No API calls or keys are required.

## Install

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
git clone https://github.com/comrender/ComfyUI-Gpt-Image-Size-Calculator.git
```

Restart ComfyUI and search for **Gpt Image Size Calculator**. Requires Python 3.10+ and uses PyTorch already supplied by ComfyUI; no additional runtime packages or API keys are needed. You can also download the repository ZIP and extract its folder into `custom_nodes`. This package is independent of Nano Banana and does not import or change its node.

Once published and indexed in ComfyUI Registry, its package ID is `gpt-image-size-calculator`, under publisher `comrender`.

## Size rules

The [OpenAI size documentation](https://developers.openai.com/api/docs/guides/image-generation#size-and-quality-options), checked September 9, 2026, specifies:

- Width and height are multiples of 16.
- Each edge is at most 3840 pixels.
- Width/height is between 1/3 and 3, inclusive.
- Area is between 655,360 and 8,294,400 pixels, inclusive.
- Resolutions above 2560 × 1440 are experimental.

These are custom-size constraints, not a fixed list of aspect-ratio buckets. Ratios such as 1:1, 4:5, 3:4, 2:3, 9:16, 16:9, and 3:1 are possible when the actual dimensions meet every constraint. The size output describes the actual canvas ratio, not a rounded conventional label.

## Verified NanoSeed 4K behavior

All **12 user-supplied Sunburst test outputs** match the sizes calculated by the current local NanoSeed function, using 4K, GPT image size auto, width/height zero, and aspect ratio auto. Request sizes were reconstructed from that source code, not read from historical Fal request logs. The new default is **NanoSeed 4K (verified)**. This verifies the tested workflow; it does not reveal the provider's hidden auto-sizing algorithm or establish Flare behavior empirically.

The calculation preserves orientation, clamps the long/short ratio to 3, scales toward an 8,294,400-pixel budget with a 3840-pixel maximum edge, and tries the floor/ceiling multiples of 16 on each edge. It selects the largest valid area among those four candidates, using ratio error as a tie-breaker. This is a local rounding rule, not a search over every possible valid canvas. NanoSeed submits an explicit size even when its GPT image size control says auto.

The repeated **3678 × 4598** input produced **2576 × 3216** both times. The earlier reported **2592 × 3200** result was not reproduced for that input. The initial area-first heuristic matched only 4 of the 12 tests and is retained only as an explicitly experimental alternative. Exact dimensions can still reproduce the earlier target.

Crop mode fits the selected canvas by trimming edges before resizing; padding retains the full frame. Integer crop boundaries introduce less than a pixel of rounding error relative to the ideal crop. Dimensions alone do not prove whether generated content was cropped, stretched, or recomposed by the model.

See [the complete test analysis](TEST_RESULTS.md).

## Controls

| Control | Behavior |
| --- | --- |
| NanoSeed 4K (verified) | Default. Matches all 12 supplied test sizes using NanoSeed's four-candidate calculation. Fixed 8,294,400-pixel budget and 3840-pixel edge; pixel_budget and max_crop_percent are ignored. |
| Max area (experimental estimate) | Selects the largest valid canvas within the allowed aspect crop loss; default tolerance is 2%. Reproduces the observed example but does not predict provider auto sizing. |
| Closest aspect | Favors the closest ratio among canvases using at least 95% of the ideal feasible area. Allows a small area sacrifice to preserve composition. |
| Exact dimensions | Uses target width/height exactly after validating all size rules. Pixel budget and crop tolerance do not apply. |
| pixel_budget | Maximum area for Max area and Closest aspect modes; defaults to 8,294,400. It is a node policy, not a quality setting. Smaller images may be upscaled. |
| max_crop_percent | Aspect tolerance for the area-first calculation, not a hard limit on final crop. Inputs outside 1:3–3:1 must first fit the supported range. If no candidate meets tolerance, the closest available ratio is selected. |
| Crop | Removes edges to fit the target, then resizes. |
| Pad | Resizes the full frame to fit, then adds black padding. |
| crop_x / crop_y | Position from 0 (left/top) through 0.5 (center) to 1 (right/bottom); also positions content in Pad mode. |

Quality is deliberately separate: `high`, `xhigh`, and `max` are generation settings, not resolution presets.

## Connect to generation

Connect the prepared **IMAGE** to the editing node and use the **width** and **height** outputs as the explicit generation dimensions. For direct OpenAI requests, `size` is the returned `WIDTHxHEIGHT` string. For Fal, pass `image_size: {"width": width, "height": height}`. Preprocessing alone cannot force a downstream node or service using `auto` to keep this canvas.

For a NanoSeed workflow, select **NanoSeed 4K (verified)**, connect the prepared image and both dimension outputs, and keep NanoSeed at **4K**. Its size calculation preserves the selected sizes in the tested cases. To connect width/height to widget-based inputs, use ComfyUI's convert-widget-to-input option on the receiving node.

Other modes may select targets NanoSeed recalculates; exact general passthrough requires support in the calling node. NanoSeed was not modified by this package.

Outputs: prepared image, width, height, size string, reduced aspect-ratio string, and an information string describing crop/padding and sizing limitations. Batches share the same crop and canvas. No subject detection is performed.

## Verification

Run `python -m unittest discover -s tests -v` with a Python environment containing PyTorch. Tests cover supplied output sizes, constraints, crop/pad geometry, and image tensors. The 12 live results were generated by the user with NanoSeed; the new calculator's full ComfyUI UI integration has not been exercised here.

## Publishing releases

`pyproject.toml` contains the Comfy Registry metadata. Registry publication additionally requires a publisher API key saved in GitHub Actions as `REGISTRY_ACCESS_TOKEN`. Creating the file or repository alone does not publish a registry listing.

Update the semantic version and changelog, push the changes, then create a GitHub release for that version. The publishing workflow runs tests on Python 3.10 and 3.12 before uploading to Comfy Registry. It can also be started manually under Actions. Each version must be unique; do not republish an existing version. Registry indexing may take additional time after a successful upload.

## License

[MIT](LICENSE). Maintained by comrender. Not affiliated with OpenAI or Fal.
