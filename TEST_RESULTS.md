# NanoSeed / Fal Sunburst sizing test results

Settings reported by user: 4K; GPT image size auto; width=0; height=0; aspect ratio auto.

All output dimensions below were read from actual JPEG files. Request dimensions were reconstructed from the current local NanoSeed function, not captured from API logs.

| Test | Input | Calculated request | Actual output | Matches |
| --- | --- | --- | --- | --- |
| 01 | 3678 x 4598 | 2576 x 3216 | 2576 x 3216 | Yes |
| 02 | 3678 x 4598 | 2576 x 3216 | 2576 x 3216 | Yes |
| 03 | 4598 x 3678 | 3216 x 2576 | 3216 x 2576 | Yes |
| 04 | 3600 x 4500 | 2576 x 3216 | 2576 x 3216 | Yes |
| 05 | 2592 x 3200 | 2592 x 3200 | 2592 x 3200 | Yes |
| 06 | 2576 x 3216 | 2576 x 3216 | 2576 x 3216 | Yes |
| 07 | 1800 x 2250 | 2576 x 3216 | 2576 x 3216 | Yes |
| 08 | 4000 x 4000 | 2880 x 2880 | 2880 x 2880 | Yes |
| 09 | 3000 x 4000 | 2496 x 3312 | 2496 x 3312 | Yes |
| 10 | 3000 x 4500 | 2352 x 3520 | 2352 x 3520 | Yes |
| 11 | 2160 x 3840 | 2160 x 3840 | 2160 x 3840 | Yes |
| 12 | 4500 x 1500 | 3840 x 1280 | 3840 x 1280 | Yes |

## Conclusions

- 12/12 dimensions match the current NanoSeed algorithm. The repeated input returns the same dimensions.
- Tests 4 and 7 share an aspect ratio but differ in source resolution; both return 2576 x 3216. This supports ratio-driven target sizing at this setting.
- Tests 5 and 6 retain their respective valid input sizes, despite nearly identical aspect ratios. A single fixed portrait bucket would not explain both.
- Test 12 is limited by the 3840-pixel edge, so it uses only 4,915,200 pixels. The area ceiling is not a requirement to fill every output.
- The previous global area-first heuristic agrees with only tests 5, 8, 10, and 11 (4/12). It should not be the default for this workflow.
- The earlier 3678 x 4598 -> 2592 x 3200 observation remains unexplained; the repeated controlled tests instead produce 2576 x 3216.
- These tests establish size behavior for this NanoSeed/Sunburst workflow, not provider auto sizing, Flare, other resolutions, or pixel-level preservation.

## Evidence provenance

- NanoSeed source SHA-256 at analysis: `7cb102294d1bfde7c77b0faf838862b842d1b36dbb09e0115b66886de5ec2408`

The maintainer supplied 12 generated JPEGs. The input/output images are not distributed. Requests were reconstructed from the NanoSeed source, whose hash is recorded above.
