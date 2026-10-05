# Candidates for Idrak

Pieces built in these samples first, as workarounds on the published Idrak, that are meant to move into the library
once they have settled. The rule for the move: only what is general (useful to networks beyond the sample that found
it) becomes a library abstraction, with an interface the application can plug its own implementation into; what is
specific to one application stays in that application. Every piece that moves keeps memory (RAM and VRAM) and speed
in mind: no buffer more than it needs, reuse over allocation, work that can run on the device runs there.

## Open

| Piece | Built in | Why it is general | Library shape | Memory and speed |
|-------|----------|-------------------|---------------|------------------|
| Local (adaptive) threshold | MultiLanguageOcr `PhotoPage.LocalInk` | Uneven light and shadows in any photographed image: documents, microscopy, inspection, plates | An `IForegroundThreshold` (global Otsu, given level, local window) taken by `Foreground.Extract`; Bradley's mean window first, Sauvola (mean and deviation) next | Today one float per pixel for grey, a double per pixel for the integral image, a float per pixel for the result (899 x 1599: 5.7 + 11.5 + 5.7 MB). In the library: write into the `ForegroundImage` buffer, the integral image in rows of doubles reused per call (or float sums of 64-pixel tiles), parallel over rows; a device kernel for large images |
| Image conversion to a readable format | MultiLanguageOcr `ImageConversion` (`IImageConverter`: Windows imaging, ImageMagick, ffmpeg) | Every image-reading network meets JPEG, WebP, HEIC and TIFF | An `IImageConverter` registry next to `ImageCodecs` (`ImageCodecs.Decode` falls back to it), or better a pure C# JPEG codec (baseline and progressive) registered as a codec, no external program | A converter writes a BMP once (cached next to the source); a codec decodes in memory with no file at all, straight into `ImageData` |
| Stroke-thickness filter for regions | MultiLanguageOcr `PhotoPage.RemoveSurroundings` | Telling thin strokes (writing, wires, cracks, vessels) from thick blobs is common in vision pre-processing | Region measures in `ConnectedComponents` (perimeter, `2 · area / perimeter`), so filters need no second pass over the label map | One int per region for the perimeter, computed in the labelling pass instead of a second scan |
| Character-sequence recognition (CTC) | Not started | Joined handwriting, printed text lines, speech: any sequence read without per-item boxes | `Losses.Ctc` (forward-backward on the device) and greedy and beam decoding | The loss over [time, classes] per sequence in log space, batched; no [time × labels] matrix on the host |

## Stays in the samples

- Ruled-line removal (notebook paper) and the page's surroundings by position (binding, cover, desk): specific to
  photographed pages of text. `PhotoPage` in MultiLanguageOcr.
- Text-line layout (lines and characters by projection gaps, Arabic dots joined to their line): `TextLineProposer`, an
  `IRegionProposer` plugged into Idrak's `RegionClassifier`.
- Word context (letters or digits by probability, look-alike characters): `Words` in MultiLanguageOcr.

## Moved

- Idrak 0.3.0: `Idrak.Vision` (foreground, connected components, content framing, region classification, boxes and
  non-maximum suppression, detection and segmentation abstractions, channel normalization).
- Idrak 0.3.1: smooth enlarging in `ContentFrame` (found by MultiLanguageOcr's Arabic letters).
