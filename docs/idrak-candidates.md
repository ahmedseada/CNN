# Candidates for Idrak

Pieces built in these samples first, as workarounds on the published Idrak, that are meant to move into the library
once they have settled. The rule for the move: only what is general (useful to networks beyond the sample that found
it) becomes a library abstraction, with an interface the application can plug its own implementation into; what is
specific to one application stays in that application. Every piece that moves keeps memory (RAM and VRAM) and speed
in mind: no buffer more than it needs, reuse over allocation, work that can run on the device runs there.

## Planned for 0.3.2

Abstractions taken from the tables below, each with an interface the application can plug its own implementation
into, and each measured for memory and speed before it moves:

- **Telemetry for whole pipelines**: `Telemetry.Stage(name)` (a disposable scope publishing `StageCompleted` under a
  new `TelemetryLevel.Stages`) for applications and the library alike; stage events from `Idrak.Vision`
  (`Foreground.Extract`, `ConnectedComponents.Find`, the framing in `RegionClassifier`, `ModelDetector`,
  `ModelSegmenter`), from model loading (`Predictor.Load`, `RegionClassifier.Load`, `ModelPackage`) and from
  `ImageCodecs.Decode`; `MemoryUsage.Peak` with a reset and the device's memory on `InferenceCompleted`; a `First`
  flag (or a kernel-compilation event) on the first inference; the model's package name on its events.
- **Foreground thresholds**: `IForegroundThreshold` taken by `Foreground.Extract` (global Otsu, a given level,
  Bradley's local window; Sauvola next), writing into the `ForegroundImage` buffer, parallel over rows.
- **Region measures**: perimeter (and stroke thickness, `2 · area / perimeter`) per region from the labelling pass of
  `ConnectedComponents.Find`.
- **Image converters**: an `IImageConverter` registry beside `ImageCodecs`, which `Decode` falls back to.
- **Speed and memory of a page** (measured on an RTX 5050 laptop, 1188 x 1280 photo, 75 characters, warm: about
  100 ms a page, of which the model is 7 ms): `Foreground` and the local threshold parallel over rows and vectorized,
  into the `ForegroundImage` buffer (photo ink about 100 ms today); `RegionClassifier` framing regions in parallel
  into its batch buffer (4-10 ms on one thread today), a warm-up at load (the first classify is 43 ms against 11 ms
  warm) and the top classes without sorting all of them; `ImageCodecs.Decode(path, channels: 1)` straight to grey
  (a 1188 x 1280 page is 18 MB as float RGB, 1.5 MB as grey bytes); pooled per-page buffers (grey, ink, integral
  image, label map) reused across stages and pages; and the device's memory pool reported and trimmable (245 MB
  cached for 8 MB in use after one page).

Later: a pure C# JPEG codec (baseline and progressive, no external program, no temporary file), the CTC loss and
decoding for reading whole lines (0.4.0), and a .NET metrics and activities bridge for `dotnet-counters` and
OpenTelemetry (an optional package).

## Open

| Piece | Built in | Why it is general | Library shape | Memory and speed | Target |
|-------|----------|-------------------|---------------|------------------|--------|
| Local (adaptive) threshold | MultiLanguageOcr `PhotoPage.LocalInk` | Uneven light and shadows in any photographed image: documents, microscopy, inspection, plates | An `IForegroundThreshold` (global Otsu, given level, local window) taken by `Foreground.Extract`; Bradley's mean window first, Sauvola (mean and deviation) next | Today one float per pixel for grey, a double per pixel for the integral image, a float per pixel for the result (899 x 1599: 5.7 + 11.5 + 5.7 MB). In the library: write into the `ForegroundImage` buffer, the integral image in rows of doubles reused per call (or float sums of 64-pixel tiles), parallel over rows; a device kernel for large images | 0.3.2 |
| Image conversion to a readable format | MultiLanguageOcr `ImageConversion` (`IImageConverter`: Windows imaging, ImageMagick, ffmpeg) | Every image-reading network meets JPEG, WebP, HEIC and TIFF | An `IImageConverter` registry next to `ImageCodecs` (`ImageCodecs.Decode` falls back to it), or better a pure C# JPEG codec (baseline and progressive) registered as a codec, no external program | A converter writes a BMP once (cached next to the source); a codec decodes in memory with no file at all, straight into `ImageData` | 0.3.2 (the converter registry); the JPEG codec after |
| Stroke-thickness filter for regions | MultiLanguageOcr `PhotoPage.RemoveSurroundings` | Telling thin strokes (writing, wires, cracks, vessels) from thick blobs is common in vision pre-processing | Region measures in `ConnectedComponents` (perimeter, `2 · area / perimeter`), so filters need no second pass over the label map | One int per region for the perimeter, computed in the labelling pass instead of a second scan | 0.3.2 |
| Character-sequence recognition (CTC) | Not started | Joined handwriting, printed text lines, speech: any sequence read without per-item boxes | `Losses.Ctc` (forward-backward on the device) and greedy and beam decoding | The loss over [time, classes] per sequence in log space, batched; no [time × labels] matrix on the host | 0.4.0 |

## Telemetry gaps

Found by timing MultiLanguageOcr's `ocr` with Idrak's telemetry (`InferenceLog.cs`): the model's inference is
covered (`TelemetryLevel.Inference`, one `InferenceCompleted` per batch, timed with the device synchronized); the rest
of a page is not, so the sample still times it with a stopwatch.

| Gap | Today | Library shape | Memory and speed | Target |
|-----|-------|---------------|------------------|--------|
| Applications cannot publish their own stages | Every publishing method of `Telemetry` is internal; a hook only receives the library's events, so an application's pipeline cannot appear in the same console, JSON Lines or recorder output | `Telemetry.Stage("ink")` returning a disposable scope that publishes `StageCompleted(Name, Duration, Device?, Memory?)` under a new `TelemetryLevel.Stages` | One static read and an AND when nobody listens, as today's sources | 0.3.2 |
| `Idrak.Vision` publishes nothing | `Foreground.Extract`, `ConnectedComponents.Find`, `ContentFrame` framing inside `RegionClassifier.Classify` (CPU work beside the model), `ModelDetector` decoding and suppression, `ModelSegmenter` | Stage events from each, through the same `Stages` level | Timestamps only when enabled | 0.3.2 |
| Model loading is not reported | `Predictor.Load`, `RegionClassifier.Load` and `ModelPackage` publish nothing (only `InferenceEngine` reports `ModelLoaded`) | A load event: package, bytes read, parameters, device, duration | | 0.3.2 |
| Image decoding is not reported | `ImageCodecs.Decode` (and a converter's run) | A stage event with the codec, size and duration | | 0.3.2 |
| No memory with inference, no peak anywhere | `InferenceCompleted` has no memory (`EpochCompleted` has `MemoryUsage`); `MemoryUsage` has in use, cached and limit but no peak, so a page's VRAM peak cannot be read | `MemoryUsage.Peak` (with a reset), and the device's memory on `InferenceCompleted` | A peak is one compare per allocation in the memory pool | 0.3.2 |
| First call and warm calls look alike | The first `Predict` includes compiling kernels and JIT; `InferenceCompleted` cannot tell it apart (MultiLanguageOcr runs a warm repeat to show the steady speed) | A `First` flag on `InferenceCompleted`, or a kernel-compilation event | | 0.3.2 |
| The model's name is its type | `InferenceCompleted.Model` is the module's display name ("Sequential(5 layers)"), not the package it came from | The package name or a name given at load | | 0.3.2 |
| No .NET metrics bridge | No `System.Diagnostics.Metrics` or `ActivitySource`, so `dotnet-counters` and OpenTelemetry see nothing | A hook that turns events into meters and activities, in the core or an optional package | Only when subscribed | 0.4.0 (optional package) |

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
