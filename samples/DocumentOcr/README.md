# DocumentOcr

A .NET 10 console app that reads the text of a page of handwritten characters (optical character recognition). It
uses [Idrak](https://www.nuget.org/packages/Idrak) 0.3.0 and its fluent API. A convolutional network recognises each
character, and a page reader written in C# finds the lines and characters and assembles the text.

How it works:

1. **Character model.** The network trains on EMNIST Balanced: 131,600 handwritten characters in 47 classes. The
   classes are digits 0-9, capitals A-Z, and the 11 lower-case letters that look different from their capitals
   (`a b d e f g h n q r t`). Letters like `c`/`C` or `o`/`O` share a class.
2. **Page reader** ([`PageReader.cs`](PageReader.cs)):
   - separates ink from paper with Otsu's threshold (dark ink on light paper or the other way round, any colour);
   - finds text lines from the rows that have ink;
   - splits each line into characters at the columns without ink, and splits blobs much wider than one character;
   - marks a space where the gap is wider than 0.4 of the line height;
   - centres each character in a square and scales it to 28 x 28, the way EMNIST built its images.
3. **Recognition.** All characters of the page go through the model in one batch.
4. **Word context.** In a word that is mostly letters, a digit becomes the letter it looks like (`1` → `I`, `0` → `O`,
   `5` → `S`, `2` → `Z`, `8` → `B`, `6` → `G`), or the model's most likely letter if it has no look-alike. In a word that
   is mostly digits, it works the other way round (`O`/`D` → `0`, `I`/`L` → `1`, `S` → `5`...). Each word then takes the
   case of most of its letters, so `tHE` becomes `THE`.

The reader expects separate characters on clean lines: printed-style handwriting, form fields, the `demo` pages. It
can't read joined-up (cursive) handwriting; that needs a model that reads whole words.

## 1. Set up the Idrak CLI

From the repository root:

```sh
dotnet tool restore        # installs Idrak.Cli 0.3.0 from dotnet-tools.json
```

## 2. Download EMNIST

DocumentOcr uses the same download as [LettersCnn](../LettersCnn/README.md), NIST's `gzip.zip` (about 560 MB). If you
already downloaded it for LettersCnn, skip this step: the app also looks in `../LettersCnn/data`.

Otherwise, from this folder (`samples/DocumentOcr`):

```sh
dotnet idrak @emnist.rsp     # = dotnet idrak data download https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip --cache data
```

The app reads these entries straight from the zip:

| Entry | Contents |
|-------|----------|
| `gzip/emnist-balanced-train-images-idx3-ubyte.gz` | 112,800 training images, 28 x 28 greyscale |
| `gzip/emnist-balanced-train-labels-idx1-ubyte.gz` | Their classes, 0 to 46 |
| `gzip/emnist-balanced-test-images-idx3-ubyte.gz`  | 18,800 test images |
| `gzip/emnist-balanced-test-labels-idx1-ubyte.gz`  | Their classes |
| `gzip/emnist-balanced-mapping.txt`                | Each class's character |

The search order is: `data` (or `--data DIR`), then `../LettersCnn/data`, then Idrak's cache (`IDRAK_CACHE`, otherwise
`~/.cache/idrak`). Extracted `.gz` files anywhere in those folders work as well as the zip.

## 3. Train the character model

```sh
dotnet run -c Release -- train
```

It trains for up to 10 epochs, then prints each character's test accuracy and the most common mistakes, and saves the
model to `ocr.ikm`.

Characters cut out of a page are never centred as precisely as EMNIST's. So training uses Idrak's image transforms on
every batch: each image is moved by up to 2 pixels and turned by up to 8 degrees. Training accuracy therefore stays
below test accuracy.

| Option             | Default       | Meaning                                              |
|--------------------|---------------|------------------------------------------------------|
| `--epochs`         | 10            | Passes over the training set (at most)               |
| `--patience`       | 3             | Stop after N epochs without a lower validation loss  |
| `--batch`          | 128           | Batch size                                           |
| `--train-samples`  | all (112,800) | Train on N images chosen at random (seeded)          |
| `--model`          | `ocr.ikm`     | Where to save the model package                      |
| `--data`           | `data`        | Folder to search first for the EMNIST files          |

Expect the most mistakes between characters that look the same in handwriting: `0`/`O`, `1`/`I`/`L`, `5`/`S`,
`2`/`Z`, `9`/`q`, `g`/`9`. Word context fixes most of the digit/letter ones on a page.

## 4. Read a demo page

```sh
dotnet run -c Release -- demo
```

`demo` writes a page of handwriting for a known text and reads it back. The page is built from EMNIST test
characters, which the model never trained on, drawn as dark ink on white paper at twice EMNIST's size. It's saved to
`page.pgm`. The app prints the expected text, what it read character by character, and what it read with word
context, and the character error rate of each.

```
Expected:
HANDWRITTEN OCR WITH IDRAK
THE QUICK BROWN FOX JUMPS
OVER 13 LAZY DOGS IN 2026

Read, character by character:
HANDWR1TTEN OCR WLTH JDRAK
THE QUICK BROWN FOX JUMP5
OVER 13 LAZY DOG5 JN 2O26

Read, with each word's letters or digits:
HANDWRITTEN OCR WLTH JDRAK
THE QUICK BROWN FOX JUMPS
OVER 13 LAZY DOGS JN 2026

Character error rate, character by character: 9.0 % (7 edits over 78 characters)
Character error rate, with words: 3.8 % (3 edits over 78 characters)
```

That output is from a small test model, not one trained on the full data set. Your numbers will differ. The error
rate ignores case, because EMNIST Balanced shares one class between `c` and `C`, `o` and `O`, and so on.

A measured run: trained on all of EMNIST Balanced on an NVIDIA RTX 5070 Ti, 10 epochs in about 2.6 minutes, 89.1% test
accuracy. Published results for this split are around 88-91%. Most of its mistakes are pairs that handwriting makes
ambiguous: `F`/`f`, `L`/`1`, `O`/`0`, `I`/`1`, `q`/`9`.

Options: `--text "FIRST LINE\nSECOND LINE"` for your own text (digits, letters and spaces; `\n` starts a new line),
and `--out FILE` for where to write the page.

## 5. Read your own page

```sh
dotnet run -c Release -- ocr page.pgm                          # the demo's page
dotnet run -c Release -- ocr C:\path\to\scan.png               # a scan or photo of your own
```

The app accepts PNG, BMP, PGM and PPM files. It prints how many characters and lines it found, the three characters
it was least sure about (with the alternatives), and the text.

For the best results:

- write in capitals or separate letters, with a clear gap between characters and a larger one between words;
- keep lines straight and apart, since the reader doesn't correct tilted pages;
- crop the scan to the text, and use plain paper (no lines or squares).

## How the code uses Idrak

- **Data**: `Dataset.FromClassLabels(...).WithFeatureShape(1, 28, 28)` and `train.Subset(...)`. The EMNIST zip reader
  is shared with LettersCnn ([`EmnistArchive.cs`](../../src/CnnSamples.Shared/EmnistArchive.cs)).
- **Augmentation**: a `DataLoader` with `Transforms = [new RandomShift(2), new RandomRotation(8)]`.
- **Network and training**: the same builder chain and `TrainingRun` as LettersCnn (four convolutions with
  `BatchNorm`, `AdamW`, `CosineAnnealing`, label smoothing).
- **Inference**: `Predictor.Load(...).Classes(...).Build()`, then one `Predict(list)` call for every character on the
  page. `ClassPrediction.Scores` gives the alternatives that word context chooses from.
- **Images**: `ImageCodecs.Decode` reads the page at full size, `ImageData.Resize(1, height, width)` turns colour into
  grey, and `ImageData.Resize(1, 28, 28)` shrinks each character. Since Idrak 0.2.1 it averages the pixels it shrinks,
  so thin pen strokes survive (0.2.0 sampled single points and could step over them).
