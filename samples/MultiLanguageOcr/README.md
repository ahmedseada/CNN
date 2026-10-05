# MultiLanguageOcr

A .NET 10 console app that reads pages of handwritten characters in **English and Arabic**. It uses
[Idrak](https://www.nuget.org/packages/Idrak) 0.3.1 and its fluent API. It builds on [DocumentOcr](../DocumentOcr/README.md),
which reads English only.

How it works:

1. **One model, 85 characters, two scripts.** The network trains on three data sets at once:

   | Script | Data set | Characters |
   |--------|----------|------------|
   | Latin | EMNIST Balanced | 47: digits `0-9`, capitals `A-Z`, and `a b d e f g h n q r t` |
   | Arabic | AHCD (Arabic Handwritten Characters Dataset) | 28 letters: `ا ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي` |
   | Arabic | MADBase / AHDD (Arabic Handwritten Digits) | 10 Arabic-Indic digits: `٠ ١ ٢ ٣ ٤ ٥ ٦ ٧ ٨ ٩` |

   Every training image is framed by Idrak's `ContentFrame.Fit`: cropped to its ink, centred and scaled to 28 x 28,
   the same way Idrak frames a character cut from a page. So EMNIST's 28 x 28, AHCD's 32 x 32 and MADBase's images
   all look alike to the model, and like the characters of a page.
2. **Page layout** ([`TextLines.cs`](TextLines.cs)), plugged into Idrak's `RegionClassifier` as an `IRegionProposer`.
   Idrak separates the ink from the paper (either polarity, Otsu's threshold), frames each character and classifies
   them in batches on the device. The sample supplies the part that is specific to text: it finds lines and
   characters by the gaps between them, like DocumentOcr's. One addition for Arabic: a band of rows much thinner than
   a line (the dots above and below letters such as `ت` `ث` `ب`) joins the nearest line instead of becoming a line of
   its own. A letter's dots sit within its width, so they stay part of the letter.
3. **One script per line.** Some characters look alike across scripts: `V`/`٧`, `l`/`ا`, `0`/`٥`/`ه`. So each line takes
   the script that most of its probability is on, and every character on it is then read within that script.
4. **Word context** ([`Words.cs`](Words.cs)), per script: a word is letters or digits by the probability its
   characters put on each, summed over the word. So `HELLO` stays a word even when the model's likeliest reading of
   both `L`s is `1` and of the `O` is `0`. In a word of letters, a digit becomes the likeliest of the letters it looks
   like, by the model's own probabilities (`1` → `I` or `L`, `0` → `O` or `D`, `١` → `ا`, `٥` → `ه`). In a word of
   digits it works the other way round. English words then take one case.
5. **Reading order** ([`TextOrder.cs`](TextOrder.cs)). Arabic lines are read right to left: the words from right to
   left and each word's letters from right to left, but numbers keep their digits left to right, as Arabic writes them.

### Limits

Arabic handwriting is normally **joined**: letters connect within a word and change shape with their position (start,
middle, end). The public data sets have **isolated** letters only, so this sample reads pages of separated
characters: forms, letters written one by one, the `demo` pages. Joined Arabic (and joined English) handwriting
needs a model that reads whole words.

Letters outside AHCD's 28 aren't known either: `ة`, `ى`, `ء` and alef with hamza (`أ إ آ`).

## 1. Set up the Idrak CLI

From the repository root:

```sh
dotnet tool restore        # installs Idrak.Cli 0.3.1 from dotnet-tools.json
```

## 2. Download the data

From this folder (`samples/MultiLanguageOcr`):

**EMNIST** (about 560 MB). Skip this if DocumentOcr or LettersCnn already downloaded it: the app looks in their
`data` folders too.

```sh
dotnet idrak @emnist.rsp
```

**AHCD and MADBase** come from Kaggle, which needs a free account and an API token:

1. On kaggle.com, open Settings → API → **Create New Token**. That downloads `kaggle.json`.
2. Put it in `%USERPROFILE%\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json`. Or set `KAGGLE_USERNAME` and
   `KAGGLE_KEY`, or run `dotnet idrak login kaggle`.
3. Download both data sets:

```sh
dotnet idrak @arabic.rsp     # = dotnet idrak data download kaggle:mloey1/ahcd1 kaggle:mloey1/ahdd1 --cache data
```

The zips land in `data/downloads/kaggle/mloey1/ahcd1/latest/ahcd1.zip` and `.../ahdd1/latest/ahdd1.zip`. Keep them
zipped: the app reads the CSV files inside them directly.

| Archive | CSV files inside (found by name: train/test and image/label) |
|---------|------------------------------------------------------------|
| `ahcd1.zip` | 13,440 training and 3,360 test letters, one image of 32 x 32 values per line, labels 1 (ا) to 28 (ي) |
| `ahdd1.zip` | 60,000 training and 10,000 test digits, one image of 28 x 28 values per line, labels 0 to 9 |

AHCD stores each image column by column, and the app transposes it upright. MADBase stores its images row by row, as
they are. `train` prints a few Arabic test characters as the model sees them, so you can check they stand upright.

## 3. Train

```sh
dotnet run -c Release -- train
```

By default the app trains on 60,000 EMNIST images, AHCD's 13,440 letters twice (it's the smallest set), and 30,000
MADBase digits. It tests on 10,000 EMNIST, all 3,360 AHCD and 5,000 MADBase images, and trains for up to 12 epochs with
early stopping. It then prints:

- each character's test accuracy and the most common mistakes;
- for each script and kind (Latin letters, Arabic digits...), the share read correctly and the share read in the
  right script;
- a few Arabic test characters as text drawings.

The model is saved to `multilang.ikm`.

| Option             | Default          | Meaning                                                    |
|--------------------|------------------|------------------------------------------------------------|
| `--epochs`         | 12               | Passes over the training set (at most)                     |
| `--patience`       | 3                | Stop after N epochs without a lower validation loss        |
| `--batch`          | 128              | Batch size                                                 |
| `--train-samples`  | see above        | At most N images from each data set (for quick runs)       |
| `--model`          | `multilang.ikm`  | Where to save the model package                            |
| `--data`           | `data`           | Folder to search first for the downloads                   |

## 4. Read a demo page

```sh
dotnet run -c Release -- demo
```

`demo` writes a page in both scripts, made from test characters the model never trained on. The Arabic lines are
laid out right to left. Then it reads the page back and prints, for each line, the script it chose, the expected
text, what it read, and the number of edits, plus the character error rate:

```
Found 37 characters on 3 lines

line 1 (Latin)
  expected: HELLO WORLD 2026
  read:     HELLO WORLO 2026
  1 edit
line 2 (Arabic)
  expected: مرحبا بالعالم
  read:     مرحبا بالعالم
  0 edits
line 3 (Arabic)
  expected: نص عربي ١٢٣٤٥
  read:     نص عربي ١٢٣٤٥
  0 edits

Character error rate: 2.4% (1 edit over 42 characters)
```

A measured run on an NVIDIA RTX 5070 Ti: 12 epochs took about 4 minutes and reached 92.5% test accuracy over the 85
characters. Latin digits scored 87.6%, Latin letters 88.3%, Arabic letters 96.4% and Arabic digits 98.6%, and
98.1-99.3% of each kind landed in the right script. Most of the remaining mistakes are characters that look the same
on their own (`L`/`1`, `O`/`0`, `I`/`1`, `q`/`9`, and `F`/`f`, which EMNIST Balanced keeps apart). Word context
fixes many of them on a page.

One page says little: it always holds the same text and, with the same seed, the same test characters. For a fair
measure, read many pages of random words, each from other test characters (words of letters and numbers in both
scripts, with no spelling to lean on):

```sh
dotnet run -c Release -- demo --count 50              # pages 1-50; --seed 100 starts elsewhere
```

It prints every line with an error and the character error rate per script over all the pages.

Your own text: `--text "ROOM 101 OPENS AT 0815\nبيت ٢٠٢ مفتوح"` (`\n` starts a new line; use the 85 characters above).
`--out FILE` chooses where the page is written.

A Windows console shows Arabic letters unjoined, and may show a line in the wrong direction. The text read is also
saved next to the page as a UTF-8 `.txt` file (`page.txt`), and an editor such as Notepad or VS Code shows it joined
and right to left.

## 5. Read your own page

```sh
dotnet run -c Release -- ocr page.pgm                      # the demo's page
dotnet run -c Release -- ocr C:\path\to\form.png           # a scan or photo of your own
dotnet run -c Release -- ocr C:\path\to\pages              # every image in a folder, the model loaded once
```

`--show` prints each line's least certain character as the model saw it; `--bench` classifies a page a second time to
show the warm speed (the speed of every page after the first).

Each line prints with its script (`[en]` or `[ar]`), along with the least certain character of each line. The text is
also written to a `.txt` file next to the image.

For the best results, write characters separately with clear gaps (Arabic letters in their isolated forms). Keep lines
straight and apart, and crop the scan to the text.

**Photos.** A phone photo of a notebook page works too ([`PhotoPage.cs`](PhotoPage.cs)). When one threshold would
take more than 15% of the photo as ink (shadows, the desk, the binding), or one dark region spans half its width or a third
of its height (a desk, cover or paper edge around the page), the page is read as a photo:

- a local threshold: a pixel is ink where it is clearly darker than its surroundings, so uneven light does not count;
- the ruled lines of notebook paper are removed (thin strokes running far sideways), keeping the letters that cross
  them and the pen strokes that run along them (pen ink is darker than a printed line);
- what lies around the page goes: regions at the photo's edge, regions far larger than writing, and regions drawn much
  thicker than the writing (a binding's loops, a cover's edge).

`--photo` and `--no-photo` force the choice. `--crop x,y,width,height` reads only that part of the image (in pixels),
for a printed header or anything else that is not the writing:

```sh
dotnet run -c Release -- ocr notebook.jpeg --crop 120,240,750,590
```

**Formats.** PNG, BMP and PGM are read directly. Any other format (JPEG, WebP, HEIC, TIFF, GIF) is first converted to
an uncompressed BMP next to it (`photo.jpeg` → `photo.bmp`, reused on the next run) by the first converter available
([`ImageConversion.cs`](ImageConversion.cs)): Windows' own imaging on every Windows, else ImageMagick (`magick`) or
ffmpeg. A converter of your own implements `IImageConverter` and registers with `ImageConversion.Register`.

A dash (a short flat mark between words) is read as `-` without the model.

On the notebook photo in this sample's history, with a model of digits trained on MNIST framed the same way, the
number line `01345 - 16108` reads every digit.

**Speed.** An `ocr` of one image prints where the time went: opening the image, the ink, the lines and characters,
the model (its first run, and with `--bench` a warm repeat), and words and reading order; a folder prints each page's
time and the pages' average, the model's loading and first-call costs paid once;
then the device's memory as Idrak counts it and the process's peak memory. The model's time comes from Idrak's
inference telemetry (each batch timed with the device synchronized), the framing being the rest of a classify;
`--telemetry` also prints Idrak's own inference events. The other stages are timed in the sample: Idrak publishes no
events for them yet (see [docs/idrak-candidates.md](../../docs/idrak-candidates.md)).

Joined handwriting is still out of reach: all Arabic words and joined English ("on", "Read") come out as one box per
word, which a model of single characters cannot read.

## How the code uses Idrak

- **Data**: three data sets joined into one `Dataset.FromClassLabels(...)` of 85 classes. `DataLoader` with
  `Transforms = [new RandomShift(2), new RandomRotation(8)]`.
- **Network and training**: the builder chain and `TrainingRun` of the other samples. The network is a little wider
  (48 and 96 filters, 384 hidden units) for 85 classes. `EarlyStoppingPatience` keeps the best epoch.
- **Framing** (`Idrak.Vision`): `ContentFrame.Fit` frames the training images; `RegionClassifier` frames the
  characters of a page the same way, straight into one batch buffer.
- **Inference** (`Idrak.Vision`): `RegionClassifier.Load("multilang.ikm")` reads the package's model, classes and
  input size. `Foreground(image)` separates the ink, `TextLineProposer` finds the characters, and one
  `Classify(page, boxes)` call classifies all of them in batches. `Top(i, n)` gives every class's probability: summed
  per script it picks the line's script, and filtered to that script it picks each character.
- **Images**: `ImageCodecs.Decode` for the page (PNG, BMP, PGM, colour or grey).
