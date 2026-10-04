# LettersCnn

A .NET 10 console app that trains a convolutional network to recognise handwritten English letters A-Z (the EMNIST
Letters dataset). It uses [Idrak](https://www.nuget.org/packages/Idrak) 0.2.1 and its fluent API. The data is
downloaded with the Idrak CLI.

EMNIST Letters has 26 classes. Each class holds both cases of a letter, so `A` means "A or a". Some letters look alike
in handwriting, so expect more mistakes between pairs like I/L, G/Q and U/V than between digits.

## 1. Set up the Idrak CLI

From the repository root:

```sh
dotnet tool restore        # installs Idrak.Cli 0.2.1 from dotnet-tools.json
dotnet idrak version       # check: prints the tool, library and runtime versions
```

## 2. Download EMNIST with `idrak data download`

From this folder (`samples/LettersCnn`):

```sh
dotnet idrak @emnist.rsp
```

[`emnist.rsp`](emnist.rsp) is an Idrak argument file. It is the same as running:

```sh
dotnet idrak data download https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip --cache data
```

This downloads NIST's EMNIST archive, `gzip.zip` (about 560 MB). The archive holds every EMNIST split (letters, digits,
balanced, by class...). The file lands in `data/downloads/urls/biometrics.nist.gov/cs_links/EMNIST/gzip.zip`. You
don't need to unzip it: the app reads these four entries straight from the zip.

| Entry | Contents |
|-------|----------|
| `gzip/emnist-letters-train-images-idx3-ubyte.gz` | 124,800 training images, 28 x 28 greyscale |
| `gzip/emnist-letters-train-labels-idx1-ubyte.gz` | Their letters, 1 (A) to 26 (Z) |
| `gzip/emnist-letters-test-images-idx3-ubyte.gz`  | 20,800 test images |
| `gzip/emnist-letters-test-labels-idx1-ubyte.gz`  | Their letters |

The app searches by file name under its data folder (`data`, or the folder passed with `--data`), then under Idrak's
default cache (`IDRAK_CACHE`, otherwise `~/.cache/idrak`). So these also work:

- **Using Idrak's shared cache:** drop `--cache data` from the command, so the download goes to Idrak's cache.
- **Using already-extracted files:** put the four `emnist-letters-*.gz` files anywhere under `data/`. The app uses
  them instead of the zip.

EMNIST stores each image transposed (rows and columns swapped) relative to MNIST. The app transposes the images back
when it loads them, so the letters it prints and exports are upright.

## 3. Train

```sh
dotnet run -c Release -- train
```

The app trains for up to 8 epochs, printing the loss, accuracy and learning rate after each one. A `*` marks an epoch
with a new lowest validation loss. Training stops early when validation loss doesn't improve for `--patience`
epochs, and the saved model always uses the weights of the best epoch.

It then prints each letter's test accuracy and the most common mistakes, shows a few test letters as text drawings
with their predictions, and saves the model package to `letters.ikm`.

| Option             | Default       | Meaning                                              |
|--------------------|---------------|------------------------------------------------------|
| `--epochs`         | 8             | Passes over the training set (at most)               |
| `--patience`       | 3             | Stop after N epochs without a lower validation loss  |
| `--batch`          | 128           | Batch size                                           |
| `--train-samples`  | all (124,800) | Train on N images chosen at random (seeded)          |
| `--model`          | `letters.ikm` | Where to save the model package                      |
| `--data`           | `data`        | Folder to search for the EMNIST files                |

This network is larger than DigitsCnn's and trains on twice as many images, so each epoch takes longer. A GPU helps a
lot. For a quick check, use `--epochs 2 --train-samples 20000`.

A measured run:

| Run | Device | Training time | Test accuracy |
|-----|--------|---------------|---------------|
| `train` (8 epochs, 124,800 images) | `cuda:0` (NVIDIA RTX 5070 Ti) | about 2.3 minutes (17 s per epoch) | 95.0% |

Most mistakes in that run were between letters whose two cases look like another letter's: I and L (`I`, `l`), G and Q
(`g`, `q`), and U and V. Published results for EMNIST Letters are around 95-96%.

## 4. Predict

```sh
dotnet run -c Release -- export                                    # test images to try (step 5)
dotnet run -c Release -- predict test-letters/00000_labelA.pgm     # one of them
dotnet run -c Release -- predict C:\path\to\your-drawing.png       # or any letter image of your own
```

The app accepts PNG, BMP, PGM and PPM files. It converts the image to greyscale, resizes it to 28 x 28 and, if the
letter is dark on a light background, inverts it to match EMNIST's white-on-black style. It then prints the image, the
predicted letter and the top three probabilities.

## 5. Inference with the Idrak CLI

`letters.ikm` also runs in `idrak predict`. The CLI uses images as given, so give it white-on-black letters. To get
some, export test images:

```sh
dotnet run -c Release -- export                    # 26 images to test-letters/, one per letter (--count N, --out DIR)
dotnet idrak predict letters.ikm -i test-letters --top 3
dotnet idrak predict letters.ikm -i test-letters --top 3 -o predictions.csv
```

The files are named `<index>_label<letter>.pgm`, where `<index>` is the image's place in the test set, so each name
shows the correct answer. EMNIST's test set is sorted by letter (the first 800 images are all A), so the app picks
one image of each letter in turn instead of the first 26.

## How the code uses Idrak

**Network** ([`Program.cs`](Program.cs)): two blocks of two convolutions, each followed by batch normalization, then a
classifier head:

```csharp
var network = Network.Image(channels: 1, height: 28, width: 28)
    .Seed(1)
    .Conv2d(32, kernelSize: 3, padding: 1).BatchNorm().ReLU()
    .Conv2d(32, kernelSize: 3, padding: 1).BatchNorm().ReLU().MaxPool2d(2)    // 32 x 14 x 14
    .Conv2d(64, kernelSize: 3, padding: 1).BatchNorm().ReLU()
    .Conv2d(64, kernelSize: 3, padding: 1).BatchNorm().ReLU().MaxPool2d(2)    // 64 x 7 x 7
    .Flatten()
    .Linear(256).ReLU().Dropout(0.4f)
    .Linear(26);
```

**Training**: compared with DigitsCnn, this run adds label smoothing, AdamW with weight decay, and a cosine learning-rate
schedule with one warm-up epoch:

```csharp
var history = new TrainingRun
{
    Model = model,
    Loss = (logits, targets) => Losses.CrossEntropy(logits, targets, 0.1f),   // label smoothing 0.1
    Optimizer = p => new AdamW(p, 2e-3f, weightDecay: 1e-4f),
    Scheduler = optimizer => new CosineAnnealing(optimizer, totalEpochs: 8, minLearningRate: 1e-5f, warmupEpochs: 1),
    Train = train.Batches(128, shuffle: true, seed: 1), Validation = test.Batches(512),
    Epochs = 8, Metrics = [Metric.Accuracy], EarlyStoppingPatience = 3,
}.Fit();   // the best epoch's weights are kept, whether early stopping ended the run or not
```

**Data** ([`Emnist.cs`](Emnist.cs)): the entries are read from the zip with `System.IO.Compression`. The shared IDX
reader transposes the images, and labels 1-26 become class indices 0-25:

```csharp
Dataset.FromClassLabels(images, labels, 26, Letters).WithFeatureShape(1, 28, 28);
```

The predictor, the saved package and the console output are the same as DigitsCnn's. They live in
[`src/CnnSamples.Shared`](../../src/CnnSamples.Shared/README.md).
