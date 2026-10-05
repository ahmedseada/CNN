# DigitsCnn

A .NET 10 console app that trains a small convolutional network to recognise handwritten digits 0-9 (MNIST). It uses
[Idrak](https://www.nuget.org/packages/Idrak) 0.3.0 and its fluent API. The data is downloaded with the Idrak CLI.

## 1. Set up the Idrak CLI

From the repository root:

```sh
dotnet tool restore        # installs Idrak.Cli 0.3.0 from dotnet-tools.json
dotnet idrak version       # check: prints the tool, library and runtime versions
```

## 2. Download MNIST with `idrak data download`

From this folder (`samples/DigitsCnn`):

```sh
dotnet idrak @mnist.rsp
```

[`mnist.rsp`](mnist.rsp) is an Idrak argument file (`@FILE` reads one argument per line). It is the same as running:

```sh
dotnet idrak data download \
  https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz \
  https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz \
  https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz \
  https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz \
  --cache data
```

This downloads four files (about 11 MB) into `data/downloads/urls/ossci-datasets.s3.amazonaws.com/mnist/`:

| File | Contents |
|------|----------|
| `train-images-idx3-ubyte.gz` | 60,000 training images, 28 x 28 greyscale |
| `train-labels-idx1-ubyte.gz` | Their digits |
| `t10k-images-idx3-ubyte.gz`  | 10,000 test images |
| `t10k-labels-idx1-ubyte.gz`  | Their digits |

Running the command again does nothing if the files are already downloaded; add `--refresh` to download them again.
`dotnet idrak data cache` shows where Idrak keeps its downloads.

The app searches for these files by name under its data folder (`data`, or the folder passed with `--data`). It then
searches Idrak's default cache (`IDRAK_CACHE`, otherwise `~/.cache/idrak`). So these also work:

- **Using Idrak's shared cache:** drop `--cache data` from the command above. The files go to the shared cache and
  every sample can use them.
- **Copying files by hand:** put the four `.gz` files anywhere under `data/`.

If the S3 mirror is unreachable, use the same file names from `https://storage.googleapis.com/cvdf-datasets/mnist/`.

## 3. Train

```sh
dotnet run -c Release -- train
```

The app trains for up to 3 epochs, printing the loss and accuracy after each one. A `*` marks an epoch with a new
lowest validation loss.

The saved model always uses the weights of the best epoch (lowest validation loss), not the last epoch. If
validation loss doesn't improve for `--patience` epochs, training stops early. Either way, Idrak's `RestoreBestWeights`
(on by default) puts back the best epoch's weights at the end. The app then prints a confusion matrix of the
10,000 test images, shows a few test digits as text drawings with their predictions, and saves the model package to
`digits.ikm`. The package works in this app (step 4) and in the Idrak CLI (step 5).

| Option             | Default      | Meaning                                         |
|--------------------|--------------|-------------------------------------------------|
| `--epochs`         | 3            | Passes over the training set (at most)          |
| `--patience`       | 2            | Stop after N epochs without a lower validation loss |
| `--batch`          | 64           | Batch size                                      |
| `--train-samples`  | all (60,000) | Train on N images chosen at random (seeded)          |
| `--model`          | `digits.ikm` | Where to save the model package                 |
| `--data`           | `data`       | Folder to search for the MNIST files            |

Measured runs:

| Run | Device | Training time | Test accuracy |
|-----|--------|---------------|---------------|
| `train` (3 epochs, 60,000 images) | `cuda:0` (NVIDIA GPU) | about 12 seconds | 98.7% |
| `train` (3 epochs, 60,000 images) | 4-core CPU | about 2 minutes | 99.1% |
| `train --epochs 1 --train-samples 10000` | 4-core CPU | about 10 seconds | 96.0% |

## 4. Predict

```sh
dotnet run -c Release -- export                                   # test images to try (step 5)
dotnet run -c Release -- predict test-digits/00000_label7.pgm     # one of them
dotnet run -c Release -- predict C:\path\to\your-drawing.png      # or any digit image of your own
```

The app accepts PNG, BMP, PGM and PPM files. It converts the image to greyscale and resizes it to 28 x 28. If the
digit is dark on a light background, the app inverts it to match MNIST's white-on-black style. It then prints the
image, the predicted digit and the top three probabilities.

## 5. Inference with the Idrak CLI

`digits.ikm` also runs without this app, in `idrak predict`. The CLI takes one image or a folder of images and
prints each file's digit and probability.

The CLI uses images exactly as given. It does not invert dark-on-light drawings, so give it MNIST-style images:
a white digit on a black background. To get some, export test-set digits from the downloaded data:

```sh
dotnet run -c Release -- export                 # 20 images to test-digits/, 2 of each digit (--count N, --out DIR)
```

The files are 28 x 28 PGM images named `<index>_label<digit>.pgm`, where `<index>` is the image's place in the test
set, so each name shows the correct answer. The app picks one image of each digit in turn. Then run:

```sh
# a folder: one row per image
dotnet idrak predict digits.ikm -i test-digits

# one image, with the three most likely digits
dotnet idrak predict digits.ikm -i test-digits/00000_label7.pgm --top 3

# every row to a file (.csv, .jsonl or .json), with full file names
dotnet idrak predict digits.ikm -i test-digits --top 3 -o predictions.csv

# machine-readable output, on a chosen device
dotnet idrak predict digits.ikm -i test-digits --json -d cuda:0
```

```
file              prediction  probability
----------------  ----------  -----------
00000_label7.pgm  7           0.999578
00001_label2.pgm  2           0.998377
00002_label1.pgm  1           0.998937
```

To list what the package contains, run `dotnet idrak inspect digits.ikm`. `predictor.Save` writes three entries
besides its `manifest.json`, and both this app (`Predictor.Load`) and `idrak predict` read all of them:

| Entry | Contents |
|-------|----------|
| `architecture/model.json` | The network, as the builder's JSON |
| `weights/model.ikw` | The trained weights |
| `json/predictor.json` | Input shape, softmax and class names |

## How the code uses Idrak's fluent API

**Network** ([`Program.cs`](Program.cs)): the network builder tracks the shape after each layer, so no layer needs its
input size:

```csharp
using var model = Network.Image(channels: 1, height: 28, width: 28)
    .Seed(1)
    .Conv2d(32, kernelSize: 3, padding: 1).ReLU().MaxPool2d(2)    // 32 x 14 x 14
    .Conv2d(64, kernelSize: 3, padding: 1).ReLU().MaxPool2d(2)    // 64 x 7 x 7
    .Flatten()
    .Linear(128).ReLU().Dropout(0.25f)
    .Linear(10)
    .Build();
```

**Data** ([`Mnist.cs`](Mnist.cs), with the IDX reader in [`Idx.cs`](../../src/CnnSamples.Shared/Idx.cs)): the IDX
files become datasets through Idrak's data extensions:

```csharp
Dataset.FromClassLabels(images, labels, 10, Digits).WithFeatureShape(1, 28, 28);
train.Subset(TestImages.RandomSubset(train.Count, n));   // --train-samples: a random N (shared library)
train.Batches(64, shuffle: true, seed: 1);       // a DataLoader
```

**Training**: a single `TrainingRun` object holds the model, loss, optimizer, data and metrics:

```csharp
var history = new TrainingRun
{
    Model = model, Loss = Losses.CrossEntropy, Optimizer = p => new Adam(p, 1e-3f),
    Train = train.Batches(64, shuffle: true), Validation = test.Batches(512),
    Epochs = 3, Metrics = [Metric.Accuracy],
    EarlyStoppingPatience = 2,    // stop after 2 epochs without improvement; the best epoch's weights are kept
}.Fit();
```

**Inference**: the predictor builder turns the network's outputs into a `ClassPrediction` (the digit, its probability
and every digit's score). It also saves and reloads the model as one package:

```csharp
using var predictor = Predictor.For(model)
    .InputShape(1, 28, 28)   // large inputs (the 10,000 test images) go through in batches of 512
    .Softmax()
    .Classes(Digits)
    .Build();
predictor.Save("digits.ikm");   // for this app's predict and for idrak predict

var saved = Predictor.Load("digits.ikm");
using var loaded = saved.Classes(saved.StoredClasses!).Build();
ClassPrediction answer = loaded.Predict(pixels);
```
