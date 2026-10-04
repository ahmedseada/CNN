# DigitsCnn

A .NET 10 console app that trains a small convolutional network to recognise handwritten digits 0-9 (MNIST). It uses
[Idrak](https://www.nuget.org/packages/Idrak) 0.2.0 and its fluent API. The data is downloaded with the Idrak CLI.

## 1. Set up the Idrak CLI

From the repository root:

```sh
dotnet tool restore        # installs Idrak.Cli 0.2.0 from dotnet-tools.json
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

The app trains for 3 epochs, printing the loss and accuracy after each one. It then prints a confusion matrix of the
10,000 test images, shows a few test digits as text drawings with their predictions, and saves the model package to
`digits.ikm`.

| Option             | Default      | Meaning                                         |
|--------------------|--------------|-------------------------------------------------|
| `--epochs`         | 3            | Passes over the training set                    |
| `--batch`          | 64           | Batch size                                      |
| `--train-samples`  | all (60,000) | Train on only the first N images                |
| `--model`          | `digits.ikm` | Where to save the model package                 |
| `--data`           | `data`       | Folder to search for the MNIST files            |

Measured on a 4-core CPU without a GPU:

| Run | Time | Test accuracy |
|-----|------|---------------|
| `train` (3 epochs, 60,000 images) | about 2 minutes | 99.1% |
| `train --epochs 1 --train-samples 10000` | about 10 seconds | 96.7% |

## 4. Predict

```sh
dotnet run -c Release -- predict my-digit.png
```

The app accepts PNG, BMP, PGM and PPM files. It converts the image to greyscale and resizes it to 28 x 28. If the
digit is dark on a light background, the app inverts it to match MNIST's white-on-black style. It then prints the
image, the predicted digit and the top three probabilities.

The model package is an ordinary Idrak `.ikm` file. To list what it contains (weights, architecture, predictor
settings), run:

```sh
dotnet idrak inspect digits.ikm
```

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

**Data** ([`Mnist.cs`](Mnist.cs)): the IDX files become datasets through Idrak's data extensions:

```csharp
Dataset.FromClassLabels(images, labels, 10, Digits).WithFeatureShape(1, 28, 28);
train.Subset([.. Enumerable.Range(0, n)]);       // --train-samples
train.Batches(64, shuffle: true, seed: 1);       // a DataLoader
```

**Training**: a single `TrainingRun` object holds the model, loss, optimizer, data and metrics:

```csharp
var history = new TrainingRun
{
    Model = model, Loss = Losses.CrossEntropy, Optimizer = p => new Adam(p, 1e-3f),
    Train = train.Batches(64, shuffle: true), Validation = test.Batches(512),
    Epochs = 3, Metrics = [Metric.Accuracy],
}.Fit();
```

**Inference**: the predictor builder turns the network's outputs into a `ClassPrediction` (the digit, its probability
and every digit's score). It also saves and reloads the model as one package:

```csharp
using var predictor = Predictor.For(model).InputShape(1, 28, 28).Softmax().Classes(Digits).Build();
predictor.Save("digits.ikm");

var saved = Predictor.Load("digits.ikm");
using var loaded = saved.Classes(saved.StoredClasses!).Build();
ClassPrediction answer = loaded.Predict(pixels);
```
