# DigitsCnn

A console app that trains a small convolutional network with [Idrak](https://www.nuget.org/packages/Idrak) 0.2.0
to recognise handwritten digits 0-9 (MNIST), then uses it to read digit images.

## Network

```
[1 x 28 x 28] -> Conv 3x3 (32) -> ReLU -> MaxPool 2
              -> Conv 3x3 (64) -> ReLU -> MaxPool 2
              -> Flatten -> Linear 128 -> ReLU -> Dropout 0.25 -> Linear 10
```

It trains with Adam (lr 1e-3) and cross-entropy loss. Idrak's `Device.Default` uses a CUDA or Vulkan GPU when the
machine has one and falls back to the CPU otherwise.

## Train

```sh
dotnet run -c Release -- train
```

On the first run the app downloads MNIST into `data/mnist/` (about 11 MB). It then trains, prints the accuracy on the
10,000 test images with a confusion matrix, shows a few test digits as ASCII art, and saves the model to `digits.ikm`.

| Option             | Default      | Meaning                                    |
|--------------------|--------------|--------------------------------------------|
| `--epochs`         | 3            | Passes over the training set               |
| `--batch`          | 64           | Batch size                                 |
| `--train-samples`  | all (60,000) | Use only the first N training images        |
| `--model`          | `digits.ikm` | Where to save the trained model            |
| `--data`           | `data/mnist` | Where to find or download the MNIST files  |

For a quick check, `--epochs 1 --train-samples 10000` takes about 10 seconds on a 4-core CPU and reaches about 96%
test accuracy. The default 3-epoch run on all 60,000 images takes about 2 minutes on the same CPU and reaches 99.1%.

## Predict

```sh
dotnet run -c Release -- predict my-digit.png
```

PNG, BMP, PGM and PPM files are accepted. The image is converted to greyscale and resized to 28 x 28. If it is a dark
digit on a light background, the app inverts it to match MNIST's light-on-dark style. It prints the image, the
predicted digit and the top three scores.
