# CnnSamples

Convolutional neural network samples in .NET.

## Layout

| Path       | Purpose                              |
|------------|--------------------------------------|
| `src/`     | Shared libraries (layers, tensors…)  |
| `samples/` | Runnable sample apps                 |
| `tests/`   | Unit tests                           |

## Samples

| Sample | What it does |
|--------|--------------|
| [`samples/DigitsCnn`](samples/DigitsCnn) | Trains a CNN on MNIST with Idrak and recognises handwritten digits 0-9 |

## Build

Requires the .NET 10 SDK (see `global.json`).

```sh
dotnet build CnnSamples.slnx
dotnet test CnnSamples.slnx
```
