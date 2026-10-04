# CnnSamples

Convolutional neural network samples for .NET 10, built with [Idrak](https://www.nuget.org/packages/Idrak) 0.2.0 and
its fluent API. Datasets are downloaded with the Idrak command-line tool (`idrak`), which the repository pins as a
local .NET tool.

## Samples

| Sample | What it does |
|--------|--------------|
| [`samples/DigitsCnn`](samples/DigitsCnn/README.md) | Trains a CNN on MNIST and recognises handwritten digits 0-9 |

Each sample has its own `README.md` with the steps to download its data and run it.

## Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0) (`global.json` asks for 10.0.100 or a later
  feature band)
- Internet access the first time, to restore the NuGet packages and download the datasets
- Optional: an NVIDIA (CUDA) or Vulkan GPU. Idrak uses one when the machine has it and the CPU otherwise.

## Getting started

```sh
git clone https://github.com/ahmedseada/CNN.git
cd CNN

# 1. Install the Idrak CLI (version 0.2.0, pinned in dotnet-tools.json) for this repository
dotnet tool restore
dotnet idrak doctor          # optional: shows the devices and backends Idrak can use here

# 2. Build every project
dotnet build CnnSamples.slnx

# 3. Follow a sample's README, e.g. samples/DigitsCnn
cd samples/DigitsCnn
dotnet idrak @mnist.rsp      # download MNIST with the Idrak CLI
dotnet run -c Release -- train
```

To install the CLI for every folder instead of this repository only, run `dotnet tool install -g Idrak.Cli --version
0.2.0`. Then run `idrak ...` in place of `dotnet idrak ...`.

## Layout

| Path                  | Purpose                                                    |
|-----------------------|------------------------------------------------------------|
| `CnnSamples.slnx`     | The solution (XML `.slnx` format)                           |
| `Directory.Build.props` | Settings shared by every project: `net10.0`, nullable, warnings as errors |
| `dotnet-tools.json`   | Local tool manifest: `Idrak.Cli` 0.2.0                      |
| `samples/`            | Runnable sample apps, one folder and README each            |
| `src/`                | Shared libraries (empty for now)                            |
| `tests/`              | Tests (empty for now)                                       |

Downloaded datasets (`samples/*/data/`) and trained models (`*.ikm`) are git-ignored.
