# CnnSamples

Convolutional neural network samples for .NET 10, built with [Idrak](https://www.nuget.org/packages/Idrak) 0.3.1 and
its fluent API. Datasets are downloaded with the Idrak command-line tool (`idrak`), which the repository pins as a
local .NET tool.

## Samples

| Sample | What it does |
|--------|--------------|
| [`samples/DigitsCnn`](samples/DigitsCnn/README.md) | Trains a CNN on MNIST and recognises handwritten digits 0-9 |
| [`samples/LettersCnn`](samples/LettersCnn/README.md) | Trains a deeper CNN (batch norm, AdamW, cosine schedule) on EMNIST Letters and recognises handwritten letters A-Z |
| [`samples/DocumentOcr`](samples/DocumentOcr/README.md) | Reads the text of a page of handwritten characters: a CNN on EMNIST Balanced (digits, letters) plus a page reader that finds lines and characters |
| [`samples/MultiLanguageOcr`](samples/MultiLanguageOcr/README.md) | Reads pages of handwritten characters in English and Arabic: one CNN on EMNIST, AHCD and MADBase (85 characters), one script per line, Arabic read right to left |

Each sample has its own `README.md` with the steps to download its data and run it. Code that all the samples use
(IDX reading, console reports, model packages) lives in [`src/CnnSamples.Shared`](src/CnnSamples.Shared/README.md).

## Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0) (`global.json` asks for 10.0.100 or a later
  feature band)
- Internet access the first time, to restore the NuGet packages and download the datasets
- Optional: an NVIDIA (CUDA) or Vulkan GPU. Idrak uses one when the machine has it and the CPU otherwise.

## Getting started

```sh
git clone https://github.com/ahmedseada/CNN.git
cd CNN

# 1. Install the Idrak CLI (version 0.3.1, pinned in dotnet-tools.json) for this repository
dotnet tool restore
dotnet idrak doctor          # optional: shows the devices and backends Idrak can use here

# 2. Build every project
dotnet build CnnSamples.slnx

# 3. Follow a sample's README, e.g. samples/DigitsCnn
cd samples/DigitsCnn
dotnet idrak @mnist.rsp      # download MNIST with the Idrak CLI
dotnet run -c Release -- train
dotnet run -c Release -- export              # some test images
dotnet idrak predict digits.ikm -i test-digits --top 3   # inference with the Idrak CLI
```

To install the CLI for every folder instead of this repository only, run `dotnet tool install -g Idrak.Cli --version
0.3.1`. Then run `idrak ...` in place of `dotnet idrak ...`.

## Layout

| Path                  | Purpose                                                    |
|-----------------------|------------------------------------------------------------|
| `CnnSamples.slnx`     | The solution (XML `.slnx` format)                           |
| `Directory.Build.props` | Settings shared by every project: `net10.0`, nullable, warnings as errors |
| `dotnet-tools.json`   | Local tool manifest: `Idrak.Cli` 0.3.1                      |
| `samples/`            | Runnable sample apps, one folder and README each            |
| `src/CnnSamples.Shared` | Code every sample uses; see its README                    |
| `tests/`              | Tests (empty for now)                                       |

Downloaded datasets (`samples/*/data/`), exported test images (`samples/*/test-digits/`, `samples/*/test-letters/`),
demo pages (`page.pgm`, `page.txt`) and trained models (`*.ikm`) are git-ignored.
