# CnnSamples.Shared

A class library with the code that every sample in this repository uses. It is not a sample to run on its own.

| File | What it does |
|------|--------------|
| [`Idx.cs`](Idx.cs) | Reads gzip-compressed IDX files (MNIST, EMNIST) as image arrays and labels; can transpose EMNIST's images |
| [`EmnistArchive.cs`](EmnistArchive.cs) | Reads an EMNIST split and its character mapping from NIST's `gzip.zip` or extracted files |
| [`DataFiles.cs`](DataFiles.cs) | Finds files that `dotnet idrak data download` fetched: in the sample's data folder, then in Idrak's cache |
| [`BestWeights.cs`](BestWeights.cs) | Keeps the best epoch's weights during training and restores them afterwards |
| [`ModelFiles.cs`](ModelFiles.cs) | Saves a predictor as a package that both `Predictor.Load` and `idrak predict` can run; writes and reads images |
| [`ConsoleReport.cs`](ConsoleReport.cs) | Prints images as text and the test results (confusion matrix, per-class accuracy, common mistakes) |
| [`TestImages.cs`](TestImages.cs) | Picks images: a seeded random training subset, and test images that cover every class in turn |
| [`SampleOptions.cs`](SampleOptions.cs) | Parses the shared command line: `train`, `predict IMAGE`, `export` and their options |

## Workarounds for Idrak 0.2.0

Two of these classes work around problems in Idrak 0.2.0 that are fixed for 0.2.1
([ahmedseada/Idrak#1](https://github.com/ahmedseada/Idrak/pull/1)):

- `BestWeights`: 0.2.0's `RestoreBestWeights` restores the best epoch only when early stopping ends the run, not when
  training reaches its last epoch.
- `ModelFiles.Save`: 0.2.0's `idrak predict` reads only the `training` entry that `idrak train` writes, not the
  `predictor` entry that `Predictor.Save` writes. Without the extra entry it treats a classifier as a regression model.

Once the samples move to 0.2.1, `BestWeights` can be replaced by `EarlyStoppingPatience` alone, and `ModelFiles.Save` by
`predictor.Save`.
