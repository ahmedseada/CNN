# CnnSamples.Shared

A class library with the code that every sample in this repository uses. It is not a sample to run on its own.

| File | What it does |
|------|--------------|
| [`Idx.cs`](Idx.cs) | Reads gzip-compressed IDX files (MNIST, EMNIST) as image arrays and labels; can transpose EMNIST's images |
| [`EmnistArchive.cs`](EmnistArchive.cs) | Reads an EMNIST split and its character mapping from NIST's `gzip.zip` or extracted files |
| [`DataFiles.cs`](DataFiles.cs) | Finds files that `dotnet idrak data download` fetched: in the sample's data folder, then in Idrak's cache |
| [`ImageFiles.cs`](ImageFiles.cs) | Writes PGM images, and reads any image as light-on-dark greyscale at a given size |
| [`ConsoleReport.cs`](ConsoleReport.cs) | Prints images as text and the test results (confusion matrix, per-class accuracy, common mistakes) |
| [`TestImages.cs`](TestImages.cs) | Picks images: a seeded random training subset, and test images that cover every class in turn |
| [`SampleOptions.cs`](SampleOptions.cs) | Parses the shared command line: `train`, `predict IMAGE`, `export` and their options |
