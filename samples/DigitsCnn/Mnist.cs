using CnnSamples.Shared;
using Idrak.Data;

namespace DigitsCnn;

/// <summary>
/// Reads the MNIST handwritten digits that <c>dotnet idrak @mnist.rsp</c> downloaded, as Idrak datasets.
/// </summary>
internal static class Mnist
{
    public const int Rows = 28;
    public const int Columns = 28;

    public static readonly string[] Digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

    private const string DownloadHint = "Download the data first: dotnet tool restore && dotnet idrak @mnist.rsp (see README.md).";

    /// <summary>
    /// The training and test sets as one-hot classification datasets of [1, 28, 28] images in [0, 1]
    /// (a white digit on black), with the test labels for reporting.
    /// </summary>
    public static (Dataset Train, Dataset Test, int[] TestLabels) Load(string dataFolder)
    {
        var folders = DataFiles.SearchFolders(dataFolder);
        var testLabels = ReadLabels("t10k-labels-idx1-ubyte.gz", folders);
        return (
            ToDataset(ReadImages("train-images-idx3-ubyte.gz", folders), ReadLabels("train-labels-idx1-ubyte.gz", folders)),
            ToDataset(ReadImages("t10k-images-idx3-ubyte.gz", folders), testLabels),
            testLabels);
    }

    private static Dataset ToDataset(float[,] images, int[] labels) =>
        Dataset.FromClassLabels(images, labels, Digits.Length, Digits).WithFeatureShape(1, Rows, Columns);

    private static float[,] ReadImages(string file, IReadOnlyList<string> folders)
    {
        using var stream = File.OpenRead(DataFiles.Find(file, folders, DownloadHint));
        return Idx.ReadImages(stream, Rows, Columns);
    }

    private static int[] ReadLabels(string file, IReadOnlyList<string> folders)
    {
        using var stream = File.OpenRead(DataFiles.Find(file, folders, DownloadHint));
        return Idx.ReadLabels(stream);
    }
}
