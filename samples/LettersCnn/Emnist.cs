using System.IO.Compression;
using CnnSamples.Shared;
using Idrak.Data;

namespace LettersCnn;

/// <summary>
/// Reads the EMNIST Letters split (handwritten A-Z, upper and lower case together) that
/// <c>dotnet idrak @emnist.rsp</c> downloaded, as Idrak datasets.
/// </summary>
internal static class Emnist
{
    public const int Rows = 28;
    public const int Columns = 28;

    /// <summary>"A" to "Z". EMNIST Letters labels them 1 to 26 and puts both cases of a letter in one class.</summary>
    public static readonly string[] Letters = [.. Enumerable.Range('A', 26).Select(c => ((char)c).ToString())];

    private const string Archive = "gzip.zip";
    private const string DownloadHint =
        "Download EMNIST first: dotnet tool restore && dotnet idrak @emnist.rsp (see README.md).";

    /// <summary>
    /// The training and test sets as one-hot classification datasets of [1, 28, 28] images in [0, 1] (a white letter on
    /// black, the right way up), with the test labels (0 = A) for reporting.
    /// </summary>
    public static (Dataset Train, Dataset Test, int[] TestLabels) Load(string dataFolder)
    {
        var folders = DataFiles.SearchFolders(dataFolder);
        var (trainImages, trainLabels) = ReadSplit("train", folders);
        var (testImages, testLabels) = ReadSplit("test", folders);
        return (ToDataset(trainImages, trainLabels), ToDataset(testImages, testLabels), testLabels);
    }

    private static Dataset ToDataset(float[,] images, int[] labels) =>
        Dataset.FromClassLabels(images, labels, Letters.Length, Letters).WithFeatureShape(1, Rows, Columns);

    // The split's two files, from the files themselves when someone extracted them, else from NIST's gzip.zip.
    private static (float[,] Images, int[] Labels) ReadSplit(string split, IReadOnlyList<string> folders)
    {
        string images = $"emnist-letters-{split}-images-idx3-ubyte.gz", labels = $"emnist-letters-{split}-labels-idx1-ubyte.gz";
        if (DataFiles.TryFind(images, folders) is { } imagePath && DataFiles.TryFind(labels, folders) is { } labelPath)
        {
            using var imageFile = File.OpenRead(imagePath);
            using var labelFile = File.OpenRead(labelPath);
            return Read(imageFile, labelFile);
        }

        using var zip = ZipFile.OpenRead(DataFiles.Find(Archive, folders, DownloadHint));
        ZipArchiveEntry Entry(string name) => zip.Entries.FirstOrDefault(e => e.Name == name)
            ?? throw new InvalidDataException($"{Archive} has no {name}; is it NIST's EMNIST download?");
        using var imageEntry = Entry(images).Open();
        using var labelEntry = Entry(labels).Open();
        return Read(imageEntry, labelEntry);
    }

    private static (float[,] Images, int[] Labels) Read(Stream images, Stream labels)
    {
        // EMNIST stores each image transposed relative to MNIST: transpose it back so letters are upright.
        var pixels = Idx.ReadImages(images, Rows, Columns, transpose: true);
        var letters = Idx.ReadLabels(labels);
        for (int i = 0; i < letters.Length; i++)
        {
            letters[i]--;   // 1-26 -> 0-25
            if ((uint)letters[i] >= 26)
                throw new InvalidDataException($"Label {letters[i] + 1} is not a letter (EMNIST Letters uses 1-26).");
        }
        return (pixels, letters);
    }
}
