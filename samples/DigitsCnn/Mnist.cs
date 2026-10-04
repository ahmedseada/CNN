using System.Buffers.Binary;
using System.IO.Compression;
using Idrak.Data;

namespace DigitsCnn;

/// <summary>
/// Reads the MNIST handwritten digits that <c>dotnet idrak @mnist.rsp</c> downloaded, as Idrak datasets.
/// </summary>
internal static class Mnist
{
    public const int Rows = 28;
    public const int Columns = 28;
    public const int Pixels = Rows * Columns;

    public static readonly string[] Digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

    private const string TrainImages = "train-images-idx3-ubyte.gz";
    private const string TrainLabels = "train-labels-idx1-ubyte.gz";
    private const string TestImages = "t10k-images-idx3-ubyte.gz";
    private const string TestLabels = "t10k-labels-idx1-ubyte.gz";

    /// <summary>
    /// The training and test sets as one-hot classification datasets of [1, 28, 28] images in [0, 1]
    /// (a white digit on black), with each set's labels for reporting.
    /// </summary>
    public static (Dataset Train, Dataset Test, int[] TestLabels) Load(string dataFolder)
    {
        var folders = SearchFolders(dataFolder).ToArray();
        return (
            ToDataset(Find(TrainImages, folders), Find(TrainLabels, folders)),
            ToDataset(Find(TestImages, folders), Find(TestLabels, folders)),
            ReadLabels(Find(TestLabels, folders)));
    }

    private static Dataset ToDataset(string images, string labels) =>
        Dataset.FromClassLabels(ReadImages(images), ReadLabels(labels), Digits.Length, Digits)
            .WithFeatureShape(1, Rows, Columns);

    // The sample's data folder first, then Idrak's own cache (what `idrak data download` uses without --cache).
    private static IEnumerable<string> SearchFolders(string dataFolder)
    {
        yield return dataFolder;
        var cache = Environment.GetEnvironmentVariable("IDRAK_CACHE");
        yield return string.IsNullOrEmpty(cache)
            ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "idrak")
            : cache;
    }

    private static string Find(string file, string[] folders)
    {
        foreach (var folder in folders.Where(Directory.Exists))
        {
            var match = Directory.EnumerateFiles(folder, file, SearchOption.AllDirectories).FirstOrDefault();
            if (match is not null)
                return match;
        }

        throw new FileNotFoundException(
            $"MNIST file '{file}' not found under {string.Join(" or ", folders.Select(f => $"'{f}'"))}. " +
            "Download the data first: dotnet tool restore && dotnet idrak @mnist.rsp (see README.md).");
    }

    private static byte[] Decompress(string path)
    {
        using var gzip = new GZipStream(File.OpenRead(path), CompressionMode.Decompress);
        using var memory = new MemoryStream();
        gzip.CopyTo(memory);
        return memory.ToArray();
    }

    private static float[,] ReadImages(string path)
    {
        var span = Decompress(path).AsSpan();
        if (BinaryPrimitives.ReadInt32BigEndian(span) != 2051)
            throw new InvalidDataException($"{path} is not an IDX image file.");

        int count = BinaryPrimitives.ReadInt32BigEndian(span[4..]);
        int rows = BinaryPrimitives.ReadInt32BigEndian(span[8..]);
        int columns = BinaryPrimitives.ReadInt32BigEndian(span[12..]);
        if (rows != Rows || columns != Columns)
            throw new InvalidDataException($"{path} has {rows}x{columns} images, expected {Rows}x{Columns}.");

        var images = new float[count, Pixels];
        var pixels = span[16..];
        for (int i = 0; i < count; i++)
            for (int p = 0; p < Pixels; p++)
                images[i, p] = pixels[i * Pixels + p] / 255f;
        return images;
    }

    private static int[] ReadLabels(string path)
    {
        var span = Decompress(path).AsSpan();
        if (BinaryPrimitives.ReadInt32BigEndian(span) != 2049)
            throw new InvalidDataException($"{path} is not an IDX label file.");

        int count = BinaryPrimitives.ReadInt32BigEndian(span[4..]);
        var labels = new int[count];
        for (int i = 0; i < count; i++)
            labels[i] = span[8 + i];
        return labels;
    }
}
