using System.Buffers.Binary;
using System.IO.Compression;

namespace DigitsCnn;

/// <summary>Downloads the MNIST handwritten digits (once, into a cache folder) and reads its IDX files.</summary>
internal static class Mnist
{
    public const int Rows = 28;
    public const int Columns = 28;
    public const int Pixels = Rows * Columns;

    private static readonly string[] Mirrors =
    [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
    ];

    /// <summary>Images as [count, 784] values in [0, 1] (white digit on black) and their labels 0-9.</summary>
    public sealed record Split(float[,] Images, int[] Labels)
    {
        public int Count => Labels.Length;
    }

    public static async Task<(Split Train, Split Test)> LoadAsync(string folder, CancellationToken ct = default)
    {
        Directory.CreateDirectory(folder);
        var train = new Split(
            ReadImages(await FetchAsync(folder, "train-images-idx3-ubyte.gz", ct)),
            ReadLabels(await FetchAsync(folder, "train-labels-idx1-ubyte.gz", ct)));
        var test = new Split(
            ReadImages(await FetchAsync(folder, "t10k-images-idx3-ubyte.gz", ct)),
            ReadLabels(await FetchAsync(folder, "t10k-labels-idx1-ubyte.gz", ct)));
        return (train, test);
    }

    private static async Task<string> FetchAsync(string folder, string file, CancellationToken ct)
    {
        var path = Path.Combine(folder, file);
        if (File.Exists(path))
            return path;

        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(5) };
        foreach (var mirror in Mirrors)
        {
            try
            {
                Console.WriteLine($"Downloading {mirror}{file}");
                var bytes = await http.GetByteArrayAsync(mirror + file, ct);
                var partial = path + ".part";
                await File.WriteAllBytesAsync(partial, bytes, ct);
                File.Move(partial, path, overwrite: true);
                return path;
            }
            catch (HttpRequestException e)
            {
                Console.WriteLine($"  failed: {e.Message}");
            }
        }

        throw new IOException($"Could not download {file}. Place the MNIST .gz files in '{folder}' and run again.");
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
        var data = Decompress(path);
        var span = data.AsSpan();
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
        var data = Decompress(path);
        var span = data.AsSpan();
        if (BinaryPrimitives.ReadInt32BigEndian(span) != 2049)
            throw new InvalidDataException($"{path} is not an IDX label file.");

        int count = BinaryPrimitives.ReadInt32BigEndian(span[4..]);
        var labels = new int[count];
        for (int i = 0; i < count; i++)
            labels[i] = span[8 + i];
        return labels;
    }
}
