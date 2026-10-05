using System.IO.Compression;

namespace CnnSamples.Shared;

/// <summary>
/// Reads EMNIST splits from NIST's archive (<c>gzip.zip</c>, as <c>dotnet idrak data download</c> saves it) or from
/// files someone extracted from it.
/// </summary>
public static class EmnistArchive
{
    public const int Rows = 28;
    public const int Columns = 28;

    /// <summary>What the samples print when the data is missing.</summary>
    public const string DownloadHint = "Download EMNIST first: dotnet tool restore && dotnet idrak @emnist.rsp (see README.md).";

    private const string Archive = "gzip.zip";

    /// <summary>
    /// The images (upright: EMNIST stores them transposed) and labels of one split, e.g. ("letters", "train") or
    /// ("balanced", "test").
    /// </summary>
    public static (float[,] Images, int[] Labels) ReadSplit(string name, string split, IReadOnlyList<string> folders)
    {
        using var images = Open($"emnist-{name}-{split}-images-idx3-ubyte.gz", folders);
        using var labels = Open($"emnist-{name}-{split}-labels-idx1-ubyte.gz", folders);
        return (Idx.ReadImages(images.Stream, Rows, Columns, transpose: true), Idx.ReadLabels(labels.Stream));
    }

    /// <summary>
    /// The images of one split as bytes, one per pixel, image after image (upright), and their labels: a quarter of the
    /// memory of <see cref="ReadSplit"/>.
    /// </summary>
    public static (byte[] Images, int[] Labels) ReadSplitBytes(string name, string split, IReadOnlyList<string> folders)
    {
        using var images = Open($"emnist-{name}-{split}-images-idx3-ubyte.gz", folders);
        using var labels = Open($"emnist-{name}-{split}-labels-idx1-ubyte.gz", folders);
        return (Idx.ReadImageBytes(images.Stream, Rows, Columns, transpose: true).Pixels, Idx.ReadLabels(labels.Stream));
    }

    /// <summary>The characters of a split's labels, from its mapping file (lines of "label ascii-code").</summary>
    public static Dictionary<int, char> ReadMapping(string name, IReadOnlyList<string> folders)
    {
        using var file = Open($"emnist-{name}-mapping.txt", folders);
        using var reader = new StreamReader(file.Stream);
        var mapping = new Dictionary<int, char>();
        while (reader.ReadLine() is { } line)
        {
            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length >= 2 && int.TryParse(parts[0], out int label) && int.TryParse(parts[1], out int code))
                mapping[label] = (char)code;
        }
        return mapping;
    }

    // A file extracted from the archive when there is one, else the archive's entry.
    private static Entry Open(string file, IReadOnlyList<string> folders)
    {
        if (DataFiles.TryFind(file, folders) is { } path)
            return new Entry(File.OpenRead(path), null);

        var zip = ZipFile.OpenRead(DataFiles.Find(Archive, folders, DownloadHint));
        var entry = zip.Entries.FirstOrDefault(e => e.Name == file);
        if (entry is null)
        {
            zip.Dispose();
            throw new InvalidDataException($"{Archive} has no {file}; is it NIST's EMNIST download?");
        }
        return new Entry(entry.Open(), zip);
    }

    private sealed class Entry(Stream stream, ZipArchive? zip) : IDisposable
    {
        public Stream Stream { get; } = stream;

        public void Dispose()
        {
            Stream.Dispose();
            zip?.Dispose();
        }
    }
}
