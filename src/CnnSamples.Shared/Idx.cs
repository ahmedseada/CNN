using System.Buffers.Binary;
using System.IO.Compression;

namespace CnnSamples.Shared;

/// <summary>Reads the IDX files of MNIST and EMNIST: gzip-compressed, big-endian headers, one byte per pixel or label.</summary>
public static class Idx
{
    /// <summary>
    /// Images as [count, rows * columns] values in [0, 1]. <paramref name="transpose"/> flips rows and columns (EMNIST
    /// stores its images transposed relative to MNIST).
    /// </summary>
    public static float[,] ReadImages(Stream gzip, int rows, int columns, bool transpose = false)
    {
        var span = Decompress(gzip).AsSpan();
        if (BinaryPrimitives.ReadInt32BigEndian(span) != 2051)
            throw new InvalidDataException("Not an IDX image file (magic number 2051).");

        int count = BinaryPrimitives.ReadInt32BigEndian(span[4..]);
        int fileRows = BinaryPrimitives.ReadInt32BigEndian(span[8..]);
        int fileColumns = BinaryPrimitives.ReadInt32BigEndian(span[12..]);
        if (fileRows != rows || fileColumns != columns)
            throw new InvalidDataException($"The file has {fileRows}x{fileColumns} images, expected {rows}x{columns}.");

        int pixels = rows * columns;
        var images = new float[count, pixels];
        var data = span[16..];
        for (int i = 0; i < count; i++)
        {
            var image = data.Slice(i * pixels, pixels);
            for (int y = 0; y < rows; y++)
                for (int x = 0; x < columns; x++)
                    images[i, y * columns + x] = image[transpose ? x * rows + y : y * columns + x] / 255f;
        }
        return images;
    }

    /// <summary>Labels, one byte each.</summary>
    public static int[] ReadLabels(Stream gzip)
    {
        var span = Decompress(gzip).AsSpan();
        if (BinaryPrimitives.ReadInt32BigEndian(span) != 2049)
            throw new InvalidDataException("Not an IDX label file (magic number 2049).");

        int count = BinaryPrimitives.ReadInt32BigEndian(span[4..]);
        var labels = new int[count];
        for (int i = 0; i < count; i++)
            labels[i] = span[8 + i];
        return labels;
    }

    private static byte[] Decompress(Stream compressed)
    {
        using var gzip = new GZipStream(compressed, CompressionMode.Decompress, leaveOpen: true);
        using var memory = new MemoryStream();
        gzip.CopyTo(memory);
        return memory.ToArray();
    }
}
