using System.Buffers;
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
        var (pixels, count) = ReadImageBytes(gzip, rows, columns, transpose);
        int size = rows * columns;
        var images = new float[count, size];
        for (int i = 0; i < count; i++)
            for (int p = 0; p < size; p++)
                images[i, p] = pixels[i * size + p] / 255f;
        return images;
    }

    /// <summary>
    /// Images as their bytes, one per pixel, image after image (upright: <paramref name="transpose"/> flips rows and
    /// columns as the file is read): a quarter of the memory of <see cref="ReadImages"/>, read straight from the stream
    /// into one array of the exact size.
    /// </summary>
    public static (byte[] Pixels, int Count) ReadImageBytes(Stream gzip, int rows, int columns, bool transpose = false)
    {
        using var stream = new GZipStream(gzip, CompressionMode.Decompress, leaveOpen: true);
        Span<byte> header = stackalloc byte[16];
        stream.ReadExactly(header);
        if (BinaryPrimitives.ReadInt32BigEndian(header) != 2051)
            throw new InvalidDataException("Not an IDX image file (magic number 2051).");

        int count = BinaryPrimitives.ReadInt32BigEndian(header[4..]);
        int fileRows = BinaryPrimitives.ReadInt32BigEndian(header[8..]);
        int fileColumns = BinaryPrimitives.ReadInt32BigEndian(header[12..]);
        if (fileRows != rows || fileColumns != columns)
            throw new InvalidDataException($"The file has {fileRows}x{fileColumns} images, expected {rows}x{columns}.");

        int size = rows * columns;
        var pixels = new byte[(long)count * size];
        stream.ReadExactly(pixels);
        if (transpose)
        {
            var stored = ArrayPool<byte>.Shared.Rent(size);
            try
            {
                for (int i = 0; i < count; i++)
                {
                    var image = pixels.AsSpan(i * size, size);
                    image.CopyTo(stored);
                    for (int y = 0; y < rows; y++)
                        for (int x = 0; x < columns; x++)
                            image[y * columns + x] = stored[x * rows + y];
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(stored);
            }
        }
        return (pixels, count);
    }

    /// <summary>Labels, one byte each.</summary>
    public static int[] ReadLabels(Stream gzip)
    {
        using var stream = new GZipStream(gzip, CompressionMode.Decompress, leaveOpen: true);
        Span<byte> header = stackalloc byte[8];
        stream.ReadExactly(header);
        if (BinaryPrimitives.ReadInt32BigEndian(header) != 2049)
            throw new InvalidDataException("Not an IDX label file (magic number 2049).");

        var bytes = new byte[BinaryPrimitives.ReadInt32BigEndian(header[4..])];
        stream.ReadExactly(bytes);
        var labels = new int[bytes.Length];
        for (int i = 0; i < bytes.Length; i++)
            labels[i] = bytes[i];
        return labels;
    }
}
