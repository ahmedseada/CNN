using System.Buffers;
using System.IO.Compression;
using System.Runtime.InteropServices;
using CnnSamples.Shared;
using Idrak.Data;
using Idrak.Vision;

namespace MultiLanguageOcr;

/// <summary>
/// Reads the three data sets and joins them into one: EMNIST Balanced (NIST's gzip.zip), AHCD's Arabic letters and
/// MADBase's Arabic digits (Kaggle's ahcd1.zip and ahdd1.zip, from <c>dotnet idrak @arabic.rsp</c>).
/// </summary>
internal static class Datasets
{
    public const int Size = 28;

    private const string ArabicHint = "Download the Arabic data first: dotnet idrak @arabic.rsp (needs a Kaggle API token; see README.md).";

    /// <summary>
    /// Where the downloads are looked for: this sample's data folder, Idrak's cache, then DocumentOcr's and LettersCnn's
    /// data folders (their EMNIST download works here too).
    /// </summary>
    public static string[] Folders(string dataFolder) =>
        [.. DataFiles.SearchFolders(dataFolder), Path.Combine("..", "DocumentOcr", "data"), Path.Combine("..", "LettersCnn", "data")];

    /// <summary>
    /// One data set's images as its file stores them, one byte per pixel: <paramref name="Rows"/> x <paramref name="Rows"/>
    /// each, row by row or (<paramref name="ColumnMajor"/>) column by column; <paramref name="Upright"/> says whether an
    /// image whose mean is light is inverted to white on black when it is framed.
    /// </summary>
    internal sealed record Images(byte[] Pixels, int Rows, bool ColumnMajor, bool Upright);

    /// <summary>
    /// Images (framed 28 x 28 like characters cut from a page, when read) and their labels in the joint class order. The
    /// pixels stay as the files store them, one byte each, and a part holds only which of them it uses: framing happens
    /// as an image is read, with the same float steps as framing it ahead (the values come out bit for bit the same).
    /// </summary>
    public sealed class Part
    {
        private readonly (Images Source, int[] Picked, int Start)[] _segments;

        internal Part((Images Source, int[] Picked)[] segments, int[] labels)
        {
            int start = 0;
            _segments = new (Images, int[], int)[segments.Length];
            for (int k = 0; k < segments.Length; k++)
            {
                _segments[k] = (segments[k].Source, segments[k].Picked, start);
                start += segments[k].Picked.Length;
            }
            Labels = labels;
        }

        /// <summary>The labels, in the joint class order.</summary>
        public int[] Labels { get; }

        /// <summary>The number of images.</summary>
        public int Count => Labels.Length;

        /// <summary>Image <paramref name="index"/> framed into <paramref name="destination"/> (28 x 28 values in [0, 1]).</summary>
        public void Frame(int index, Span<float> destination)
        {
            int k = _segments.Length - 1;
            while (_segments[k].Start > index)
                k--;
            var (source, picked, start) = _segments[k];
            int rows = source.Rows, pixels = rows * rows;
            var raw = source.Pixels.AsSpan(picked[index - start] * pixels, pixels);
            var image = ArrayPool<float>.Shared.Rent(pixels);
            try
            {
                // Upright, ink high: as the file has it divided by 255 (transposed from column order), and inverted when
                // mostly light; the same operations, in the same order, as before framing ahead of time.
                float sum = 0;
                for (int y = 0; y < rows; y++)
                    for (int x = 0; x < rows; x++)
                        sum += image[y * rows + x] = raw[source.ColumnMajor ? x * rows + y : y * rows + x] / 255f;
                if (source.Upright && sum / pixels > 0.5f)
                    for (int i = 0; i < pixels; i++)
                        image[i] = 1f - image[i];
                ContentFrame.Fit(image.AsSpan(0, pixels), rows, rows, destination, Size);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(image);
            }
        }

        /// <summary>Image <paramref name="index"/>, framed, in a new array (for printing and the demo's pages).</summary>
        public float[] Image(int index)
        {
            var framed = new float[Size * Size];
            Frame(index, framed);
            return framed;
        }

        internal (Images Source, int[] Picked)[] Segments => [.. _segments.Select(s => (s.Source, s.Picked))];
    }

    /// <summary>
    /// The joint training and test sets, at most <paramref name="perSource"/> images from each data set (random,
    /// seeded); by default 60,000 EMNIST, all 13,440 AHCD letters twice (it is the smallest set) and 30,000 MADBase digits.
    /// Without <paramref name="training"/>, only the test sets are read (Train is empty).
    /// </summary>
    public static (Part Train, Part Test, CharacterClass[] Classes) Load(string dataFolder, int? perSource, bool training = true)
    {
        var folders = Folders(dataFolder);
        var mapping = EmnistArchive.ReadMapping("balanced", folders);
        var classes = Characters.Classes([.. Enumerable.Range(0, mapping.Count).Select(i => mapping[i].ToString())]);
        int arabicLetters = mapping.Count, arabicDigits = mapping.Count + Characters.ArabicLetters.Length;

        Console.WriteLine("Reading EMNIST Balanced, AHCD (Arabic letters) and MADBase (Arabic digits)...");
        var none = new Part([], []);
        var emnistTrain = training ? Emnist("train", folders, perSource ?? 60_000) : none;
        var emnistTest = Emnist("test", folders, perSource ?? 10_000);
        // AHCD stores each image column by column; MADBase stores it row by row (an upright ٢ came out on its side when transposed).
        var lettersTrain = training ? Kaggle("ahcd1", "train", 32, folders, perSource, offset: arabicLetters - 1, columnMajor: true) : none;   // labels 1-28
        var lettersTest = Kaggle("ahcd1", "test", 32, folders, perSource, offset: arabicLetters - 1, columnMajor: true);
        var digitsTrain = training ? Kaggle("ahdd1", "train", 28, folders, perSource ?? 30_000, offset: arabicDigits, columnMajor: false) : none;   // labels 0-9
        var digitsTest = Kaggle("ahdd1", "test", 28, folders, perSource ?? 5_000, offset: arabicDigits, columnMajor: false);
        Console.WriteLine($"  EMNIST {emnistTrain.Count:N0} + {emnistTest.Count:N0}, AHCD {lettersTrain.Count:N0} + {lettersTest.Count:N0}, " +
            $"MADBase {digitsTrain.Count:N0} + {digitsTest.Count:N0} (training + test)");

        return (Join(emnistTrain, lettersTrain, lettersTrain, digitsTrain), Join(emnistTest, lettersTest, digitsTest), classes);
    }

    /// <summary>
    /// A part as a [1, 28, 28] classification source over <paramref name="classes"/> for a <see cref="DataLoader"/>:
    /// each image framed as it is read, so a training set takes a byte per stored pixel instead of 28 x 28 floats per image.
    /// </summary>
    public static ISampleSource ToSource(Part part, CharacterClass[] classes) => new PartSource(part, classes.Length);

    /// <summary>A part as a [1, 28, 28] classification dataset over <paramref name="classes"/>, every image framed in memory.</summary>
    public static Dataset ToDataset(Part part, CharacterClass[] classes)
    {
        var features = new float[part.Count, Size * Size];
        var flat = MemoryMarshal.CreateSpan(ref features[0, 0], features.Length);
        for (int i = 0; i < part.Count; i++)
            part.Frame(i, flat.Slice(i * Size * Size, Size * Size));
        return Dataset.FromClassLabels(features, part.Labels, classes.Length, [.. classes.Select(c => c.Text)])
            .WithFeatureShape(1, Size, Size);
    }

    // Framed images and one-hot targets, read one at a time.
    private sealed class PartSource(Part part, int classes) : ISampleSource
    {
        public int Count => part.Count;

        public IReadOnlyList<int> FeatureShape { get; } = [1, Size, Size];

        public IReadOnlyList<int> TargetShape { get; } = [classes];

        public void Read(int index, Span<float> features, Span<float> targets)
        {
            part.Frame(index, features);
            targets.Clear();
            targets[part.Labels[index]] = 1f;
        }
    }

    private static Part Join(params Part[] parts) =>
        new([.. parts.SelectMany(p => p.Segments)], [.. parts.SelectMany(p => p.Labels)]);

    private static Part Emnist(string split, IReadOnlyList<string> folders, int count)
    {
        var (images, labels) = EmnistArchive.ReadSplitBytes("balanced", split, folders);
        var picked = TestImages.RandomSubset(labels.Length, Math.Min(count, labels.Length));
        return new([(new Images(images, Size, ColumnMajor: false, Upright: false), picked)], [.. picked.Select(i => labels[i])]);
    }

    // A Kaggle archive's CSV pair: images one per line (rows x rows values, 0-255) and labels.
    private static Part Kaggle(string archive, string split, int rows, IReadOnlyList<string> folders, int? count, int offset, bool columnMajor)
    {
        using var zip = ZipFile.OpenRead(DataFiles.Find($"{archive}.zip", folders, ArabicHint));
        ZipArchiveEntry Entry(string kind) => zip.Entries.FirstOrDefault(e =>
                e.Name.EndsWith(".csv", StringComparison.OrdinalIgnoreCase) && e.Name.Contains(split, StringComparison.OrdinalIgnoreCase)
                && e.Name.Contains(kind, StringComparison.OrdinalIgnoreCase))
            ?? throw new InvalidDataException($"{archive}.zip has no {split} {kind} CSV (it has {string.Join(", ", zip.Entries.Select(e => e.Name))}).");

        var labelTable = ReadByteCsv(Entry("label"));
        var images = ReadByteCsv(Entry("image"));
        if (images.Rows != labelTable.Rows)
            throw new InvalidDataException($"{archive}.zip: {images.Rows} {split} images but {labelTable.Rows} labels.");
        if (images.Columns != rows * rows)
            throw new InvalidDataException($"{archive}.zip: its {split} images have {images.Columns} values, expected {rows * rows}.");

        var picked = TestImages.RandomSubset(labelTable.Rows, Math.Min(count ?? labelTable.Rows, labelTable.Rows));
        return new([(new Images(images.Values, rows, columnMajor, Upright: true), picked)], [.. picked.Select(i => labelTable.Row(i)[0] + offset)]);
    }

    /// <summary>A CSV file of small whole numbers (0-255: pixels, labels), one byte each, row after row.</summary>
    internal sealed record ByteTable(byte[] Values, int Rows, int Columns)
    {
        public ReadOnlySpan<byte> Row(int row) => Values.AsSpan(row * Columns, Columns);
    }

    /// <summary>
    /// Reads a CSV of whole numbers from 0 to 255 straight from its UTF-8 bytes, in blocks: no line, cell or number
    /// becomes a string, and the table is one byte per value (a 60,000 x 784 file is 47 MB, not 376 MB of doubles).
    /// A line holding anything but numbers (a header) is skipped; a value such as 255.0 reads as 255.
    /// </summary>
    internal static ByteTable ReadByteCsv(ZipArchiveEntry entry) => ReadByteCsv(entry.Open());

    /// <inheritdoc cref="ReadByteCsv(ZipArchiveEntry)"/>
    internal static ByteTable ReadByteCsv(Stream input)
    {
        using var stream = input;
        var values = new byte[1 << 20];
        int count = 0, rows = 0, columns = -1;
        int lineStart = 0, cells = 0, value = 0;
        bool digits = false, fraction = false, skip = false;
        var block = new byte[1 << 20];

        void EndCell()
        {
            if (!digits)
            {
                skip = true;                                                      // an empty cell: not a row of numbers
                return;
            }
            if (count == values.Length)
                Array.Resize(ref values, values.Length * 2);
            values[count++] = (byte)value;
            cells++;
            (value, digits, fraction) = (0, false, false);
        }

        void EndLine()
        {
            if (digits)
                EndCell();                                                        // a line's last value (a trailing comma has none)
            if (skip || cells == 0)
                count = lineStart;                                                // a header or a blank line
            else
            {
                if (columns < 0)
                    columns = cells;
                else if (cells != columns)
                    throw new InvalidDataException($"Row {rows + 1} has {cells} values, the rows before it {columns}.");
                rows++;
            }
            (lineStart, cells, value, digits, fraction, skip) = (count, 0, 0, false, false, false);
        }

        int read;
        while ((read = stream.Read(block, 0, block.Length)) > 0)
        {
            foreach (byte b in block.AsSpan(0, read))
            {
                if (skip && b != (byte)'\n')
                    continue;
                switch (b)
                {
                    case >= (byte)'0' and <= (byte)'9' when fraction:
                        if (b != (byte)'0')
                            skip = true;                                          // 12.5: not a whole number
                        break;
                    case >= (byte)'0' and <= (byte)'9':
                        value = value * 10 + (b - '0');
                        digits = true;
                        if (value > 255)
                            throw new InvalidDataException($"Row {rows + 1} holds a value past 255; this reader takes pixels and labels.");
                        break;
                    case (byte)'.' when digits && !fraction:
                        fraction = true;
                        break;
                    case (byte)',':
                        EndCell();
                        break;
                    case (byte)'\n':
                        EndLine();
                        break;
                    case (byte)'\r' or (byte)' ' or (byte)'\t':
                        break;
                    case 0xEF or 0xBB or 0xBF when count == lineStart && cells == 0 && !digits:
                        break;                                                    // a UTF-8 byte order mark
                    default:
                        skip = true;                                              // a header's text
                        break;
                }
            }
        }
        EndLine();
        if (columns < 0)
            throw new InvalidDataException("The CSV has no rows of numbers.");
        return new ByteTable(values, rows, columns);
    }
}
