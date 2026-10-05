using System.Globalization;
using System.IO.Compression;
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

    /// <summary>Images (28 x 28, framed like characters cut from a page) and their labels in the joint class order.</summary>
    public sealed record Part(float[][] Images, int[] Labels);

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
        Console.WriteLine($"  EMNIST {emnistTrain.Labels.Length:N0} + {emnistTest.Labels.Length:N0}, AHCD {lettersTrain.Labels.Length:N0} + {lettersTest.Labels.Length:N0}, " +
            $"MADBase {digitsTrain.Labels.Length:N0} + {digitsTest.Labels.Length:N0} (training + test)");

        return (Join(emnistTrain, lettersTrain, lettersTrain, digitsTrain), Join(emnistTest, lettersTest, digitsTest), classes);
    }

    /// <summary>A part as a [1, 28, 28] classification dataset over <paramref name="classes"/>.</summary>
    public static Dataset ToDataset(Part part, CharacterClass[] classes)
    {
        var features = new float[part.Labels.Length, Size * Size];
        for (int i = 0; i < part.Labels.Length; i++)
            for (int p = 0; p < Size * Size; p++)
                features[i, p] = part.Images[i][p];
        return Dataset.FromClassLabels(features, part.Labels, classes.Length, [.. classes.Select(c => c.Text)])
            .WithFeatureShape(1, Size, Size);
    }

    private static Part Join(params Part[] parts) =>
        new([.. parts.SelectMany(p => p.Images)], [.. parts.SelectMany(p => p.Labels)]);

    private static Part Emnist(string split, IReadOnlyList<string> folders, int count)
    {
        var (images, labels) = EmnistArchive.ReadSplit("balanced", split, folders);
        var picked = TestImages.RandomSubset(labels.Length, Math.Min(count, labels.Length));
        return new([.. picked.Select(i => Frame(Row(images, i), Size))], [.. picked.Select(i => labels[i])]);
    }

    // A Kaggle archive's CSV pair: images one per line (rows x rows values, 0-255) and labels.
    private static Part Kaggle(string archive, string split, int rows, IReadOnlyList<string> folders, int? count, int offset, bool columnMajor)
    {
        using var zip = ZipFile.OpenRead(DataFiles.Find($"{archive}.zip", folders, ArabicHint));
        ZipArchiveEntry Entry(string kind) => zip.Entries.FirstOrDefault(e =>
                e.Name.EndsWith(".csv", StringComparison.OrdinalIgnoreCase) && e.Name.Contains(split, StringComparison.OrdinalIgnoreCase)
                && e.Name.Contains(kind, StringComparison.OrdinalIgnoreCase))
            ?? throw new InvalidDataException($"{archive}.zip has no {split} {kind} CSV (it has {string.Join(", ", zip.Entries.Select(e => e.Name))}).");

        var labels = ReadCsv(Entry("label")).Select(v => (int)v[0] + offset).ToArray();
        var images = ReadCsv(Entry("image")).ToArray();
        if (images.Length != labels.Length)
            throw new InvalidDataException($"{archive}.zip: {images.Length} {split} images but {labels.Length} labels.");

        var picked = TestImages.RandomSubset(labels.Length, Math.Min(count ?? labels.Length, labels.Length));
        return new([.. picked.Select(i => Frame(Upright(images[i], rows, columnMajor), rows))], [.. picked.Select(i => labels[i])]);
    }

    // The image upright (a column-by-column image is transposed), with ink high (white on black) whichever way the file has it.
    private static float[] Upright(double[] values, int rows, bool columnMajor)
    {
        if (values.Length != rows * rows)
            throw new InvalidDataException($"An image has {values.Length} values, expected {rows * rows}.");
        var image = new float[rows * rows];
        for (int y = 0; y < rows; y++)
            for (int x = 0; x < rows; x++)
                image[y * rows + x] = (float)(values[columnMajor ? x * rows + y : y * rows + x] / 255.0);
        return image.Average() > 0.5f ? [.. image.Select(v => 1f - v)] : image;
    }

    private static IEnumerable<double[]> ReadCsv(ZipArchiveEntry entry)
    {
        using var reader = new StreamReader(entry.Open());
        while (reader.ReadLine() is { } line)
        {
            var cells = line.Split(',');
            var values = new double[cells.Length];
            bool numeric = true;
            for (int i = 0; i < cells.Length && numeric; i++)
                numeric = double.TryParse(cells[i], NumberStyles.Float, CultureInfo.InvariantCulture, out values[i]);
            if (numeric && cells.Length > 0)
                yield return values;   // a header line, if any, is skipped
        }
    }

    // A data set's image framed by Idrak's ContentFrame the way the RegionClassifier frames a character cut from a page:
    // cropped to its ink, centred and scaled to 28 x 28. Characters from EMNIST, AHCD's 32 x 32 and MADBase then look
    // alike, and like the characters of a page.
    private static float[] Frame(float[] image, int rows)
    {
        var framed = new float[Size * Size];
        ContentFrame.Fit(image, rows, rows, framed, Size);
        return framed;
    }

    private static float[] Row(float[,] images, int index)
    {
        var row = new float[Size * Size];
        Buffer.BlockCopy(images, index * Size * Size * sizeof(float), row, 0, Size * Size * sizeof(float));
        return row;
    }
}
