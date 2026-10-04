using CnnSamples.Shared;
using Idrak.Data;

namespace LettersCnn;

/// <summary>
/// Reads the EMNIST Letters split (handwritten A-Z, upper and lower case together) that
/// <c>dotnet idrak @emnist.rsp</c> downloaded, as Idrak datasets.
/// </summary>
internal static class Emnist
{
    public const int Rows = EmnistArchive.Rows;
    public const int Columns = EmnistArchive.Columns;

    /// <summary>"A" to "Z". EMNIST Letters labels them 1 to 26 and puts both cases of a letter in one class.</summary>
    public static readonly string[] Letters = [.. Enumerable.Range('A', 26).Select(c => ((char)c).ToString())];

    /// <summary>
    /// The training and test sets as one-hot classification datasets of [1, 28, 28] images in [0, 1] (a white letter on
    /// black, the right way up), with the test labels (0 = A) for reporting.
    /// </summary>
    public static (Dataset Train, Dataset Test, int[] TestLabels) Load(string dataFolder)
    {
        var folders = DataFiles.SearchFolders(dataFolder);
        var (trainImages, trainLabels) = Read("train", folders);
        var (testImages, testLabels) = Read("test", folders);
        return (ToDataset(trainImages, trainLabels), ToDataset(testImages, testLabels), testLabels);
    }

    private static Dataset ToDataset(float[,] images, int[] labels) =>
        Dataset.FromClassLabels(images, labels, Letters.Length, Letters).WithFeatureShape(1, Rows, Columns);

    private static (float[,] Images, int[] Labels) Read(string split, IReadOnlyList<string> folders)
    {
        var (images, letters) = EmnistArchive.ReadSplit("letters", split, folders);
        for (int i = 0; i < letters.Length; i++)
        {
            letters[i]--;   // 1-26 -> 0-25
            if ((uint)letters[i] >= 26)
                throw new InvalidDataException($"Label {letters[i] + 1} is not a letter (EMNIST Letters uses 1-26).");
        }
        return (images, letters);
    }
}
