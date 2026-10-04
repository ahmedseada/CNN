using CnnSamples.Shared;
using Idrak.Data;

namespace DocumentOcr;

/// <summary>
/// The EMNIST Balanced split: handwritten digits, capital letters and the 11 lower-case letters that look different
/// from their capitals (a b d e f g h n q r t); 47 classes.
/// </summary>
internal static class Characters
{
    public const int Rows = EmnistArchive.Rows;
    public const int Columns = EmnistArchive.Columns;

    /// <summary>
    /// The folders searched for gzip.zip: this sample's data folder, LettersCnn's (the same download), then Idrak's cache.
    /// </summary>
    public static string[] Folders(string dataFolder) =>
        [.. DataFiles.SearchFolders(dataFolder), Path.Combine("..", "LettersCnn", "data")];

    /// <summary>Each class's character, in label order, from the split's mapping file.</summary>
    public static string[] Classes(string dataFolder)
    {
        var mapping = EmnistArchive.ReadMapping("balanced", Folders(dataFolder));
        return [.. Enumerable.Range(0, mapping.Count).Select(label => mapping[label].ToString())];
    }

    /// <summary>The training and test sets as [1, 28, 28] image datasets, with the test images and labels.</summary>
    public static (Dataset Train, Dataset Test, float[,] TestImages, int[] TestLabels, string[] Classes) Load(string dataFolder)
    {
        var folders = Folders(dataFolder);
        var classes = Classes(dataFolder);
        var (trainImages, trainLabels) = EmnistArchive.ReadSplit("balanced", "train", folders);
        var (testImages, testLabels) = EmnistArchive.ReadSplit("balanced", "test", folders);
        return (ToDataset(trainImages, trainLabels, classes), ToDataset(testImages, testLabels, classes), testImages, testLabels, classes);
    }

    private static Dataset ToDataset(float[,] images, int[] labels, string[] classes) =>
        Dataset.FromClassLabels(images, labels, classes.Length, classes).WithFeatureShape(1, Rows, Columns);
}
