namespace CnnSamples.Shared;

/// <summary>Console output shared by the samples: images as text, and per-class results.</summary>
public static class ConsoleReport
{
    /// <summary>Prints a greyscale image (values in [0, 1]) as text, every other row so it keeps its shape in a terminal.</summary>
    public static void PrintImage(ReadOnlySpan<float> pixels, int rows, int columns)
    {
        const string shades = " .:-=+*#%@";
        for (int y = 0; y < rows; y += 2)
        {
            var line = new char[columns];
            for (int x = 0; x < columns; x++)
            {
                float v = y + 1 < rows ? Math.Max(pixels[y * columns + x], pixels[(y + 1) * columns + x]) : pixels[y * columns + x];
                line[x] = shades[(int)(Math.Clamp(v, 0f, 1f) * (shades.Length - 1))];
            }
            Console.WriteLine(new string(line));
        }
    }

    /// <summary>
    /// Prints the test results: the confusion matrix for up to 12 classes; for more, each class's accuracy and the
    /// most common mistakes. Returns the accuracy.
    /// </summary>
    public static double PrintResults(IReadOnlyList<int> labels, IReadOnlyList<int> predicted, IReadOnlyList<string> classes)
    {
        int n = classes.Count;
        var confusion = new int[n, n];
        for (int i = 0; i < labels.Count; i++)
            confusion[labels[i], predicted[i]]++;

        int correct = Enumerable.Range(0, n).Sum(c => confusion[c, c]);
        double Accuracy(int c)
        {
            int total = Enumerable.Range(0, n).Sum(p => confusion[c, p]);
            return (double)confusion[c, c] / Math.Max(total, 1);
        }

        Console.WriteLine();
        if (n <= 12)
        {
            Console.WriteLine("Confusion matrix (rows: true class, columns: predicted)");
            Console.WriteLine("      " + string.Concat(classes.Select(c => $"{c,6}")) + "   accuracy");
            for (int t = 0; t < n; t++)
            {
                Console.Write($"{classes[t],6}");
                for (int p = 0; p < n; p++)
                    Console.Write($"{confusion[t, p],6}");
                Console.WriteLine($"   {Accuracy(t):P1}");
            }
        }
        else
        {
            Console.WriteLine("Accuracy per class");
            for (int c = 0; c < n; c += 6)
                Console.WriteLine(string.Join("   ", Enumerable.Range(c, Math.Min(6, n - c)).Select(i => $"{classes[i],2} {Accuracy(i),7:P1}")));

            Console.WriteLine();
            Console.WriteLine("Most common mistakes (true -> predicted: count)");
            var mistakes = from t in Enumerable.Range(0, n)
                           from p in Enumerable.Range(0, n)
                           where t != p && confusion[t, p] > 0
                           orderby confusion[t, p] descending
                           select $"{classes[t]} -> {classes[p]}: {confusion[t, p]}";
            Console.WriteLine(string.Join("   ", mistakes.Take(8)));
        }

        double accuracy = (double)correct / Math.Max(labels.Count, 1);
        Console.WriteLine($"Test accuracy: {accuracy:P2} ({correct:N0}/{labels.Count:N0})");
        return accuracy;
    }
}
