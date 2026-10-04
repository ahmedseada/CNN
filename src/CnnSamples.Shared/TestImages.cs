namespace CnnSamples.Shared;

/// <summary>Picks images to train on, show or export.</summary>
public static class TestImages
{
    /// <summary>
    /// <paramref name="count"/> image indices out of <paramref name="total"/>, chosen at random with a fixed seed, in
    /// ascending order. Taking the first <paramref name="count"/> instead would leave out whole classes from a data set
    /// sorted by class.
    /// </summary>
    public static int[] RandomSubset(int total, int count, int seed = 1)
    {
        var indices = Enumerable.Range(0, total).ToArray();
        new Random(seed).Shuffle(indices);
        return [.. indices.Take(count).Order()];
    }

    /// <summary>
    /// The indices of <paramref name="count"/> images that cover the classes in order: the first image of class 0, of
    /// class 1, ..., then the second of each, and so on. A test set sorted by class (EMNIST Letters starts with 800
    /// A's) still gives one of each.
    /// </summary>
    public static int[] OnePerClass(IReadOnlyList<int> labels, int classes, int count)
    {
        var byClass = Enumerable.Range(0, classes).Select(_ => new Queue<int>()).ToArray();
        for (int i = 0; i < labels.Count; i++)
            byClass[labels[i]].Enqueue(i);

        var picked = new List<int>(count);
        while (picked.Count < count && byClass.Any(q => q.Count > 0))
            foreach (var queue in byClass)
                if (picked.Count < count && queue.Count > 0)
                    picked.Add(queue.Dequeue());
        return [.. picked];
    }
}
