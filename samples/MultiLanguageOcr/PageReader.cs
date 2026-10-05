using Idrak.Data;

namespace MultiLanguageOcr;

/// <summary>One character found on a page: its line, its place on the page, and its 28 x 28 image for the model.</summary>
internal sealed record Glyph(int Line, int Left, int Top, int Width, int Height, float[] Image, bool SpaceBefore);

/// <summary>
/// Finds the characters of a page: separates ink from paper, splits the page into text lines and each line into
/// characters by the gaps between them, and turns each character into a 28 x 28 image the way EMNIST made its own.
/// </summary>
/// <remarks>
/// This works on clean pages with lines of separate characters: printed-style handwriting, scans of forms, the demo's
/// pages, in any script whose characters stand apart (Latin capitals, digits, isolated Arabic letters with their dots).
/// Joined-up handwriting would need a model that reads whole words.
/// </remarks>
internal static class PageReader
{
    private const int Size = 28;
    private const int Border = 1;   // EMNIST's characters fill their frame up to about a pixel from the edge

    /// <summary>The characters of a page of <paramref name="width"/> x <paramref name="height"/> greyscale pixels in [0, 1].</summary>
    public static List<Glyph> Read(float[] grey, int width, int height)
    {
        var ink = Ink(grey);
        float threshold = Otsu(ink);
        var mask = ink.Select(v => v > threshold).ToArray();

        var glyphs = new List<Glyph>();
        var lines = MergeThin(Runs(Enumerable.Range(0, height).Select(y => Count(mask, width, y, 0, width, horizontal: true)).ToArray(), minGap: 2, minLength: 1));
        foreach (var (lineIndex, (top, bottom)) in lines.Index())
        {
            int lineHeight = bottom - top;
            var columns = Enumerable.Range(0, width).Select(x => Count(mask, width, x, top, bottom, horizontal: false)).ToArray();
            var characters = Split(Runs(columns, minGap: 1, minLength: 1), columns, lineHeight);
            int previousRight = -1;
            foreach (var (left, right) in characters)
            {
                // The character's own rows within the line.
                int charTop = bottom, charBottom = top;
                for (int y = top; y < bottom; y++)
                    for (int x = left; x < right; x++)
                        if (mask[y * width + x]) { charTop = Math.Min(charTop, y); charBottom = Math.Max(charBottom, y + 1); }
                if (charBottom <= charTop || (right - left) * (charBottom - charTop) < 6)
                    continue;   // a speck

                bool space = previousRight >= 0 && left - previousRight > 0.4 * lineHeight;
                previousRight = right;
                var image = Normalize(ink, mask, width, left, charTop, right - left, charBottom - charTop);
                glyphs.Add(new Glyph(lineIndex, left, charTop, right - left, charBottom - charTop, image, space));
            }
        }
        return glyphs;
    }

    /// <summary>
    /// A data set's character image (<paramref name="rows"/> x <paramref name="columns"/>, ink high) framed the way the
    /// reader frames a character cut from a page: cropped to its ink, centred and scaled to 28 x 28. Training on images
    /// framed like this makes characters from different data sets (EMNIST, AHCD's 32 x 32, MADBase) and from pages alike.
    /// </summary>
    public static float[] Renormalize(ReadOnlySpan<float> image, int rows, int columns)
    {
        var ink = image.ToArray();
        var mask = ink.Select(v => v > 0.25f).ToArray();
        int left = columns, right = -1, top = rows, bottom = -1;
        for (int y = 0; y < rows; y++)
            for (int x = 0; x < columns; x++)
                if (mask[y * columns + x])
                    (left, right, top, bottom) = (Math.Min(left, x), Math.Max(right, x), Math.Min(top, y), Math.Max(bottom, y));
        return right < 0 ? new float[Size * Size] : Normalize(ink, mask, columns, left, top, right - left + 1, bottom - top + 1);
    }

    // Row bands much thinner than a text line (the dots above and below Arabic letters, a stray mark) join the nearer
    // of their neighbouring bands, so they belong to their line instead of making one of their own.
    private static List<(int Start, int End)> MergeThin(List<(int Start, int End)> runs)
    {
        while (runs.Count > 1)
        {
            var heights = runs.Select(r => r.End - r.Start).Order().ToArray();
            double median = heights[heights.Length / 2];
            int thin = runs.FindIndex(r => r.End - r.Start < 0.45 * median);
            if (thin < 0)
                break;

            int gapAbove = thin > 0 ? runs[thin].Start - runs[thin - 1].End : int.MaxValue;
            int gapBelow = thin < runs.Count - 1 ? runs[thin + 1].Start - runs[thin].End : int.MaxValue;
            int other = gapAbove <= gapBelow ? thin - 1 : thin + 1;
            var (first, second) = other < thin ? (other, thin) : (thin, other);
            runs[first] = (runs[first].Start, runs[second].End);
            runs.RemoveAt(second);
        }
        return [.. runs.Where(r => r.End - r.Start >= 3)];
    }

    // Ink in [0, 1]: dark on light pages are inverted, so ink is high either way.
    private static float[] Ink(float[] grey) =>
        grey.Average() > 0.5f ? [.. grey.Select(v => 1f - v)] : [.. grey];

    // Otsu's threshold: the level that best separates the two groups of a histogram (paper and ink).
    private static float Otsu(float[] values)
    {
        var histogram = new int[256];
        foreach (var v in values)
            histogram[Math.Clamp((int)(v * 255f), 0, 255)]++;

        double total = values.Length, sum = 0;
        for (int i = 0; i < 256; i++)
            sum += i * (double)histogram[i];

        double backgroundSum = 0, best = -1;
        int backgroundCount = 0, level = 128;
        for (int i = 0; i < 256; i++)
        {
            backgroundCount += histogram[i];
            if (backgroundCount == 0 || backgroundCount == total)
                continue;
            backgroundSum += i * (double)histogram[i];
            double meanBack = backgroundSum / backgroundCount, meanInk = (sum - backgroundSum) / (total - backgroundCount);
            double between = backgroundCount * (total - backgroundCount) * (meanBack - meanInk) * (meanBack - meanInk);
            if (between > best)
                (best, level) = (between, i);
        }
        return Math.Max(level / 255f, 0.1f);
    }

    private static int Count(bool[] mask, int width, int index, int from, int to, bool horizontal)
    {
        int count = 0;
        for (int i = from; i < to; i++)
            if (horizontal ? mask[index * width + i] : mask[i * width + index])
                count++;
        return count;
    }

    // Runs of non-zero counts, merging runs separated by fewer than minGap zeros and dropping runs shorter than minLength.
    private static List<(int Start, int End)> Runs(int[] counts, int minGap, int minLength)
    {
        var runs = new List<(int Start, int End)>();
        int start = -1;
        for (int i = 0; i <= counts.Length; i++)
        {
            bool on = i < counts.Length && counts[i] > 0;
            if (on && start < 0)
                start = i;
            else if (!on && start >= 0)
            {
                if (runs.Count > 0 && start - runs[^1].End < minGap)
                    runs[^1] = (runs[^1].Start, i);
                else
                    runs.Add((start, i));
                start = -1;
            }
        }
        return [.. runs.Where(r => r.End - r.Start >= minLength)];
    }

    // Splits runs much wider than a character (touching characters) at their thinnest columns.
    private static List<(int Start, int End)> Split(List<(int Start, int End)> runs, int[] columns, int lineHeight)
    {
        var result = new List<(int Start, int End)>();
        foreach (var (start, end) in runs)
        {
            int pieces = (int)Math.Round((end - start) / (0.8 * lineHeight));
            if (pieces < 2 || end - start < 1.3 * lineHeight)
            {
                result.Add((start, end));
                continue;
            }

            int from = start;
            for (int p = 1; p < pieces; p++)
            {
                int expected = start + (end - start) * p / pieces, window = Math.Max(1, (end - start) / pieces / 3);
                int cut = Enumerable.Range(Math.Max(from + 1, expected - window), 2 * window + 1)
                    .Where(x => x < end - 1).DefaultIfEmpty(expected).MinBy(x => columns[x]);
                result.Add((from, cut));
                from = cut;
            }
            result.Add((from, end));
        }
        return result;
    }

    // The character centred in a square, kept to its aspect ratio, a pixel from the edge, shrunk to 28 x 28.
    private static float[] Normalize(float[] ink, bool[] mask, int pageWidth, int left, int top, int width, int height)
    {
        int side = Math.Max(width, height);
        int frame = (int)Math.Ceiling(side * Size / (double)(Size - 2 * Border));
        var square = new float[frame * frame];
        int offsetX = (frame - width) / 2, offsetY = (frame - height) / 2;
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int i = (top + y) * pageWidth + left + x;
                square[(offsetY + y) * frame + offsetX + x] = mask[i] ? ink[i] : 0f;
            }

        var image = new ImageData(square, 1, frame, frame).Resize(1, Size, Size);   // averages the pixels it shrinks
        float max = image.Max();
        return max > 0 ? [.. image.Select(v => v / max)] : image;   // full contrast, like EMNIST's
    }
}
