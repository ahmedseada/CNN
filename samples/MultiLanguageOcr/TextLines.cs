using System.Buffers;
using Idrak.Vision;

namespace MultiLanguageOcr;

/// <summary>One character found on a page: its line, its box, and whether a space comes before it.</summary>
internal sealed record Glyph(int Line, PixelBox Box, bool SpaceBefore);

/// <summary>
/// Finds the characters of a page for Idrak's <see cref="RegionClassifier"/>: splits the page's foreground into text
/// lines and each line into characters by the gaps between them, in reading order on the page (lines top to bottom,
/// characters left to right), with the spaces between words. Idrak frames and classifies the boxes; this layout is the
/// application's part. (It is no <see cref="IRegionProposer"/>: the reader needs the lines and spaces, which a list of
/// boxes does not carry.)
/// </summary>
/// <remarks>
/// This works on clean pages with lines of separate characters: printed-style handwriting, scans of forms, the demo's
/// pages, in any script whose characters stand apart (Latin capitals, digits, isolated Arabic letters with their dots).
/// Joined-up handwriting would need a model that reads whole words.
/// </remarks>
internal static class TextLines
{
    /// <summary>The characters of a page, with their lines and the spaces between words.</summary>
    /// <remarks>
    /// A space is a gap wider than half the line's typical character height (the line's band is no measure: tall
    /// letters, descenders and Arabic dots make it far taller than a character). A line whose characters are much
    /// smaller than the page's (fine print at the paper's edge, specks, a printed header) is left out; the page's
    /// typical height weighs each line by its ink, so the writing sets it.
    /// </remarks>
    public static List<Glyph> Find(ForegroundImage image)
    {
        int width = image.Width, height = image.Height;

        // One byte per pixel, 1 on ink, read row by row from here on: each row's ink is counted once, a line's columns are
        // summed a row at a time (in memory order), and a character's own rows are found with vectorized searches.
        var mask = ArrayPool<byte>.Shared.Rent(width * height);               // scratch: every entry is written below
        try
        {
            return Find(image, mask);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(mask);
        }
    }

    private static List<Glyph> Find(ForegroundImage image, byte[] mask)
    {
        int width = image.Width, height = image.Height;
        var values = image.Values;
        float threshold = image.Threshold;
        var rows = new int[height];
        for (int y = 0; y < height; y++)
        {
            int count = 0;
            for (int x = 0, i = y * width; x < width; x++, i++)
            {
                bool ink = values[i] > threshold;
                mask[i] = ink ? (byte)1 : (byte)0;
                count += ink ? 1 : 0;
            }
            rows[y] = count;
        }

        var lines = new List<List<PixelBox>>();
        var columns = new int[width];
        foreach (var (top, bottom) in SplitMerged(MergeThin(Runs(rows, minGap: 2, minLength: 1)), rows))
        {
            Array.Clear(columns);
            for (int y = top; y < bottom; y++)
            {
                var row = mask.AsSpan(y * width, width);
                for (int x = 0; x < width; x++)
                    columns[x] += row[x];
            }

            var boxes = new List<PixelBox>();
            foreach (var (left, right) in Split(Runs(columns, minGap: 1, minLength: 1), columns, bottom - top))
            {
                // The character's own rows within the line: the first and last rows with ink between its columns.
                int charTop = top, charBottom = bottom;
                while (charTop < bottom && !mask.AsSpan(charTop * width + left, right - left).ContainsAnyExcept((byte)0))
                    charTop++;
                while (charBottom > charTop && !mask.AsSpan((charBottom - 1) * width + left, right - left).ContainsAnyExcept((byte)0))
                    charBottom--;
                if (charBottom > charTop && (right - left) * (charBottom - charTop) >= 6)   // else a speck
                    boxes.Add(new PixelBox(left, charTop, right - left, charBottom - charTop));
            }
            if (boxes.Count > 0)
                lines.Add(boxes);
        }

        // The page's typical character height: the lines' typical heights, each weighted by its line's area of boxes, so
        // the writing counts and specks or fine print do not.
        var weighted = lines.Select(l => (Height: Median(l.Select(b => b.Height)), Weight: l.Sum(b => (long)b.Area))).OrderBy(l => l.Height).ToList();
        long half = weighted.Sum(l => l.Weight) / 2, seen = 0;
        int page = weighted.FirstOrDefault(l => (seen += l.Weight) > half).Height;
        var glyphs = new List<Glyph>();
        foreach (var boxes in lines)
        {
            int typical = Median(boxes.Select(b => b.Height));
            if (typical < 0.4 * page)
                continue;
            int line = glyphs.Count == 0 ? 0 : glyphs[^1].Line + 1;
            for (int i = 0; i < boxes.Count; i++)
                glyphs.Add(new Glyph(line, boxes[i], SpaceBefore: i > 0 && boxes[i].X - boxes[i - 1].Right > 0.5 * typical));
        }
        return glyphs;
    }

    private static int Median(IEnumerable<int> values)
    {
        var sorted = values.Order().ToArray();
        return sorted.Length == 0 ? 0 : sorted[sorted.Length / 2];
    }

    // Bands much taller than the others that hold an almost empty row in their middle are lines that a descender, a
    // dot or a stroke joined: each is cut at its emptiest middle row (a row with at most 8% of the band's fullest
    // row's ink, in its middle 70%) when both parts are line-sized and neither is mostly empty, and the parts are
    // checked again. A tall line of its own (Arabic with dots above and below) has no such row, and a line's
    // descenders hold far less ink than the line: both stay whole.
    private static List<(int Start, int End)> SplitMerged(List<(int Start, int End)> bands, int[] rows)
    {
        if (bands.Count < 2)
            return bands;
        double typical = bands.Select(b => b.End - b.Start).Order().ElementAt(bands.Count / 2);
        var result = new List<(int Start, int End)>();
        var pending = new Stack<(int Start, int End)>(Enumerable.Reverse(bands));
        while (pending.Count > 0)
        {
            var (start, end) = pending.Pop();
            int length = end - start;
            if (length > 1.5 * typical)
            {
                int full = 0;
                for (int y = start; y < end; y++)
                    full = Math.Max(full, rows[y]);
                int from = start + (int)(0.15 * length), to = end - (int)(0.15 * length), cut = -1;
                for (int y = from; y < to; y++)
                    if (cut < 0 || rows[y] < rows[cut])
                        cut = y;
                long Ink(int a, int b) { long n = 0; for (int y = a; y < b; y++) n += rows[y]; return n; }
                bool lineSized = cut - start >= 0.6 * typical && end - cut >= 0.6 * typical;
                long above = Ink(start, cut), below = Ink(cut, end);
                if (cut > start && rows[cut] <= 0.08 * full && lineSized && Math.Min(above, below) >= 0.3 * Math.Max(above, below))
                {
                    pending.Push((cut, end));
                    pending.Push((start, cut));
                    continue;
                }
            }
            result.Add((start, end));
        }
        return result;
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
}
