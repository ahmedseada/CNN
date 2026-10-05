using Idrak.Vision;

namespace MultiLanguageOcr;

/// <summary>One character found on a page: its line, its box, and whether a space comes before it.</summary>
internal sealed record Glyph(int Line, PixelBox Box, bool SpaceBefore);

/// <summary>
/// Finds the characters of a page for Idrak's <see cref="RegionClassifier"/>: splits the page's foreground into text
/// lines and each line into characters by the gaps between them, in reading order on the page (lines top to bottom,
/// characters left to right). Idrak extracts the foreground and frames and classifies the boxes; this layout is the
/// application's part.
/// </summary>
/// <remarks>
/// This works on clean pages with lines of separate characters: printed-style handwriting, scans of forms, the demo's
/// pages, in any script whose characters stand apart (Latin capitals, digits, isolated Arabic letters with their dots).
/// Joined-up handwriting would need a model that reads whole words.
/// </remarks>
internal sealed class TextLineProposer : IRegionProposer
{
    /// <inheritdoc />
    public IReadOnlyList<PixelBox> Propose(ForegroundImage image) => [.. Find(image).Select(g => g.Box)];

    /// <summary>The characters of a page, with their lines and the spaces between words.</summary>
    /// <remarks>
    /// A space is a gap wider than half the line's typical character height (the line's band is no measure: tall
    /// letters, descenders and Arabic dots make it far taller than a character). A line whose characters are much
    /// smaller than the page's (fine print at the paper's edge, specks) is left out.
    /// </remarks>
    public List<Glyph> Find(ForegroundImage image)
    {
        int width = image.Width, height = image.Height;
        var lines = new List<List<PixelBox>>();
        foreach (var (top, bottom) in MergeThin(Runs([.. Enumerable.Range(0, height).Select(y => Count(image, y, 0, width, horizontal: true))], minGap: 2, minLength: 1)))
        {
            var columns = Enumerable.Range(0, width).Select(x => Count(image, x, top, bottom, horizontal: false)).ToArray();
            var boxes = new List<PixelBox>();
            foreach (var (left, right) in Split(Runs(columns, minGap: 1, minLength: 1), columns, bottom - top))
            {
                // The character's own rows within the line.
                int charTop = bottom, charBottom = top;
                for (int y = top; y < bottom; y++)
                    for (int x = left; x < right; x++)
                        if (image.IsForeground(x, y)) { charTop = Math.Min(charTop, y); charBottom = Math.Max(charBottom, y + 1); }
                if (charBottom > charTop && (right - left) * (charBottom - charTop) >= 6)   // else a speck
                    boxes.Add(new PixelBox(left, charTop, right - left, charBottom - charTop));
            }
            if (boxes.Count > 0)
                lines.Add(boxes);
        }

        int page = Median(lines.SelectMany(l => l.Select(b => b.Height)));
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

    private static int Count(ForegroundImage image, int index, int from, int to, bool horizontal)
    {
        int count = 0;
        for (int i = from; i < to; i++)
            if (horizontal ? image.IsForeground(i, index) : image.IsForeground(index, i))
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
}
