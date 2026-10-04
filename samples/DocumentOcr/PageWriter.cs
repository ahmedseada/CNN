namespace DocumentOcr;

/// <summary>
/// Writes a page of handwriting for a text out of EMNIST test characters (images the model never trained on): dark
/// ink on white paper, one line per text line, at twice EMNIST's size.
/// </summary>
internal static class PageWriter
{
    private const int Scale = 2;          // page pixels per EMNIST pixel, so the reader has to shrink characters
    private const int LineHeight = 32;    // in EMNIST pixels
    private const int Margin = 16;

    /// <summary>The page's greyscale pixels in [0, 1], and its width and height.</summary>
    public static (float[] Pixels, int Width, int Height) Write(string text, float[,] images, int[] labels, string[] classes, int seed = 1)
    {
        // Each character's images; letters that EMNIST Balanced merges (c and C...) use the class that has them.
        var byClass = classes.Select((c, i) => (c, i)).ToDictionary(p => p.c, p => p.i);
        var pools = Enumerable.Range(0, classes.Length).Select(_ => new List<int>()).ToArray();
        for (int i = 0; i < labels.Length; i++)
            pools[labels[i]].Add(i);

        var random = new Random(seed);
        var lines = text.Replace("\r", "").Split('\n');
        var placed = new List<(int X, int Y, float[] Glyph, int Width, int Height)>();
        int pageWidth = 0, y = Margin;
        foreach (var line in lines)
        {
            int x = Margin;
            foreach (char ch in line)
            {
                if (ch == ' ')
                {
                    x += 16;
                    continue;
                }
                string key = byClass.ContainsKey(ch.ToString()) ? ch.ToString() : char.ToUpperInvariant(ch).ToString();
                if (!byClass.TryGetValue(key, out int label))
                    throw new ArgumentException($"'{ch}' is not one of EMNIST Balanced's characters ({string.Concat(classes)}).");

                var pool = pools[label];
                var (glyph, width, height) = Crop(images, pool[random.Next(pool.Count)]);
                placed.Add((x, y + (LineHeight - height) / 2, glyph, width, height));
                x += width + random.Next(3, 7);
            }
            pageWidth = Math.Max(pageWidth, x + Margin);
            y += LineHeight + 8;
        }

        int w = pageWidth * Scale, h = (y + Margin) * Scale;
        var page = Enumerable.Repeat(1f, w * h).ToArray();
        foreach (var (gx, gy, glyph, gw, gh) in placed)
            for (int py = 0; py < gh * Scale; py++)
                for (int px = 0; px < gw * Scale; px++)
                {
                    int i = (gy * Scale + py) * w + gx * Scale + px;
                    page[i] = Math.Min(page[i], 1f - glyph[py / Scale * gw + px / Scale]);
                }
        return (page, w, h);
    }

    // A test image cropped to its ink.
    private static (float[] Glyph, int Width, int Height) Crop(float[,] images, int index)
    {
        int left = 28, right = -1, top = 28, bottom = -1;
        for (int yy = 0; yy < 28; yy++)
            for (int xx = 0; xx < 28; xx++)
                if (images[index, yy * 28 + xx] > 0.2f)
                    (left, right, top, bottom) = (Math.Min(left, xx), Math.Max(right, xx), Math.Min(top, yy), Math.Max(bottom, yy));

        int width = right - left + 1, height = bottom - top + 1;
        var glyph = new float[width * height];
        for (int yy = 0; yy < height; yy++)
            for (int xx = 0; xx < width; xx++)
                glyph[yy * width + xx] = images[index, (top + yy) * 28 + left + xx];
        return (glyph, width, height);
    }
}
