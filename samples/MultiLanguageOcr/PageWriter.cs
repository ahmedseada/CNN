namespace MultiLanguageOcr;

/// <summary>
/// Writes a page of handwriting for a text out of character images (test images a model never trained on): dark ink
/// on white paper, one line per text line, at twice the images' size. Right-to-left lines are laid out with
/// <see cref="TextOrder"/>.
/// </summary>
internal static class PageWriter
{
    private const int Scale = 2;          // page pixels per image pixel, so the reader has to shrink characters
    private const int LineHeight = 36;    // in image pixels
    private const int Margin = 16;

    /// <summary>
    /// The page's greyscale pixels in [0, 1], and its width and height. <paramref name="glyph"/> gives one image of a
    /// character, cropped to its ink (see <see cref="Crop"/>); <paramref name="rightToLeft"/> says which lines run right to left.
    /// </summary>
    public static (float[] Pixels, int Width, int Height) Write(
        string text, Func<char, Random, (float[] Pixels, int Width, int Height)> glyph, Func<string, bool>? rightToLeft = null, int seed = 1)
    {
        var random = new Random(seed);
        var placed = new List<(int X, int Y, float[] Glyph, int Width, int Height)>();
        int pageWidth = 0, y = Margin;
        foreach (var line in text.Replace("\r", "").Split('\n'))
        {
            int x = Margin;
            foreach (var (i, word) in TextOrder.Visual(line, rightToLeft?.Invoke(line) ?? false).Index())
            {
                if (i > 0)
                    x += 16;   // a space
                foreach (char ch in word)
                {
                    var (pixels, width, height) = glyph(ch, random);
                    placed.Add((x, y + (LineHeight - height) / 2, pixels, width, height));
                    x += width + random.Next(3, 7);
                }
            }
            pageWidth = Math.Max(pageWidth, x + Margin);
            y += LineHeight + 8;
        }

        int w = pageWidth * Scale, h = (y + Margin) * Scale;
        var page = Enumerable.Repeat(1f, w * h).ToArray();
        foreach (var (gx, gy, pixels, gw, gh) in placed)
            for (int py = 0; py < gh * Scale; py++)
                for (int px = 0; px < gw * Scale; px++)
                {
                    int i = (gy * Scale + py) * w + gx * Scale + px;
                    page[i] = Math.Min(page[i], 1f - pixels[py / Scale * gw + px / Scale]);
                }
        return (page, w, h);
    }

    /// <summary>A character image (ink high) cropped to its ink.</summary>
    public static (float[] Pixels, int Width, int Height) Crop(ReadOnlySpan<float> image, int rows, int columns)
    {
        int left = columns, right = -1, top = rows, bottom = -1;
        for (int y = 0; y < rows; y++)
            for (int x = 0; x < columns; x++)
                if (image[y * columns + x] > 0.2f)
                    (left, right, top, bottom) = (Math.Min(left, x), Math.Max(right, x), Math.Min(top, y), Math.Max(bottom, y));
        if (right < 0)
            return (new float[1], 1, 1);

        int width = right - left + 1, height = bottom - top + 1;
        var pixels = new float[width * height];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                pixels[y * width + x] = image[(top + y) * columns + left + x];
        return (pixels, width, height);
    }
}
