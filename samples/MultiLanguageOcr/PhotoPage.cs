using System.Buffers;
using Idrak.Data;
using Idrak.Vision;

namespace MultiLanguageOcr;

/// <summary>
/// The ink of a photographed or scanned page. A clean page (dark writing on even paper) takes Idrak's global
/// threshold. A photo, where one threshold also takes in shadows, the desk, a notebook's binding and its cover, takes
/// a local threshold instead (a pixel is ink where it is clearly darker than its surroundings). Then the ruled lines of
/// notebook paper are removed, and so is what lies around the page: regions touching the photo's edge and regions far
/// larger than writing.
/// </summary>
/// <remarks>
/// The local threshold is general (uneven light affects every photographed image) and is meant to move into
/// Idrak.Vision once it has settled here; ruled lines and the page's surroundings are particular to pages of text.
/// </remarks>
internal static class PhotoPage
{
    /// <summary>A page whose global threshold takes in more than this share of its pixels is read as a photo (so is one where
    /// a single region of it spans half the image's width or a third of its height).</summary>
    private const double PhotoInk = 0.15;

    /// <summary>What was done to a page, for the console.</summary>
    public sealed record Report(bool Photo, double GlobalInk, double Ink, int RuledPixels, int SurroundingRegions);

    /// <summary>
    /// The ink of <paramref name="image"/> (optionally cropped to <paramref name="crop"/> first): the global
    /// foreground for a clean page (as a <see cref="RegionClassifier"/>'s default <c>Foreground</c> makes it), or the photo pipeline when that takes in too much (or when
    /// <paramref name="photo"/> forces it).
    /// </summary>
    public static (ForegroundImage Ink, Report Report) Foreground(ImageData image, PixelBox? crop = null, bool? photo = null)
    {
        if (crop is { } box)
            image = Crop(image, box);

        // One grey image serves both thresholds: the global foreground (Idrak's polarity rule and Otsu, as the
        // classifier's default Foreground) and, for a photo, the local threshold.
        int w = image.Width, h = image.Height;
        var grey = Idrak.Vision.Foreground.Grey(image);
        var global = Global(grey, w, h);
        double globalInk = Share(global.Values, global.Threshold);
        if (!(photo ?? (globalInk > PhotoInk || SpansPage(global))))
            return (global, new Report(false, globalInk, globalInk, 0, 0));

        var ink = LocalInk(grey, w, h, radius: Math.Max(8, Math.Min(w, h) / 60), margin: 0.12f);
        int ruled = RemoveRuledLines(ink, w, h, run: Math.Max(40, w / 15));
        var page = new ForegroundImage(ink, w, h, threshold: 0.05f, inverted: true);
        int surroundings = RemoveSurroundings(page, ink, edges: crop is null);
        return (page, new Report(true, globalInk, Share(ink, 0.05f), ruled, surroundings));
    }

    // The global foreground of a grey image, as Foreground.Extract makes it by default: dark ink inverted when the
    // image is mostly light, Otsu's threshold. It takes its own copy, so the grey stays for the local threshold.
    private static ForegroundImage Global(float[] grey, int w, int h)
    {
        double sum = 0;
        foreach (float v in grey)
            sum += v;
        bool invert = sum / Math.Max(grey.Length, 1) > 0.5;
        var values = new float[grey.Length];
        for (int i = 0; i < values.Length; i++)
            values[i] = invert ? 1f - grey[i] : grey[i];
        return new ForegroundImage(values, w, h, Idrak.Vision.Foreground.Otsu(values), invert);
    }

    // Whether one region of a foreground spans half the image's width or a third of its height: writing never does (a
    // line of it is far less tall), the dark desk, cover or paper edge around a photographed page does (even when they
    // are less than PhotoInk of the pixels). Judged on a copy a quarter the size each way (a 4 x 4 block is ink when
    // any pixel of it is), where finding the regions takes a sixteenth of the work and the spans hardly change.
    private static bool SpansPage(ForegroundImage foreground)
    {
        const int Block = 4;
        int w = foreground.Width, h = foreground.Height, sw = (w + Block - 1) / Block, sh = (h + Block - 1) / Block;
        var small = new float[sw * sh];
        var values = foreground.Values;
        float threshold = foreground.Threshold;
        for (int y = 0; y < h; y++)
        {
            int row = y / Block * sw;
            for (int x = 0, i = y * w; x < w; x++, i++)
                if (values[i] > threshold)
                    small[row + x / Block] = 1f;
        }
        return ConnectedComponents.Find(new ForegroundImage(small, sw, sh, 0.5f), Connectivity.Eight).Regions
            .Any(r => r.Box.Width > sw / 2 || r.Box.Height > sh / 3);
    }

    /// <summary>Parses a crop given as x,y,width,height.</summary>
    public static PixelBox ParseCrop(string text)
    {
        var parts = text.Split(',', StringSplitOptions.TrimEntries);
        return parts.Length == 4 && parts.All(p => int.TryParse(p, out _)) && int.Parse(parts[2]) > 0 && int.Parse(parts[3]) > 0
            ? new PixelBox(int.Parse(parts[0]), int.Parse(parts[1]), int.Parse(parts[2]), int.Parse(parts[3]))
            : throw new ArgumentException($"--crop takes x,y,width,height in pixels (e.g. 120,240,750,590), not '{text}'.");
    }

    // Ink where a pixel is darker than the mean of the (2r + 1)² window around it by more than `margin`
    // (Bradley's local threshold, the window sums from an integral image); 0 elsewhere.
    private static float[] LocalInk(float[] grey, int w, int h, int radius, float margin)
    {
        // The integral image is scratch, rented and returned (a page's is 11 MB: allocating one per page churns the large
        // object heap). Every entry is written below except the first row and column, which stay zero.
        var sum = ArrayPool<double>.Shared.Rent((w + 1) * (h + 1));
        try
        {
            return LocalInk(grey, w, h, radius, margin, sum);
        }
        finally
        {
            ArrayPool<double>.Shared.Return(sum);
        }
    }

    private static float[] LocalInk(float[] grey, int w, int h, int radius, float margin, double[] sum)
    {
        sum.AsSpan(0, w + 1).Clear();
        for (int y = 0; y < h; y++)
            sum[(y + 1) * (w + 1)] = 0;
        for (int y = 0; y < h; y++)
        {
            double row = 0;
            for (int x = 0; x < w; x++)
            {
                row += grey[y * w + x];
                sum[(y + 1) * (w + 1) + x + 1] = sum[y * (w + 1) + x + 1] + row;
            }
        }

        var ink = new float[w * h];
        Parallel.For(0, h, y =>                                                   // rows are independent
        {
            int y0 = Math.Max(0, y - radius), y1 = Math.Min(h, y + radius + 1);
            for (int x = 0; x < w; x++)
            {
                int x0 = Math.Max(0, x - radius), x1 = Math.Min(w, x + radius + 1);
                double mean = (sum[y1 * (w + 1) + x1] - sum[y0 * (w + 1) + x1] - sum[y1 * (w + 1) + x0] + sum[y0 * (w + 1) + x0])
                    / ((x1 - x0) * (y1 - y0));
                float darker = (float)(mean - grey[y * w + x]);
                ink[y * w + x] = darker > margin ? darker : 0f;   // how much darker: pen ink more than a printed line
            }
        });
        return ink;
    }

    // Ruled lines: thin, faint strokes that run far sideways. A pixel is on one where ink continues sideways for `run` pixels
    // through it (a pixel up or down per step, for lines photographed at a slant) and is at most 4 pixels thick there,
    // so the strokes of letters crossing a line keep their ink above and below it.
    private static int RemoveRuledLines(float[] ink, int w, int h, int run)
    {
        // Scratch, rented and returned: every entry of each is written before it is read.
        var on = ArrayPool<byte>.Shared.Rent(w * h);
        var right = ArrayPool<ushort>.Shared.Rent(w * h);
        var left = ArrayPool<ushort>.Shared.Rent(w * h);
        try
        {
            return RemoveRuledLines(ink, w, h, run, on, right, left);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(on);
            ArrayPool<ushort>.Shared.Return(right);
            ArrayPool<ushort>.Shared.Return(left);
        }
    }

    private static int RemoveRuledLines(float[] ink, int w, int h, int run, byte[] on, ushort[] right, ushort[] left)
    {
        // Column by column from here on (index x * h + y): every step below reads a neighbouring column or walks up and
        // down one, which in this order is memory read in sequence.
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                on[x * h + y] = ink[y * w + x] > 0 ? (byte)1 : (byte)0;
        bool On(int x, int y) => x >= 0 && y >= 0 && x < w && y < h && on[x * h + y] != 0;

        // How far ink continues sideways from each pixel (at most `run` steps), each step to the next column on the
        // same row, else a row up, else a row down: one pass per direction, each pixel's reach one more than that of
        // the pixel it steps to, instead of walking up to `run` steps from every pixel.
        right.AsSpan((w - 1) * h, h).Clear();                                    // the last column reaches nothing to its right
        left.AsSpan(0, h).Clear();                                               // the first nothing to its left
        for (int x = w - 2; x >= 0; x--)
        {
            int c = x * h, n = c + h;                                              // this column and the next
            for (int y = 0; y < h; y++)
            {
                int next = on[n + y] != 0 ? y : y > 0 && on[n + y - 1] != 0 ? y - 1 : y + 1 < h && on[n + y + 1] != 0 ? y + 1 : -1;
                right[c + y] = next >= 0 ? (ushort)Math.Min(run, 1 + right[n + next]) : (ushort)0;
            }
        }
        for (int x = 1; x < w; x++)
        {
            int c = x * h, p = c - h;                                              // this column and the previous
            for (int y = 0; y < h; y++)
            {
                int next = on[p + y] != 0 ? y : y > 0 && on[p + y - 1] != 0 ? y - 1 : y + 1 < h && on[p + y + 1] != 0 ? y + 1 : -1;
                left[c + y] = next >= 0 ? (ushort)Math.Min(run, 1 + left[p + next]) : (ushort)0;
            }
        }

        var ruled = new List<int>();
        for (int x = 0; x < w; x++)
            for (int y = 0; y < h; y++)
            {
                int i = x * h + y;
                if (on[i] == 0 || right[i] + left[i] < run)
                    continue;
                int up = 0, down = 0;
                while (up < 6 && On(x, y - up - 1)) up++;
                while (down < 6 && On(x, y + down + 1)) down++;
                if (up + down + 1 <= 4)
                    ruled.Add(y * w + x);                                          // back in the image's row order
            }

        // A pen stroke running along a ruled line stays: pen ink is darker than the printed line, so a pixel on a line
        // well darker than the line's own median (most of what was found is the line) is kept. On the notebook photo this
        // keeps the 5's bar and the 3's lower curve, which the line erased, and every digit of its number line reads.
        var lineInk = ruled.Select(i => ink[i]).Order().ToArray();
        float keep = lineInk.Length > 0 ? 1.9f * lineInk[lineInk.Length / 2] : float.MaxValue;
        foreach (int i in ruled)
            if (ink[i] < keep)
                ink[i] = 0;
        return ruled.Count;
    }

    // What lies around the page: regions at the photo's edge (within 1.5%; not in a crop, whose edges the reader chose),
    // regions far larger than writing (a page edge, a shadow: past a quarter of the photo and several times the writing's
    // height), and regions drawn much thicker than the writing (a notebook's binding, a cover's edge). A region's
    // stroke is about 2 · area / perimeter thick; the writing's is the median stroke over the regions (pieces of
    // writing far outnumber the binding's loops, though these hold more ink). Returns how many regions were removed.
    private static int RemoveSurroundings(ForegroundImage page, float[] ink, bool edges)
    {
        int w = page.Width, h = page.Height;
        var map = ConnectedComponents.Find(page, Connectivity.Eight);
        var labels = map.Labels;
        var perimeter = new int[map.Regions.Count + 1];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int l = labels[y * w + x];
                if (l != 0 && (x == 0 || y == 0 || x == w - 1 || y == h - 1
                    || labels[y * w + x - 1] != l || labels[y * w + x + 1] != l || labels[(y - 1) * w + x] != l || labels[(y + 1) * w + x] != l))
                    perimeter[l]++;
            }

        double Thickness(Region r) => 2.0 * r.Area / Math.Max(perimeter[r.Label], 1);
        var pieces = map.Regions.Where(r => r.Area >= 20).ToArray();
        var strokes = pieces.Select(Thickness).Order().ToArray();
        double writing = strokes.Length > 0 ? strokes[strokes.Length / 2] : 0;
        var heights = pieces.Select(r => r.Box.Height).Order().ToArray();
        int height = heights.Length > 0 ? heights[heights.Length / 2] : h;      // the writing's typical height

        var drop = new HashSet<int>();
        foreach (var r in map.Regions)
        {
            var b = r.Box;
            int mx = Math.Max(1, w * 3 / 200), my = Math.Max(1, h * 3 / 200);     // within 1.5% of the edge
            bool edge = edges && (b.X < mx || b.Y < my || b.Right > w - mx || b.Bottom > h - my);
            bool large = b.Width > Math.Max(w / 2, 12 * height) || b.Height > Math.Max(h / 4, 4 * height);
            bool thick = writing > 0 && r.Area >= 20 && Thickness(r) > 1.6 * writing;
            if (edge || large || thick)
                drop.Add(r.Label);
        }

        for (int i = 0; i < ink.Length; i++)
            if (labels[i] != 0 && drop.Contains(labels[i]))
                ink[i] = 0;
        return drop.Count;
    }

    private static double Share(ReadOnlySpan<float> values, float threshold)
    {
        int on = 0;
        foreach (float v in values)
            if (v > threshold)
                on++;
        return on / (double)Math.Max(values.Length, 1);
    }

    private static ImageData Crop(ImageData image, PixelBox box)
    {
        if (box.X < 0 || box.Y < 0 || box.Right > image.Width || box.Bottom > image.Height)
            throw new ArgumentException($"--crop {box.X},{box.Y},{box.Width},{box.Height} is not inside the {image.Width} x {image.Height} image.");
        int c = image.Channels, plane = image.Width * image.Height;
        var pixels = new float[c * box.Width * box.Height];
        for (int k = 0; k < c; k++)
            for (int y = 0; y < box.Height; y++)
                image.Pixels.AsSpan(k * plane + (box.Y + y) * image.Width + box.X, box.Width)
                    .CopyTo(pixels.AsSpan((k * box.Height + y) * box.Width, box.Width));
        return new ImageData(pixels, c, box.Height, box.Width);
    }
}
