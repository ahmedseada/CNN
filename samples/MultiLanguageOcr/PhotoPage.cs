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
    /// <summary>A page whose global threshold takes in more than this share of its pixels is read as a photo.</summary>
    private const double PhotoInk = 0.15;

    /// <summary>What was done to a page, for the console.</summary>
    public sealed record Report(bool Photo, double GlobalInk, double Ink, int RuledPixels, int SurroundingRegions);

    /// <summary>
    /// The ink of <paramref name="image"/> (optionally cropped to <paramref name="crop"/> first): the classifier's
    /// global foreground for a clean page, or the photo pipeline when that takes in too much (or when
    /// <paramref name="photo"/> forces it).
    /// </summary>
    public static (ForegroundImage Ink, Report Report) Foreground(ImageData image, RegionClassifier classifier, PixelBox? crop = null, bool? photo = null)
    {
        if (crop is { } box)
            image = Crop(image, box);

        var global = classifier.Foreground(image);
        double globalInk = Share(global.Values, global.Threshold);
        if (!(photo ?? globalInk > PhotoInk))
            return (global, new Report(false, globalInk, globalInk, 0, 0));

        int w = image.Width, h = image.Height;
        var ink = LocalInk(Idrak.Vision.Foreground.Grey(image), w, h, radius: Math.Max(8, Math.Min(w, h) / 60), margin: 0.12f);
        int ruled = RemoveRuledLines(ink, w, h, run: Math.Max(40, w / 15));
        var page = new ForegroundImage(ink, w, h, threshold: 0.05f, inverted: true);
        int surroundings = RemoveSurroundings(page, ink, edges: crop is null);
        return (page, new Report(true, globalInk, Share(ink, 0.05f), ruled, surroundings));
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
        var sum = new double[(w + 1) * (h + 1)];
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
        for (int y = 0; y < h; y++)
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
        }
        return ink;
    }

    // Ruled lines: thin, faint strokes that run far sideways. A pixel is on one where ink continues sideways for `run` pixels
    // through it (a pixel up or down per step, for lines photographed at a slant) and is at most 4 pixels thick there,
    // so the strokes of letters crossing a line keep their ink above and below it.
    private static int RemoveRuledLines(float[] ink, int w, int h, int run)
    {
        bool On(int x, int y) => x >= 0 && y >= 0 && x < w && y < h && ink[y * w + x] > 0;
        int Reach(int x, int y, int dx)
        {
            int n = 0;
            while (n < run)
            {
                int nx = x + dx;
                if (On(nx, y)) { }
                else if (On(nx, y - 1)) y--;
                else if (On(nx, y + 1)) y++;
                else break;
                x = nx;
                n++;
            }
            return n;
        }

        var ruled = new List<int>();
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                if (!On(x, y) || Reach(x, y, 1) + Reach(x, y, -1) < run)
                    continue;
                int up = 0, down = 0;
                while (up < 6 && On(x, y - up - 1)) up++;
                while (down < 6 && On(x, y + down + 1)) down++;
                if (up + down + 1 <= 4)
                    ruled.Add(y * w + x);
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
