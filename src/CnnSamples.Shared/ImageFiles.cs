using System.Text;

namespace CnnSamples.Shared;

/// <summary>Image files shared by the samples.</summary>
public static class ImageFiles
{
    /// <summary>Writes a greyscale image (values in [0, 1]) as a binary PGM file.</summary>
    public static void WritePgm(string path, ReadOnlySpan<float> pixels, int rows, int columns)
    {
        using var file = File.Create(path);
        file.Write(Encoding.ASCII.GetBytes($"P5\n{columns} {rows}\n255\n"));
        foreach (var v in pixels)
            file.WriteByte((byte)MathF.Round(Math.Clamp(v, 0f, 1f) * 255f));
    }

    /// <summary>
    /// Reads an image file (PNG, BMP, PGM, PPM) as greyscale at the given size, inverted when it is dark on light, to
    /// match light-on-dark training data such as MNIST and EMNIST.
    /// </summary>
    public static float[] LoadLightOnDark(string path, int rows, int columns)
    {
        var pixels = Idrak.Data.ImageCodecs.Load(path, channels: 1, height: rows, width: columns);
        if (pixels.Average() > 0.5f)
            for (int i = 0; i < pixels.Length; i++)
                pixels[i] = 1f - pixels[i];
        return pixels;
    }
}
