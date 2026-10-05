using System.Diagnostics;
using Idrak.Data;

namespace MultiLanguageOcr;

/// <summary>
/// Turns an image in a format Idrak does not decode (JPEG, WebP, HEIC, TIFF, GIF...) into an uncompressed 24-bit BMP
/// that it does. Register converters with <see cref="ImageConversion.Register"/>; the first available one that
/// converts the file wins.
/// </summary>
internal interface IImageConverter
{
    /// <summary>A name for messages.</summary>
    string Name { get; }

    /// <summary>Whether the converter can run here (its program is installed, the system has it).</summary>
    bool IsAvailable { get; }

    /// <summary>Converts <paramref name="source"/> into a BMP at <paramref name="bmp"/>; null on success, else why not.</summary>
    string? ConvertToBmp(string source, string bmp);
}

/// <summary>
/// Makes image files readable: one Idrak's codecs decode is used as it is; any other is converted to an uncompressed
/// BMP next to it (photo.jpeg → photo.bmp) by the registered converters, tried in order. Built in: Windows' own
/// imaging (through Windows PowerShell, on every Windows), ImageMagick and ffmpeg (where installed).
/// </summary>
/// <remarks>
/// A converter is a plug-in point meant to move into Idrak once it has settled here; a codec registered with
/// <see cref="ImageCodecs.Register"/> (a JPEG decoder, say) makes conversion unnecessary for its format.
/// </remarks>
internal static class ImageConversion
{
    private static readonly List<IImageConverter> Converters =
    [
        new WindowsImaging(),
        new ExternalTool("ImageMagick", "magick", (source, bmp) => [source, "-auto-orient", "-flatten", "-type", "TrueColor", "BMP3:" + bmp]),
        new ExternalTool("ffmpeg", "ffmpeg", (source, bmp) => ["-y", "-loglevel", "error", "-i", source, "-pix_fmt", "bgr24", bmp]),
    ];

    /// <summary>Registers a converter, asked before the ones registered earlier.</summary>
    public static void Register(IImageConverter converter) => Converters.Insert(0, converter ?? throw new ArgumentNullException(nameof(converter)));

    /// <summary>
    /// A path Idrak decodes holding <paramref name="path"/>'s image: the path itself when a codec reads it, else the
    /// BMP a converter wrote next to it. Says on the console what it converted.
    /// </summary>
    public static string Readable(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"No image at '{path}'.");
        if (ImageCodecs.ReadInfo(path) is not null)
            return path;

        string bmp = Path.ChangeExtension(path, ".bmp");
        if (File.Exists(bmp) && File.GetLastWriteTimeUtc(bmp) >= File.GetLastWriteTimeUtc(path) && ImageCodecs.ReadInfo(bmp) is not null)
        {
            Console.WriteLine($"Using the BMP converted earlier: {bmp}");
            return bmp;
        }

        var tried = new List<string>();
        foreach (var converter in Converters.ToArray())
        {
            if (!converter.IsAvailable)
            {
                tried.Add($"{converter.Name}: not installed");
                continue;
            }

            string? error = converter.ConvertToBmp(Path.GetFullPath(path), Path.GetFullPath(bmp));
            if (error is null && File.Exists(bmp) && ImageCodecs.ReadInfo(bmp) is not null)
            {
                Console.WriteLine($"Converted {Path.GetFileName(path)} to an uncompressed BMP with {converter.Name}: {bmp}");
                return bmp;
            }
            tried.Add($"{converter.Name}: {error ?? "wrote no readable BMP"}");
        }

        throw new InvalidDataException(
            $"{path}: Idrak reads {string.Join(", ", ImageCodecs.Names)} files, and no converter turned this one into a BMP " +
            $"({string.Join("; ", tried)}). Save it as PNG or BMP, or install ImageMagick or ffmpeg.");
    }

    // Runs a program; null when it exits with 0, else its error output (or why it could not start).
    internal static string? Run(string program, IEnumerable<string> arguments, IDictionary<string, string>? environment = null)
    {
        var start = new ProcessStartInfo(program) { RedirectStandardError = true, RedirectStandardOutput = true, UseShellExecute = false };
        foreach (var a in arguments)
            start.ArgumentList.Add(a);
        foreach (var (key, value) in environment ?? new Dictionary<string, string>())
            start.Environment[key] = value;
        try
        {
            using var process = Process.Start(start)!;
            var stderr = process.StandardError.ReadToEndAsync();
            process.StandardOutput.ReadToEnd();
            if (!process.WaitForExit(60_000))
            {
                process.Kill(entireProcessTree: true);
                return "took longer than a minute";
            }
            return process.ExitCode == 0 ? null : $"exit code {process.ExitCode}: {stderr.Result.Trim()}";
        }
        catch (Exception e) when (e is System.ComponentModel.Win32Exception or InvalidOperationException)
        {
            return e.Message;
        }
    }

    // Whether a program can be found on PATH (with Windows' executable extensions).
    internal static bool OnPath(string program)
    {
        var extensions = OperatingSystem.IsWindows() ? (Environment.GetEnvironmentVariable("PATHEXT") ?? ".EXE").Split(';') : [""];
        return (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)
            .Any(folder => extensions.Any(e => File.Exists(Path.Combine(folder, program + e))));
    }

    // Windows' own imaging (GDI+, through System.Drawing in Windows PowerShell 5.1, part of every Windows): JPEG, PNG,
    // GIF, TIFF and BMP, with the photo's EXIF orientation applied. The paths go in environment variables, so no quoting
    // reaches the script.
    private sealed class WindowsImaging : IImageConverter
    {
        private const string Script = """
            $ErrorActionPreference = 'Stop'
            Add-Type -AssemblyName System.Drawing
            $image = [System.Drawing.Image]::FromFile($env:IDRAK_IMAGE_IN)
            try {
                if ($image.PropertyIdList -contains 0x0112) {
                    switch ([int]$image.GetPropertyItem(0x0112).Value[0]) {
                        3 { $image.RotateFlip('Rotate180FlipNone') }
                        6 { $image.RotateFlip('Rotate90FlipNone') }
                        8 { $image.RotateFlip('Rotate270FlipNone') }
                    }
                }
                $bitmap = New-Object System.Drawing.Bitmap $image.Width, $image.Height, ([System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
                $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
                $graphics.Clear([System.Drawing.Color]::White)
                $graphics.DrawImage($image, 0, 0, $image.Width, $image.Height)
                $graphics.Dispose()
                $bitmap.Save($env:IDRAK_IMAGE_OUT, [System.Drawing.Imaging.ImageFormat]::Bmp)
                $bitmap.Dispose()
            } finally { $image.Dispose() }
            """;

        public string Name => "Windows imaging";

        public bool IsAvailable => OperatingSystem.IsWindows() && OnPath("powershell");

        public string? ConvertToBmp(string source, string bmp) =>
            Run("powershell", ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", Script],
                new Dictionary<string, string> { ["IDRAK_IMAGE_IN"] = source, ["IDRAK_IMAGE_OUT"] = bmp });
    }

    // A command-line converter: `arguments` gives its arguments for a source and a BMP path (each passed as one argument).
    private sealed class ExternalTool(string name, string program, Func<string, string, string[]> arguments) : IImageConverter
    {
        public string Name => name;

        public bool IsAvailable => OnPath(program);

        public string? ConvertToBmp(string source, string bmp) => Run(program, arguments(source, bmp));
    }
}
