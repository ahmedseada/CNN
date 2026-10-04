namespace CnnSamples.Shared;

/// <summary>
/// The command line of a sample: <c>train</c> (the default), <c>predict IMAGE</c> or <c>export</c>, and their options.
/// </summary>
public sealed record SampleOptions(
    string Command, string? ImagePath, int Epochs, int Patience, int BatchSize, int? TrainSamples, string ModelPath,
    string DataFolder, int ExportCount, string ExportFolder)
{
    /// <summary>Parses <paramref name="args"/> over <paramref name="defaults"/>.</summary>
    /// <exception cref="ArgumentException">An unknown option, or an option without its value.</exception>
    public static SampleOptions Parse(string[] args, SampleOptions defaults)
    {
        var o = defaults;
        int i = 0;
        if (args.Length > 0 && !args[0].StartsWith("--"))
            o = o with { Command = args[i++].ToLowerInvariant() };
        if (o.Command == "predict" && i < args.Length && !args[i].StartsWith("--"))
            o = o with { ImagePath = args[i++] };

        for (; i < args.Length; i++)
        {
            string Value() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value.");
            int Number()
            {
                string option = args[i], value = Value();
                return int.TryParse(value, out int n) && n > 0 ? n : throw new ArgumentException($"{option} needs a positive number, not '{value}'.");
            }

            o = args[i] switch
            {
                "--epochs" => o with { Epochs = Number() },
                "--patience" => o with { Patience = Number() },
                "--batch" => o with { BatchSize = Number() },
                "--train-samples" => o with { TrainSamples = Number() },
                "--model" => o with { ModelPath = Value() },
                "--data" => o with { DataFolder = Value() },
                "--count" => o with { ExportCount = Number() },
                "--out" => o with { ExportFolder = Value() },
                _ => throw new ArgumentException($"Unknown option {args[i]}."),
            };
        }
        return o;
    }
}
