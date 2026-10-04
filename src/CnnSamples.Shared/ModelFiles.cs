using System.Text;
using System.Text.Json.Nodes;
using Idrak.Inference;
using Idrak.Layers;

namespace CnnSamples.Shared;

/// <summary>Model packages and image files shared by the samples.</summary>
public static class ModelFiles
{
    /// <summary>
    /// Saves <paramref name="predictor"/> as a package that works in <c>Predictor.Load</c> and in <c>idrak predict</c>.
    /// </summary>
    /// <remarks>
    /// predictor.Save writes the "predictor" entry that Predictor.Load reads. Idrak 0.2.0's <c>idrak predict</c> reads
    /// only a "training" entry (the one <c>idrak train</c> writes): the task, the class names and the image shape. This
    /// rewrites the package with both (0.2.1 reads the predictor entry too: https://github.com/ahmedseada/Idrak/pull/1).
    /// </remarks>
    public static void Save<TIn, TOut>(Predictor<TIn, TOut> predictor, string path, NetworkBuilder network, Module model,
        IReadOnlyList<string> classes, int[] inputShape, double testAccuracy)
    {
        predictor.Save(path);

        JsonNode predictorSettings;
        using (var saved = ModelPackage.Open(path))
            predictorSettings = saved.Json("predictor").DeepClone();

        ModelPackage.Create(path)
            .Architecture(network)
            .Weights(model)
            .Json("predictor", predictorSettings)
            .Json("training", new JsonObject
            {
                ["format"] = "idrak-train/1",
                ["task"] = "classification",
                ["input"] = "images",
                ["targets"] = new JsonArray("class"),
                ["classes"] = new JsonArray([.. classes.Select(c => JsonValue.Create(c))]),
                ["inputShape"] = new JsonArray([.. inputShape.Select(d => JsonValue.Create(d))]),
                ["metrics"] = new JsonObject { ["accuracy"] = testAccuracy },
            })
            .Save();
    }

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
