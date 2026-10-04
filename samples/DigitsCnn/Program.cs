using System.Diagnostics;
using DigitsCnn;
using Idrak;
using Idrak.Data;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;

// DigitsCnn: a small convolutional network that reads handwritten digits 0-9 (MNIST),
// written with Idrak's fluent API (network builder, data extensions, TrainingRun, predictor builder).
//
//   dotnet idrak @mnist.rsp                       download MNIST with the Idrak CLI (once)
//   dotnet run -- train   [--epochs 3] [--batch 64] [--train-samples 60000] [--model digits.ikm] [--data data]
//   dotnet run -- predict <image.png|bmp|pgm> [--model digits.ikm]

Options options;
try
{
    options = Options.Parse(args);
}
catch (Exception e) when (e is ArgumentException or FormatException)
{
    Console.Error.WriteLine(e.Message);
    return 1;
}
Console.WriteLine($"Device: {Device.Default}");

try
{
    switch (options.Command)
    {
        case "train":
            Train(options);
            break;
        case "predict":
            Predict(options);
            break;
        default:
            Console.Error.WriteLine($"Unknown command '{options.Command}'. Use 'train' or 'predict <image>'.");
            return 1;
    }
}
catch (Exception e) when (e is IOException or ArgumentException or InvalidDataException)
{
    Console.Error.WriteLine(e.Message);
    return 1;
}
return 0;

static void Train(Options o)
{
    var (train, test, testLabels) = Mnist.Load(o.DataFolder);
    if (o.TrainSamples is int n && n < train.Count)
        train = train.Subset([.. Enumerable.Range(0, n)]);
    Console.WriteLine($"MNIST: {train.Count:N0} training and {test.Count:N0} test images");

    // Two convolution blocks, then a small classifier head with one logit per digit.
    using var model = Network.Image(channels: 1, height: Mnist.Rows, width: Mnist.Columns)
        .Seed(1)
        .Conv2d(32, kernelSize: 3, padding: 1).ReLU().MaxPool2d(2)    // 32 x 14 x 14
        .Conv2d(64, kernelSize: 3, padding: 1).ReLU().MaxPool2d(2)    // 64 x 7 x 7
        .Flatten()
        .Linear(128).ReLU().Dropout(0.25f)
        .Linear(Mnist.Digits.Length)
        .Build();

    var stopwatch = Stopwatch.StartNew();
    var history = new TrainingRun
    {
        Model = model,
        Loss = Losses.CrossEntropy,
        Optimizer = p => new Adam(p, 1e-3f),
        Train = train.Batches(o.BatchSize, shuffle: true, seed: 1),
        Validation = test.Batches(512),
        Epochs = o.Epochs,
        Metrics = [Metric.Accuracy],
        OnEpoch = e => Console.WriteLine(
            $"epoch {e.Epoch}/{e.Epochs}  loss {e.Loss:F4}  acc {e.Metrics["accuracy"]:P2}  " +
            $"val loss {e.ValidationLoss:F4}  val acc {e.ValidationMetrics!["accuracy"]:P2}  ({e.Duration.TotalSeconds:F1}s)"),
    }.Fit();
    Console.WriteLine($"Trained in {stopwatch.Elapsed.TotalSeconds:F1}s, best epoch {history.BestEpoch}");

    // One image in, the most likely digit (with every digit's probability) out.
    using var predictor = Predictor.For(model)
        .InputShape(1, Mnist.Rows, Mnist.Columns)
        .Softmax()
        .Classes(Mnist.Digits)
        .Build();

    var predictions = predictor.Predict(test);
    PrintConfusion(testLabels, predictions.Select(p => p.Index).ToArray());

    Console.WriteLine();
    Console.WriteLine("A few test digits:");
    foreach (var i in new[] { 0, 1, 2, 3, 4 })
    {
        var image = test.GetFeatures(i).ToArray();
        var answer = predictor.Predict(image);
        PrintDigit(image);
        Console.WriteLine($"label {testLabels[i]}  predicted {answer.Class} ({answer.Probability:P1})");
        Console.WriteLine();
    }

    predictor.Save(o.ModelPath);
    Console.WriteLine($"Saved the model to {Path.GetFullPath(o.ModelPath)}");
}

static void Predict(Options o)
{
    if (o.ImagePath is null)
        throw new ArgumentException("predict needs an image path: dotnet run -- predict digit.png");
    if (!File.Exists(o.ModelPath))
        throw new FileNotFoundException($"No model at '{o.ModelPath}'. Run 'dotnet run -- train' first.");

    // The package restores the architecture, weights, input shape, softmax and class names.
    var saved = Predictor.Load(o.ModelPath);
    using var predictor = saved
        .Classes(saved.StoredClasses ?? Mnist.Digits)
        .Build();

    // MNIST digits are light strokes on a dark background; flip images drawn dark on light.
    var pixels = ImageCodecs.Load(o.ImagePath, channels: 1, height: Mnist.Rows, width: Mnist.Columns);
    if (pixels.Average() > 0.5f)
        for (int i = 0; i < pixels.Length; i++)
            pixels[i] = 1f - pixels[i];

    PrintDigit(pixels);
    var answer = predictor.Predict(pixels);
    Console.WriteLine($"Predicted digit: {answer.Class} ({answer.Probability:P1})");
    foreach (var score in answer.Scores.Take(3))
        Console.WriteLine($"  {score.Class}: {score.Score:P1}");
}

static void PrintDigit(ReadOnlySpan<float> pixels)
{
    const string shades = " .:-=+*#%@";
    for (int y = 0; y < Mnist.Rows; y += 2)   // every other row keeps the aspect ratio in a terminal
    {
        var line = new char[Mnist.Columns];
        for (int x = 0; x < Mnist.Columns; x++)
        {
            float v = Math.Max(pixels[y * Mnist.Columns + x], pixels[(y + 1) * Mnist.Columns + x]);
            line[x] = shades[(int)(Math.Clamp(v, 0f, 1f) * (shades.Length - 1))];
        }
        Console.WriteLine(new string(line));
    }
}

static void PrintConfusion(int[] labels, int[] predicted)
{
    var digits = Mnist.Digits;
    var confusion = new int[digits.Length, digits.Length];
    for (int i = 0; i < labels.Length; i++)
        confusion[labels[i], predicted[i]]++;

    int correct = 0;
    Console.WriteLine();
    Console.WriteLine("Confusion matrix (rows: true digit, columns: predicted)");
    Console.WriteLine("      " + string.Concat(digits.Select(d => $"{d,6}")) + "   accuracy");
    for (int t = 0; t < digits.Length; t++)
    {
        int rowTotal = 0;
        Console.Write($"{digits[t],6}");
        for (int p = 0; p < digits.Length; p++)
        {
            Console.Write($"{confusion[t, p],6}");
            rowTotal += confusion[t, p];
        }
        correct += confusion[t, t];
        Console.WriteLine($"   {(double)confusion[t, t] / Math.Max(rowTotal, 1):P1}");
    }
    Console.WriteLine($"Test accuracy: {(double)correct / labels.Length:P2} ({correct:N0}/{labels.Length:N0})");
}

internal sealed record Options(
    string Command, string? ImagePath, int Epochs, int BatchSize, int? TrainSamples, string ModelPath, string DataFolder)
{
    public static Options Parse(string[] args)
    {
        var o = new Options("train", null, 3, 64, null, "digits.ikm", "data");
        int i = 0;
        if (args.Length > 0 && !args[0].StartsWith("--"))
            o = o with { Command = args[i++].ToLowerInvariant() };
        if (o.Command == "predict" && i < args.Length && !args[i].StartsWith("--"))
            o = o with { ImagePath = args[i++] };

        for (; i < args.Length; i++)
        {
            string Value() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value.");
            o = args[i] switch
            {
                "--epochs" => o with { Epochs = int.Parse(Value()) },
                "--batch" => o with { BatchSize = int.Parse(Value()) },
                "--train-samples" => o with { TrainSamples = int.Parse(Value()) },
                "--model" => o with { ModelPath = Value() },
                "--data" => o with { DataFolder = Value() },
                _ => throw new ArgumentException($"Unknown option {args[i]}."),
            };
        }
        return o;
    }
}
