using System.Diagnostics;
using DigitsCnn;
using Idrak;
using Idrak.Data;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;

// DigitsCnn: a small convolutional network that reads handwritten digits 0-9 (MNIST).
//
//   dotnet run -- train   [--epochs 3] [--batch 64] [--train-samples 60000] [--model digits.ikm] [--data data/mnist]
//   dotnet run -- predict <image.png|bmp|pgm> [--model digits.ikm]

string[] digitNames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

var options = Options.Parse(args);
Console.WriteLine($"Device: {Device.Default}");

switch (options.Command)
{
    case "train":
        await TrainAsync(options);
        break;
    case "predict":
        Predict(options);
        break;
    default:
        Console.WriteLine($"Unknown command '{options.Command}'. Use 'train' or 'predict <image>'.");
        return 1;
}
return 0;

async Task TrainAsync(Options o)
{
    var (train, test) = await Mnist.LoadAsync(o.DataFolder);
    Console.WriteLine($"MNIST: {train.Count:N0} training and {test.Count:N0} test images");

    var trainSet = ToDataset(train, o.TrainSamples);
    var testSet = ToDataset(test, null);

    // Two convolution blocks, then a small classifier head that outputs one logit per digit.
    using var model = Network.Image(channels: 1, height: Mnist.Rows, width: Mnist.Columns)
        .Seed(1)
        .Conv2d(32, kernelSize: 3, padding: 1).ReLU().MaxPool2d(2)    // 32 x 14 x 14
        .Conv2d(64, kernelSize: 3, padding: 1).ReLU().MaxPool2d(2)    // 64 x 7 x 7
        .Flatten()
        .Linear(128).ReLU().Dropout(0.25f)
        .Linear(digitNames.Length)
        .Build();
    Console.WriteLine(model);

    var stopwatch = Stopwatch.StartNew();
    var history = new TrainingRun
    {
        Model = model,
        Loss = Losses.CrossEntropy,
        Optimizer = p => new Adam(p, 1e-3f),
        Train = trainSet.Batches(o.BatchSize, shuffle: true),
        Validation = testSet.Batches(512),
        Epochs = o.Epochs,
        Metrics = [Metric.Accuracy],
        OnEpoch = e => Console.WriteLine(
            $"epoch {e.Epoch}/{e.Epochs}  loss {e.Loss:F4}  acc {e.Metrics["accuracy"]:P2}  " +
            $"val loss {e.ValidationLoss:F4}  val acc {e.ValidationMetrics!["accuracy"]:P2}  ({e.Duration.TotalSeconds:F1}s)"),
    }.Fit();
    Console.WriteLine($"Trained in {stopwatch.Elapsed.TotalSeconds:F1}s, best epoch {history.BestEpoch}");

    using var predictor = Predictor.For(model)
        .InputShape(1, Mnist.Rows, Mnist.Columns)
        .Softmax()
        .Classes(digitNames)
        .Build();

    var confusion = new int[digitNames.Length, digitNames.Length];
    var predictions = predictor.Predict(testSet);
    for (int i = 0; i < predictions.Count; i++)
        confusion[test.Labels[i], predictions[i].Index]++;
    PrintConfusion(confusion);

    Console.WriteLine();
    Console.WriteLine("A few test digits:");
    foreach (var i in new[] { 0, 1, 2, 3, 4 })
    {
        var image = Row(test.Images, i);
        var answer = predictor.Predict(image);
        PrintDigit(image);
        Console.WriteLine($"label {test.Labels[i]}  predicted {answer.Class} ({answer.Probability:P1})");
        Console.WriteLine();
    }

    predictor.Save(o.ModelPath);
    Console.WriteLine($"Saved the model to {Path.GetFullPath(o.ModelPath)}");
}

void Predict(Options o)
{
    if (o.ImagePath is null)
        throw new ArgumentException("predict needs an image path: dotnet run -- predict digit.png");
    if (!File.Exists(o.ModelPath))
        throw new FileNotFoundException($"No model at '{o.ModelPath}'. Run 'dotnet run -- train' first.");

    var saved = Predictor.Load(o.ModelPath);   // architecture, weights, input shape, softmax and classes
    using var predictor = saved.Classes(saved.StoredClasses ?? digitNames).Build();

    // MNIST digits are light strokes on a dark background; flip images drawn dark on light.
    var pixels = ImageCodecs.Load(o.ImagePath, 1, Mnist.Rows, Mnist.Columns);
    if (pixels.Average() > 0.5f)
        for (int i = 0; i < pixels.Length; i++)
            pixels[i] = 1f - pixels[i];

    PrintDigit(pixels);
    var answer = predictor.Predict(pixels);
    Console.WriteLine($"Predicted digit: {answer.Class} ({answer.Probability:P1})");
    foreach (var score in answer.Scores.Take(3))
        Console.WriteLine($"  {score.Class}: {score.Score:P1}");
}

Dataset ToDataset(Mnist.Split split, int? limit)
{
    int count = Math.Min(limit ?? split.Count, split.Count);
    var images = split.Images;
    if (count < split.Count)
    {
        images = new float[count, Mnist.Pixels];
        Buffer.BlockCopy(split.Images, 0, images, 0, count * Mnist.Pixels * sizeof(float));
    }
    return Dataset.FromClassLabels(images, split.Labels.AsSpan(0, count), digitNames.Length, digitNames)
        .WithFeatureShape(1, Mnist.Rows, Mnist.Columns);
}

static float[] Row(float[,] images, int index)
{
    var row = new float[Mnist.Pixels];
    Buffer.BlockCopy(images, index * Mnist.Pixels * sizeof(float), row, 0, Mnist.Pixels * sizeof(float));
    return row;
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

void PrintConfusion(int[,] confusion)
{
    int correct = 0, total = 0;
    Console.WriteLine();
    Console.WriteLine("Confusion matrix (rows: true digit, columns: predicted)");
    Console.WriteLine("      " + string.Concat(digitNames.Select(d => $"{d,6}")) + "   accuracy");
    for (int t = 0; t < digitNames.Length; t++)
    {
        int rowTotal = 0;
        Console.Write($"{digitNames[t],6}");
        for (int p = 0; p < digitNames.Length; p++)
        {
            Console.Write($"{confusion[t, p],6}");
            rowTotal += confusion[t, p];
        }
        correct += confusion[t, t];
        total += rowTotal;
        Console.WriteLine($"   {(double)confusion[t, t] / Math.Max(rowTotal, 1):P1}");
    }
    Console.WriteLine($"Test accuracy: {(double)correct / total:P2} ({correct:N0}/{total:N0})");
}

internal sealed record Options(
    string Command, string? ImagePath, int Epochs, int BatchSize, int? TrainSamples, string ModelPath, string DataFolder)
{
    public static Options Parse(string[] args)
    {
        var o = new Options("train", null, 3, 64, null, "digits.ikm", Path.Combine("data", "mnist"));
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
