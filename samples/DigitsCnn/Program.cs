using System.Diagnostics;
using CnnSamples.Shared;
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
//   dotnet run -- train   [--epochs 3] [--patience 2] [--batch 64] [--train-samples 60000] [--model digits.ikm] [--data data]
//   dotnet run -- predict <image.png|bmp|pgm> [--model digits.ikm]
//   dotnet run -- export  [--count 20] [--out test-digits]   test images as PGM files, e.g. for `idrak predict`

int[] imageShape = [1, Mnist.Rows, Mnist.Columns];
var defaults = new SampleOptions(
    Command: "train", ImagePath: null, Epochs: 3, Patience: 2, BatchSize: 64, TrainSamples: null,
    ModelPath: "digits.ikm", DataFolder: "data", ExportCount: 20, ExportFolder: "test-digits");

SampleOptions options;
try
{
    options = SampleOptions.Parse(args, defaults);
}
catch (ArgumentException e)
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
        case "export":
            Export(options);
            break;
        default:
            Console.Error.WriteLine($"Unknown command '{options.Command}'. Use 'train', 'predict <image>' or 'export'.");
            return 1;
    }
}
catch (Exception e) when (e is IOException or ArgumentException or InvalidDataException)
{
    Console.Error.WriteLine(e.Message);
    return 1;
}
return 0;

void Train(SampleOptions o)
{
    const int EvaluationBatchSize = 512;

    var (train, test, testLabels) = Mnist.Load(o.DataFolder);
    if (o.TrainSamples is int n && n < train.Count)
        train = train.Subset(TestImages.RandomSubset(train.Count, n));   // a random N, so every class is in it
    Console.WriteLine($"MNIST: {train.Count:N0} training and {test.Count:N0} test images");

    // Two convolution blocks, then a small classifier head with one logit per digit.
    var network = Network.Image(channels: 1, height: Mnist.Rows, width: Mnist.Columns)
        .Seed(1)
        .Conv2d(32, kernelSize: 3, padding: 1).ReLU().MaxPool2d(2)    // 32 x 14 x 14
        .Conv2d(64, kernelSize: 3, padding: 1).ReLU().MaxPool2d(2)    // 64 x 7 x 7
        .Flatten()
        .Linear(128).ReLU().Dropout(0.25f)
        .Linear(Mnist.Digits.Length);
    using var model = network.Build();

    using var best = new BestWeights(model);
    var stopwatch = Stopwatch.StartNew();
    var history = new TrainingRun
    {
        Model = model,
        Loss = Losses.CrossEntropy,
        Optimizer = p => new Adam(p, 1e-3f),
        Train = train.Batches(o.BatchSize, shuffle: true, seed: 1),
        Validation = test.Batches(EvaluationBatchSize),
        Epochs = o.Epochs,
        Metrics = [Metric.Accuracy],
        EarlyStoppingPatience = o.Patience,    // stop after this many epochs without a lower validation loss
        OnEpoch = e =>
        {
            Console.WriteLine(
                $"epoch {e.Epoch}/{e.Epochs}  loss {e.Loss:F4}  acc {e.Metrics["accuracy"]:P2}  " +
                $"val loss {e.ValidationLoss:F4}  val acc {e.ValidationMetrics!["accuracy"]:P2}  " +
                $"({e.Duration.TotalSeconds:F1}s){(e.IsBest ? "  *" : "")}");
            best.Track(e);
        },
    }.Fit();
    best.Restore();
    Console.WriteLine(
        $"Trained in {stopwatch.Elapsed.TotalSeconds:F1}s; kept the weights of epoch {best.Epoch} " +
        $"(val loss {history.BestLoss:F4}){(history.StoppedEarly ? ", stopped early" : "")}");

    // One image in, the most likely digit (with every digit's probability) out. BatchSize splits large inputs
    // (the 10,000 test images below) into GPU-sized batches; one 10,000-image batch fails to launch on CUDA in 0.2.0.
    using var predictor = Predictor.For(model)
        .InputShape(imageShape)
        .BatchSize(EvaluationBatchSize)
        .Softmax()
        .Classes(Mnist.Digits)
        .Build();

    var predicted = predictor.Predict(test).Select(p => p.Index).ToArray();
    double accuracy = ConsoleReport.PrintResults(testLabels, predicted, Mnist.Digits);

    Console.WriteLine();
    Console.WriteLine("A few test digits:");
    foreach (var i in TestImages.OnePerClass(testLabels, Mnist.Digits.Length, 5))
    {
        var image = test.GetFeatures(i).ToArray();
        var answer = predictor.Predict(image);
        ConsoleReport.PrintImage(image, Mnist.Rows, Mnist.Columns);
        Console.WriteLine($"label {testLabels[i]}  predicted {answer.Class} ({answer.Probability:P1})");
        Console.WriteLine();
    }

    ModelFiles.Save(predictor, o.ModelPath, network, model, Mnist.Digits, imageShape, accuracy);
    Console.WriteLine($"Saved the model to {Path.GetFullPath(o.ModelPath)}");
    Console.WriteLine($"Run it with the Idrak CLI: dotnet idrak predict {o.ModelPath} -i <image or folder> --top 3");
}

void Predict(SampleOptions o)
{
    if (o.ImagePath is null)
        throw new ArgumentException("predict needs an image path: dotnet run -- predict digit.png");
    if (!File.Exists(o.ModelPath))
        throw new FileNotFoundException($"No model at '{o.ModelPath}'. Run 'dotnet run -- train' first.");

    var saved = Predictor.Load(o.ModelPath);   // architecture, weights, input shape, softmax and classes
    using var predictor = saved.Classes(saved.StoredClasses ?? Mnist.Digits).Build();

    var pixels = ModelFiles.LoadLightOnDark(o.ImagePath, Mnist.Rows, Mnist.Columns);
    ConsoleReport.PrintImage(pixels, Mnist.Rows, Mnist.Columns);
    var answer = predictor.Predict(pixels);
    Console.WriteLine($"Predicted digit: {answer.Class} ({answer.Probability:P1})");
    foreach (var score in answer.Scores.Take(3))
        Console.WriteLine($"  {score.Class}: {score.Score:P1}");
}

// Writes test images, one of each digit in turn, as 28 x 28 PGM files named <index>_label<digit>.pgm.
void Export(SampleOptions o)
{
    var (_, test, testLabels) = Mnist.Load(o.DataFolder);
    Directory.CreateDirectory(o.ExportFolder);
    var picked = TestImages.OnePerClass(testLabels, Mnist.Digits.Length, o.ExportCount);
    foreach (int i in picked)
        ModelFiles.WritePgm(Path.Combine(o.ExportFolder, $"{i:D5}_label{testLabels[i]}.pgm"), test.GetFeatures(i), Mnist.Rows, Mnist.Columns);
    Console.WriteLine($"Wrote {picked.Length} test images, one of each class in turn, to {Path.GetFullPath(o.ExportFolder)}");
}
