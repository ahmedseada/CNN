using System.Diagnostics;
using CnnSamples.Shared;
using Idrak;
using Idrak.Data;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;
using LettersCnn;

// LettersCnn: a convolutional network that reads handwritten English letters A-Z (EMNIST Letters),
// written with Idrak's fluent API (network builder, data extensions, TrainingRun, predictor builder).
//
//   dotnet idrak @emnist.rsp                      download EMNIST with the Idrak CLI (once, about 560 MB)
//   dotnet run -- train   [--epochs 8] [--patience 3] [--batch 128] [--train-samples N] [--model letters.ikm] [--data data]
//   dotnet run -- predict <image.png|bmp|pgm> [--model letters.ikm]
//   dotnet run -- export  [--count 26] [--out test-letters]   test images as PGM files, e.g. for `idrak predict`

int[] imageShape = [1, Emnist.Rows, Emnist.Columns];
var defaults = new SampleOptions(
    Command: "train", ImagePath: null, Epochs: 8, Patience: 3, BatchSize: 128, TrainSamples: null,
    ModelPath: "letters.ikm", DataFolder: "data", ExportCount: 26, ExportFolder: "test-letters");

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

    var (train, test, testLabels) = Emnist.Load(o.DataFolder);
    if (o.TrainSamples is int n && n < train.Count)
        train = train.Subset(TestImages.RandomSubset(train.Count, n));   // a random N, so every class is in it
    Console.WriteLine($"EMNIST Letters: {train.Count:N0} training and {test.Count:N0} test images");

    // Two blocks of two convolutions with batch normalization, then a classifier head with one logit per letter.
    var network = Network.Image(channels: 1, height: Emnist.Rows, width: Emnist.Columns)
        .Seed(1)
        .Conv2d(32, kernelSize: 3, padding: 1).BatchNorm().ReLU()
        .Conv2d(32, kernelSize: 3, padding: 1).BatchNorm().ReLU().MaxPool2d(2)    // 32 x 14 x 14
        .Conv2d(64, kernelSize: 3, padding: 1).BatchNorm().ReLU()
        .Conv2d(64, kernelSize: 3, padding: 1).BatchNorm().ReLU().MaxPool2d(2)    // 64 x 7 x 7
        .Flatten()
        .Linear(256).ReLU().Dropout(0.4f)
        .Linear(Emnist.Letters.Length);
    using var model = network.Build();

    using var best = new BestWeights(model);
    var stopwatch = Stopwatch.StartNew();
    var history = new TrainingRun
    {
        Model = model,
        Loss = (logits, targets) => Losses.CrossEntropy(logits, targets, 0.1f),   // label smoothing 0.1
        Optimizer = p => new AdamW(p, 2e-3f, weightDecay: 1e-4f),
        Scheduler = optimizer => new CosineAnnealing(optimizer, totalEpochs: o.Epochs, minLearningRate: 1e-5f, warmupEpochs: 1),
        Train = train.Batches(o.BatchSize, shuffle: true, seed: 1),
        Validation = test.Batches(EvaluationBatchSize),
        Epochs = o.Epochs,
        Metrics = [Metric.Accuracy],
        EarlyStoppingPatience = o.Patience,   // stop after this many epochs without a lower validation loss
        OnEpoch = e =>
        {
            Console.WriteLine(
                $"epoch {e.Epoch}/{e.Epochs}  loss {e.Loss:F4}  acc {e.Metrics["accuracy"]:P2}  " +
                $"val loss {e.ValidationLoss:F4}  val acc {e.ValidationMetrics!["accuracy"]:P2}  " +
                $"lr {e.LearningRate:G3}  ({e.Duration.TotalSeconds:F1}s){(e.IsBest ? "  *" : "")}");
            best.Track(e);
        },
    }.Fit();
    best.Restore();
    Console.WriteLine(
        $"Trained in {stopwatch.Elapsed.TotalSeconds:F1}s; kept the weights of epoch {best.Epoch} " +
        $"(val loss {history.BestLoss:F4}){(history.StoppedEarly ? ", stopped early" : "")}");

    using var predictor = Predictor.For(model)
        .InputShape(imageShape)
        .BatchSize(EvaluationBatchSize)
        .Softmax()
        .Classes(Emnist.Letters)
        .Build();

    var predicted = predictor.Predict(test).Select(p => p.Index).ToArray();
    double accuracy = ConsoleReport.PrintResults(testLabels, predicted, Emnist.Letters);

    Console.WriteLine();
    Console.WriteLine("A few test letters:");
    foreach (var i in TestImages.OnePerClass(testLabels, Emnist.Letters.Length, 5))
    {
        var image = test.GetFeatures(i).ToArray();
        var answer = predictor.Predict(image);
        ConsoleReport.PrintImage(image, Emnist.Rows, Emnist.Columns);
        Console.WriteLine($"label {Emnist.Letters[testLabels[i]]}  predicted {answer.Class} ({answer.Probability:P1})");
        Console.WriteLine();
    }

    ModelFiles.Save(predictor, o.ModelPath, network, model, Emnist.Letters, imageShape, accuracy);
    Console.WriteLine($"Saved the model to {Path.GetFullPath(o.ModelPath)}");
    Console.WriteLine($"Run it with the Idrak CLI: dotnet idrak predict {o.ModelPath} -i <image or folder> --top 3");
}

void Predict(SampleOptions o)
{
    if (o.ImagePath is null)
        throw new ArgumentException("predict needs an image path: dotnet run -- predict letter.png");
    if (!File.Exists(o.ModelPath))
        throw new FileNotFoundException($"No model at '{o.ModelPath}'. Run 'dotnet run -- train' first.");

    var saved = Predictor.Load(o.ModelPath);   // architecture, weights, input shape, softmax and classes
    using var predictor = saved.Classes(saved.StoredClasses ?? Emnist.Letters).Build();

    var pixels = ModelFiles.LoadLightOnDark(o.ImagePath, Emnist.Rows, Emnist.Columns);
    ConsoleReport.PrintImage(pixels, Emnist.Rows, Emnist.Columns);
    var answer = predictor.Predict(pixels);
    Console.WriteLine($"Predicted letter: {answer.Class} ({answer.Probability:P1})");
    foreach (var score in answer.Scores.Take(3))
        Console.WriteLine($"  {score.Class}: {score.Score:P1}");
}

// Writes test images, one of each letter in turn and upright, as 28 x 28 PGM files named <index>_label<letter>.pgm.
void Export(SampleOptions o)
{
    var (_, test, testLabels) = Emnist.Load(o.DataFolder);
    Directory.CreateDirectory(o.ExportFolder);
    var picked = TestImages.OnePerClass(testLabels, Emnist.Letters.Length, o.ExportCount);
    foreach (int i in picked)
        ModelFiles.WritePgm(Path.Combine(o.ExportFolder, $"{i:D5}_label{Emnist.Letters[testLabels[i]]}.pgm"),
            test.GetFeatures(i), Emnist.Rows, Emnist.Columns);
    Console.WriteLine($"Wrote {picked.Length} test images, one of each class in turn, to {Path.GetFullPath(o.ExportFolder)}");
}
