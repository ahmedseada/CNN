using System.Diagnostics;
using System.Text;
using CnnSamples.Shared;
using DocumentOcr;
using Idrak;
using Idrak.Data;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;

// DocumentOcr: reads the text of a page of handwritten characters. A convolutional network trained on EMNIST Balanced
// (digits, capitals and 11 lower-case letters) recognises each character; the page reader finds the lines and
// characters and puts the text back together.
//
//   dotnet idrak @emnist.rsp                      download EMNIST with the Idrak CLI (once; LettersCnn's download works too)
//   dotnet run -- train   [--epochs 10] [--patience 3] [--batch 128] [--train-samples N] [--model ocr.ikm] [--data data]
//   dotnet run -- demo    [--text "LINE ONE\nLINE TWO"] [--out page.pgm]   writes a handwritten page, then reads it
//   dotnet run -- ocr     <page.png|bmp|pgm> [--model ocr.ikm]              reads a page of your own

int[] imageShape = [1, Characters.Rows, Characters.Columns];
const string DemoText = "HANDWRITTEN OCR WITH IDRAK\nTHE QUICK BROWN FOX JUMPS\nOVER 13 LAZY DOGS IN 2026";
var defaults = new SampleOptions(
    Command: "train", ImagePath: null, Epochs: 10, Patience: 3, BatchSize: 128, TrainSamples: null,
    ModelPath: "ocr.ikm", DataFolder: "data", ExportCount: 0, ExportFolder: "page.pgm");

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
        case "ocr":
            Ocr(options);
            break;
        case "demo":
            Demo(options);
            break;
        default:
            Console.Error.WriteLine($"Unknown command '{options.Command}'. Use 'train', 'demo' or 'ocr <page image>'.");
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

    var (train, test, _, testLabels, classes) = Characters.Load(o.DataFolder);
    if (o.TrainSamples is int n && n < train.Count)
        train = train.Subset(TestImages.RandomSubset(train.Count, n));
    Console.WriteLine($"EMNIST Balanced: {train.Count:N0} training and {test.Count:N0} test images, {classes.Length} characters: {string.Concat(classes)}");

    var network = Network.Image(channels: 1, height: Characters.Rows, width: Characters.Columns)
        .Seed(1)
        .Conv2d(32, kernelSize: 3, padding: 1).BatchNorm().ReLU()
        .Conv2d(32, kernelSize: 3, padding: 1).BatchNorm().ReLU().MaxPool2d(2)    // 32 x 14 x 14
        .Conv2d(64, kernelSize: 3, padding: 1).BatchNorm().ReLU()
        .Conv2d(64, kernelSize: 3, padding: 1).BatchNorm().ReLU().MaxPool2d(2)    // 64 x 7 x 7
        .Flatten()
        .Linear(256).ReLU().Dropout(0.4f)
        .Linear(classes.Length);
    using var model = network.Build();

    // Characters cut from a page are never centred as exactly as EMNIST's: train on slightly moved and turned copies.
    var batches = new DataLoader(train, o.BatchSize, shuffle: true, seed: 1)
    {
        Transforms = [new RandomShift(2), new RandomRotation(8)],
    };

    var stopwatch = Stopwatch.StartNew();
    var history = new TrainingRun
    {
        Model = model,
        Loss = (logits, targets) => Losses.CrossEntropy(logits, targets, 0.1f),
        Optimizer = p => new AdamW(p, 2e-3f, weightDecay: 1e-4f),
        Scheduler = optimizer => new CosineAnnealing(optimizer, totalEpochs: o.Epochs, minLearningRate: 1e-5f, warmupEpochs: 1),
        Train = batches,
        Validation = test.Batches(EvaluationBatchSize),
        Epochs = o.Epochs,
        Metrics = [Metric.Accuracy],
        EarlyStoppingPatience = o.Patience,
        OnEpoch = e =>
        {
            Console.WriteLine(
                $"epoch {e.Epoch}/{e.Epochs}  loss {e.Loss:F4}  acc {e.Metrics["accuracy"]:P2}  " +
                $"val loss {e.ValidationLoss:F4}  val acc {e.ValidationMetrics!["accuracy"]:P2}  " +
                $"lr {e.LearningRate:G3}  ({e.Duration.TotalSeconds:F1}s){(e.IsBest ? "  *" : "")}");
        },
    }.Fit();
    Console.WriteLine(
        $"Trained in {stopwatch.Elapsed.TotalSeconds:F1}s; kept the weights of epoch {history.BestEpoch} " +
        $"(val loss {history.BestLoss:F4}){(history.StoppedEarly ? ", stopped early" : "")}");

    using var predictor = Predictor.For(model)
        .InputShape(imageShape)
        .Softmax()
        .Classes(classes)
        .Build();

    var predicted = predictor.Predict(test).Select(p => p.Index).ToArray();
    ConsoleReport.PrintResults(testLabels, predicted, classes);

    predictor.Save(o.ModelPath);   // weights, architecture, input shape, softmax and classes: Predictor.Load and idrak predict read it
    Console.WriteLine($"Saved the model to {Path.GetFullPath(o.ModelPath)}");
    Console.WriteLine("Try it: dotnet run -c Release -- demo");
}

// Writes a page of handwriting for a known text, reads it back and compares.
void Demo(SampleOptions o)
{
    string text = (o.Text ?? DemoText).Replace("\\n", "\n");
    var (_, _, testImages, testLabels, classes) = Characters.Load(o.DataFolder);
    var (pixels, width, height) = PageWriter.Write(text, testImages, testLabels, classes);
    string page = o.ExportFolder;
    ImageFiles.WritePgm(page, pixels, height, width);
    Console.WriteLine($"Wrote a {width} x {height} page of handwriting to {Path.GetFullPath(page)}");

    var (raw, read) = ReadPage(page, o.ModelPath, show: false);
    Console.WriteLine();
    Console.WriteLine("Expected:");
    Console.WriteLine(text);
    Console.WriteLine();
    Console.WriteLine("Read, character by character:");
    Console.WriteLine(raw);
    Console.WriteLine();
    Console.WriteLine("Read, with each word's letters or digits:");
    Console.WriteLine(read);

    string expected = Comparable(text);
    Console.WriteLine();
    foreach (var (name, result) in new[] { ("character by character", raw), ("with words", read) })
    {
        int errors = Levenshtein(expected, Comparable(result));
        Console.WriteLine($"Character error rate, {name}: {errors / (double)Math.Max(expected.Length, 1):P1} ({errors} edit{(errors == 1 ? "" : "s")} over {expected.Length} characters)");
    }
}

void Ocr(SampleOptions o)
{
    if (o.ImagePath is null)
        throw new ArgumentException("ocr needs an image of a page: dotnet run -- ocr page.png (or run 'demo' to make one)");
    Console.WriteLine(ReadPage(o.ImagePath, o.ModelPath, show: true).Text);
}

// The text of a page image: its characters, recognised in one batch, with spaces and line breaks; as read character by
// character (Raw), and with each word's characters kept to letters or to digits (Text).
(string Raw, string Text) ReadPage(string path, string modelPath, bool show)
{
    if (!File.Exists(modelPath))
        throw new FileNotFoundException($"No model at '{modelPath}'. Run 'dotnet run -- train' first.");
    var saved = Predictor.Load(modelPath);
    using var predictor = saved.Classes(saved.StoredClasses!).Build();

    var image = ImageCodecs.Decode(path);
    var grey = image.Resize(1, image.Height, image.Width);   // colour averaged to grey, same size
    var glyphs = PageReader.Read(grey, image.Width, image.Height);
    Console.WriteLine($"Found {glyphs.Count} characters on {glyphs.Select(g => g.Line).Distinct().Count()} lines");

    var answers = predictor.Predict([.. glyphs.Select(g => g.Image)]);
    var raw = new StringBuilder();
    var text = new StringBuilder();
    foreach (var word in Words(glyphs))
    {
        if (word[0] > 0)
        {
            char separator = glyphs[word[0]].Line != glyphs[word[0] - 1].Line ? '\n' : ' ';
            raw.Append(separator);
            text.Append(separator);
        }

        foreach (int i in word)
            raw.Append(answers[i].Class);
        text.Append(InContext([.. word.Select(i => answers[i])]));
    }

    if (show)
    {
        Console.WriteLine("Least certain characters:");
        foreach (var (g, a) in glyphs.Zip(answers).OrderBy(p => p.Second.Probability).Take(3))
        {
            ConsoleReport.PrintImage(g.Image, Characters.Rows, Characters.Columns);
            Console.WriteLine($"line {g.Line + 1} at x {g.Left}: {a.Class} ({a.Probability:P0}), else {string.Join(", ", a.Scores.Skip(1).Take(2).Select(s => $"{s.Class} {s.Score:P0}"))}");
        }
        Console.WriteLine();
    }
    return (raw.ToString(), text.ToString());
}

// A word read with its context. In a word of mostly letters, a digit becomes the letter it looks like (1 -> I, 0 -> O,
// 5 -> S...), or else the model's likeliest letter; in a word of mostly digits, the other way round. A half-and-half
// word keeps what the model said. Then the word takes one case: most of its letters' (EMNIST Balanced has separate
// classes for a b d e f g h n q r t, so "tHE" and "fOX" happen).
static string InContext(IReadOnlyList<ClassPrediction> word)
{
    int digits = word.Count(a => char.IsDigit(a.Class[0]));
    var chars = word.Select(a => a.Class[0]).ToArray();
    for (int i = 0; i < chars.Length; i++)
    {
        if (digits * 2 < word.Count && char.IsDigit(chars[i]))
            chars[i] = LooksLikeLetter.TryGetValue(chars[i], out char letter) ? letter : word[i].Scores.First(s => char.IsLetter(s.Class[0])).Class[0];
        else if (digits * 2 > word.Count && char.IsLetter(chars[i]))
            chars[i] = LooksLikeDigit.TryGetValue(chars[i], out char digit) ? digit : word[i].Scores.First(s => char.IsDigit(s.Class[0])).Class[0];
    }

    var letters = chars.Where(char.IsLetter).ToArray();
    bool upper = letters.Count(char.IsUpper) * 2 >= letters.Length;
    return new string([.. chars.Select(c => upper ? char.ToUpperInvariant(c) : char.ToLowerInvariant(c))]);
}

// The glyphs' indices grouped into words: a new word at a space or a new line.
static List<List<int>> Words(List<Glyph> glyphs)
{
    var words = new List<List<int>>();
    for (int i = 0; i < glyphs.Count; i++)
    {
        if (i == 0 || glyphs[i].SpaceBefore || glyphs[i].Line != glyphs[i - 1].Line)
            words.Add([]);
        words[^1].Add(i);
    }
    return words;
}

// Upper case, single spaces: how the comparison sees a text (EMNIST Balanced merges c/C, o/O, s/S...).
static string Comparable(string text) =>
    string.Join('\n', text.Replace("\r", "").Split('\n').Select(l => string.Join(' ', l.ToUpperInvariant().Split(' ', StringSplitOptions.RemoveEmptyEntries))));

static int Levenshtein(string a, string b)
{
    var previous = Enumerable.Range(0, b.Length + 1).ToArray();
    for (int i = 1; i <= a.Length; i++)
    {
        var current = new int[b.Length + 1];
        current[0] = i;
        for (int j = 1; j <= b.Length; j++)
            current[j] = Math.Min(Math.Min(current[j - 1] + 1, previous[j] + 1), previous[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1));
        previous = current;
    }
    return previous[b.Length];
}

// Characters that handwriting makes hard to tell apart, for words whose kind (letters or digits) is known.
internal static partial class Program
{
    private static readonly Dictionary<char, char> LooksLikeLetter = new()
    {
        ['0'] = 'O', ['1'] = 'I', ['2'] = 'Z', ['5'] = 'S', ['6'] = 'G', ['8'] = 'B',
    };

    private static readonly Dictionary<char, char> LooksLikeDigit = new()
    {
        ['O'] = '0', ['D'] = '0', ['I'] = '1', ['L'] = '1', ['Z'] = '2', ['S'] = '5', ['G'] = '6', ['b'] = '6',
        ['B'] = '8', ['g'] = '9', ['q'] = '9',
    };
}
