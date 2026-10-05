using System.Diagnostics;
using System.Text;
using CnnSamples.Shared;
using Idrak;
using Idrak.Data;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;
using Idrak.Vision;
using MultiLanguageOcr;

// MultiLanguageOcr: reads pages of handwritten characters in English and Arabic. One convolutional network knows 85
// characters (EMNIST Balanced's digits and letters, AHCD's 28 Arabic letters, MADBase's Arabic-Indic digits); each line
// takes the script most of its characters look like, every character is read within that script, and Arabic lines are
// put back in reading order (words right to left, numbers left to right).
//
//   dotnet idrak @emnist.rsp        EMNIST (or reuse DocumentOcr's or LettersCnn's download)
//   dotnet idrak @arabic.rsp        AHCD and MADBase from Kaggle (needs a Kaggle API token)
//   dotnet run -- train   [--epochs 12] [--patience 3] [--batch 128] [--train-samples N] [--model multilang.ikm] [--data data]
//   dotnet run -- demo    [--text "HELLO\nمرحبا"] [--out page.pgm]    writes a page in both scripts, then reads it
//   dotnet run -- ocr     <page.png|bmp|pgm> [--model multilang.ikm]

Console.OutputEncoding = Encoding.UTF8;
int[] imageShape = [1, Datasets.Size, Datasets.Size];
const string DemoText = "HELLO WORLD 2026\nمرحبا بالعالم\nنص عربي ١٢٣٤٥";
var defaults = new SampleOptions(
    Command: "train", ImagePath: null, Epochs: 12, Patience: 3, BatchSize: 128, TrainSamples: null,
    ModelPath: "multilang.ikm", DataFolder: "data", ExportCount: 0, ExportFolder: "page.pgm");

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
        case "demo":
            Demo(options);
            break;
        case "ocr":
            Ocr(options);
            break;
        default:
            Console.Error.WriteLine($"Unknown command '{options.Command}'. Use 'train', 'demo' or 'ocr <page image>'.");
            return 1;
    }
}
catch (Exception e) when (e is IOException or ArgumentException or InvalidDataException or InvalidOperationException)
{
    Console.Error.WriteLine(e.Message);
    return 1;
}
return 0;

void Train(SampleOptions o)
{
    var (trainPart, testPart, classes) = Datasets.Load(o.DataFolder, o.TrainSamples);
    var train = Datasets.ToDataset(trainPart, classes);
    var test = Datasets.ToDataset(testPart, classes);
    Console.WriteLine($"{train.Count:N0} training and {test.Count:N0} test images, {classes.Length} characters");

    // Wider than DocumentOcr's network: 85 characters in two scripts.
    var network = Network.Image(channels: 1, height: Datasets.Size, width: Datasets.Size)
        .Seed(1)
        .Conv2d(48, kernelSize: 3, padding: 1).BatchNorm().ReLU()
        .Conv2d(48, kernelSize: 3, padding: 1).BatchNorm().ReLU().MaxPool2d(2)    // 48 x 14 x 14
        .Conv2d(96, kernelSize: 3, padding: 1).BatchNorm().ReLU()
        .Conv2d(96, kernelSize: 3, padding: 1).BatchNorm().ReLU().MaxPool2d(2)    // 96 x 7 x 7
        .Flatten()
        .Linear(384).ReLU().Dropout(0.4f)
        .Linear(classes.Length);
    using var model = network.Build();

    // Characters cut from a page are never centred exactly: train on slightly moved and turned copies.
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
        Validation = test.Batches(512),
        Epochs = o.Epochs,
        Metrics = [Metric.Accuracy],
        EarlyStoppingPatience = o.Patience,
        OnEpoch = e => Console.WriteLine(
            $"epoch {e.Epoch}/{e.Epochs}  loss {e.Loss:F4}  acc {e.Metrics["accuracy"]:P2}  " +
            $"val loss {e.ValidationLoss:F4}  val acc {e.ValidationMetrics!["accuracy"]:P2}  " +
            $"lr {e.LearningRate:G3}  ({e.Duration.TotalSeconds:F1}s){(e.IsBest ? "  *" : "")}"),
    }.Fit();
    Console.WriteLine(
        $"Trained in {stopwatch.Elapsed.TotalSeconds:F1}s; kept the weights of epoch {history.BestEpoch} " +
        $"(val loss {history.BestLoss:F4}){(history.StoppedEarly ? ", stopped early" : "")}");

    using var predictor = Predictor.For(model)
        .InputShape(imageShape)
        .Softmax()
        .Classes([.. classes.Select(c => c.Text)])
        .Build();

    var predicted = predictor.Predict(test).Select(p => p.Index).ToArray();
    ConsoleReport.PrintResults(testPart.Labels, predicted, [.. classes.Select(c => c.Text)]);
    Console.WriteLine();
    foreach (var group in classes.GroupBy(c => (c.Script, c.IsDigit)))
    {
        var members = group.Select(c => c.Index).ToHashSet();
        var rows = Enumerable.Range(0, testPart.Labels.Length).Where(i => members.Contains(testPart.Labels[i])).ToArray();
        int right = rows.Count(i => predicted[i] == testPart.Labels[i]);
        int sameScript = rows.Count(i => classes[predicted[i]].Script == group.Key.Script);
        Console.WriteLine($"{group.Key.Script} {(group.Key.IsDigit ? "digits" : "letters"),-8} {right / (double)rows.Length,7:P1} correct, " +
            $"{sameScript / (double)rows.Length:P1} in the right script ({rows.Length:N0} test images)");
    }

    // One test image per Arabic class, as the model sees it: the letters should stand upright.
    Console.WriteLine();
    Console.WriteLine("Arabic test characters as the model sees them (they should look upright, not turned or mirrored):");
    foreach (var c in classes.Where(c => c.Script == Script.Arabic).Where((_, i) => i % 10 == 0))
    {
        int i = Array.IndexOf(testPart.Labels, c.Index);
        if (i < 0)
            continue;
        ConsoleReport.PrintImage(testPart.Images[i], Datasets.Size, Datasets.Size);
        Console.WriteLine($"label {c.Text}  predicted {classes[predicted[i]].Text}");
    }

    predictor.Save(o.ModelPath);
    Console.WriteLine($"Saved the model to {Path.GetFullPath(o.ModelPath)}");
    Console.WriteLine("Try it: dotnet run -c Release -- demo");
}

// Writes a page of handwriting in both scripts for a known text, reads it back and compares.
void Demo(SampleOptions o)
{
    string text = (o.Text ?? DemoText).Replace("\\n", "\n");
    var (_, testPart, classes) = Datasets.Load(o.DataFolder, perSource: 20_000, training: false);
    var byText = classes.ToDictionary(c => c.Text);
    var pools = classes.Select(c => new List<int>()).ToArray();
    for (int i = 0; i < testPart.Labels.Length; i++)
        pools[testPart.Labels[i]].Add(i);

    (float[], int, int) Glyph(char ch, Random random)
    {
        // EMNIST Balanced has one class for both cases of most letters (c and C...): use the class that has them.
        string key = byText.ContainsKey(ch.ToString()) ? ch.ToString() : char.ToUpperInvariant(ch).ToString();
        if (!byText.TryGetValue(key, out var c) || pools[c.Index].Count == 0)
            throw new ArgumentException($"'{ch}' is not one of the model's characters: {string.Concat(classes.Select(k => k.Text))}");
        var pool = pools[c.Index];
        return PageWriter.Crop(testPart.Images[pool[random.Next(pool.Count)]], Datasets.Size, Datasets.Size);
    }

    var (pixels, width, height) = PageWriter.Write(text, Glyph, Characters.IsRightToLeft);
    string page = o.ExportFolder;
    ImageFiles.WritePgm(page, pixels, height, width);
    Console.WriteLine($"Wrote a {width} x {height} page of handwriting to {Path.GetFullPath(page)}");

    var lines = ReadPage(page, o.ModelPath, show: false);
    var expected = text.Replace("\r", "").Split('\n');
    Console.WriteLine();
    int errors = 0, total = 0;
    for (int i = 0; i < Math.Max(expected.Length, lines.Count); i++)
    {
        string want = i < expected.Length ? expected[i] : "", got = i < lines.Count ? lines[i].Text : "";
        int e = Levenshtein(Comparable(want), Comparable(got));
        (errors, total) = (errors + e, total + Comparable(want).Length);
        Console.WriteLine($"line {i + 1} ({(i < lines.Count ? lines[i].Script.ToString() : "-")})");
        Console.WriteLine($"  expected: {want}");
        Console.WriteLine($"  read:     {got}");
        Console.WriteLine($"  {e} edit{(e == 1 ? "" : "s")}");
    }
    Console.WriteLine();
    Console.WriteLine($"Character error rate: {errors / (double)Math.Max(total, 1):P1} ({errors} edit{(errors == 1 ? "" : "s")} over {total} characters)");
    Console.WriteLine($"The text read is also in {Path.GetFullPath(Path.ChangeExtension(page, ".txt"))} (an editor shows Arabic joined and right to left).");
}

void Ocr(SampleOptions o)
{
    if (o.ImagePath is null)
        throw new ArgumentException("ocr needs an image of a page: dotnet run -- ocr page.png (or run 'demo' to make one)");
    foreach (var line in ReadPage(o.ImagePath, o.ModelPath, show: true))
        Console.WriteLine($"[{(line.Script == Script.Arabic ? "ar" : "en")}] {line.Text}");
    Console.WriteLine($"Also written to {Path.GetFullPath(Path.ChangeExtension(o.ImagePath, ".txt"))}");
}

// The lines of a page image, each with its script and its text in reading order. Also writes them to a .txt file.
// Idrak's RegionClassifier separates the ink, frames each character the layout finds and classifies them in batches.
List<(Script Script, string Text)> ReadPage(string path, string modelPath, bool show)
{
    if (!File.Exists(modelPath))
        throw new FileNotFoundException($"No model at '{modelPath}'. Run 'dotnet run -- train' first.");
    using var classifier = RegionClassifier.Load(modelPath).Build();
    var byText = Characters.FromStored(classifier.Classes).ToDictionary(c => c.Text);

    var page = classifier.Foreground(ImageCodecs.Decode(path));
    var glyphs = new TextLineProposer().Find(page);
    var found = classifier.Classify(page, [.. glyphs.Select(g => g.Box)]);
    var answers = Enumerable.Range(0, found.Count).Select(i => found.Top(i, found.Classes.Count)).ToArray();
    Console.WriteLine($"Found {glyphs.Count} characters on {glyphs.Select(g => g.Line).Distinct().Count()} lines");

    var lines = new List<(Script, string)>();
    foreach (var line in Enumerable.Range(0, glyphs.Count).GroupBy(i => glyphs[i].Line))
    {
        var indices = line.ToList();
        // The line's script: the one most of its probability is on.
        var script = Enum.GetValues<Script>().MaxBy(s =>
            indices.Sum(i => answers[i].Where(c => byText[c.Class].Script == s).Sum(c => c.Score)));

        var words = new List<List<int>>();
        foreach (int i in indices)
        {
            if (words.Count == 0 || glyphs[i].SpaceBefore)
                words.Add([]);
            words[^1].Add(i);
        }

        var visual = words.Select(w => WordInContext([.. w.Select(i => answers[i])], script, byText)).ToList();
        lines.Add((script, TextOrder.Logical(visual, rightToLeft: script == Script.Arabic)));

        if (show)
            foreach (int i in indices.OrderBy(i => found.Confidence(i)).Take(1))
            {
                var image = new float[Datasets.Size * Datasets.Size];
                ContentFrame.Extract(page, glyphs[i].Box, image, Datasets.Size);   // the character as the model saw it
                ConsoleReport.PrintImage(image, Datasets.Size, Datasets.Size);
                Console.WriteLine($"least certain on line {glyphs[i].Line + 1}: {answers[i][0].Class} ({answers[i][0].Score:P0}), " +
                    $"else {string.Join(", ", answers[i].Skip(1).Take(2).Select(s => $"{s.Class} {s.Score:P0}"))}");
            }
    }

    File.WriteAllLines(Path.ChangeExtension(path, ".txt"), lines.Select(l => l.Item2), Encoding.UTF8);
    return lines;
}

// A word read within its line's script. Each character becomes its likeliest character of that script; then, in a
// word of mostly letters, a digit becomes the likeliest of the letters it looks like (1 -> I or L, 0 -> O or D,
// ١ -> ا), or else its likeliest letter, and in a word of mostly digits the other way round. The model's own
// probabilities choose among look-alikes: a 1 among letters is as often an L as an I. A Latin word then takes the case
// of most of its letters.
static string WordInContext(IReadOnlyList<IReadOnlyList<ClassScore>> word, Script script, Dictionary<string, CharacterClass> byText)
{
    IEnumerable<CharacterClass> Candidates(IReadOnlyList<ClassScore> a) => a.Select(s => byText[s.Class]).Where(c => c.Script == script);
    var chars = word.Select(a => Candidates(a).First()).ToArray();
    int digits = chars.Count(c => c.IsDigit);
    for (int i = 0; i < chars.Length; i++)
    {
        if (digits * 2 < chars.Length && chars[i].IsDigit)
            chars[i] = LikeliestOf(word[i], LooksLikeLetter.GetValueOrDefault(chars[i].Text), byText) ?? Candidates(word[i]).First(c => !c.IsDigit);
        else if (digits * 2 > chars.Length && !chars[i].IsDigit)
            chars[i] = LikeliestOf(word[i], LooksLikeDigit.GetValueOrDefault(chars[i].Text), byText) ?? Candidates(word[i]).First(c => c.IsDigit);
    }

    string text = string.Concat(chars.Select(c => c.Text));
    if (script != Script.Latin)
        return text;
    var letters = text.Where(char.IsLetter).ToArray();
    bool upper = letters.Count(char.IsUpper) * 2 >= letters.Length;
    return upper ? text.ToUpperInvariant() : text.ToLowerInvariant();
}

// The likeliest of some characters (look-alikes) by the model's probabilities, or null when none is a class.
static CharacterClass? LikeliestOf(IReadOnlyList<ClassScore> answer, string[]? options, Dictionary<string, CharacterClass> byText) =>
    options is null ? null
        : answer.Where(s => options.Contains(s.Class) && byText.ContainsKey(s.Class)).Select(s => byText[s.Class]).FirstOrDefault();

// Latin is compared without case (EMNIST Balanced shares one class between c and C...), with single spaces.
static string Comparable(string text) =>
    string.Join(' ', text.ToUpperInvariant().Split(' ', StringSplitOptions.RemoveEmptyEntries));

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
    private static readonly Dictionary<string, string[]> LooksLikeLetter = new()
    {
        ["0"] = ["O", "D"], ["1"] = ["I", "L"], ["2"] = ["Z"], ["5"] = ["S"], ["6"] = ["G", "b"], ["8"] = ["B"], ["9"] = ["g", "q"],
        ["١"] = ["ا"], ["٥"] = ["ه"],
    };

    private static readonly Dictionary<string, string[]> LooksLikeDigit = new()
    {
        ["O"] = ["0"], ["D"] = ["0"], ["I"] = ["1"], ["L"] = ["1"], ["Z"] = ["2"], ["S"] = ["5"], ["G"] = ["6"], ["b"] = ["6"],
        ["B"] = ["8"], ["g"] = ["9"], ["q"] = ["9"],
        ["ا"] = ["١"], ["ه"] = ["٥"],
    };
}
