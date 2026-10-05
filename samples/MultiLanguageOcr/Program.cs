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
//   dotnet run -- demo    --count 50 [--seed 1]                       50 pages of random words, scored per script
//   dotnet run -- ocr     <page.png|bmp|pgm> [--model multilang.ikm] [--crop x,y,w,h] [--photo | --no-photo]

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

// Writes a page of handwriting in both scripts for a known text, reads it back and compares. With --count N (or
// --seed), it writes N pages of random words instead, each from other test characters, and scores them together.
void Demo(SampleOptions o)
{
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

    int pages = Math.Max(o.ExportCount, 1);
    bool random = o.ExportCount > 0 || o.Seed is not null;
    int seed = o.Seed ?? 1;
    string page = o.ExportFolder;
    var errorsByScript = new Dictionary<Script, (int Errors, int Total)>();
    for (int p = 0; p < pages; p++)
    {
        string text = random ? RandomText(classes, new Random(seed + p)) : (o.Text ?? DemoText).Replace("\\n", "\n");
        var (pixels, width, height) = PageWriter.Write(text, Glyph, Characters.IsRightToLeft, seed: seed + p);
        ImageFiles.WritePgm(page, pixels, height, width);
        if (pages == 1)
            Console.WriteLine($"Wrote a {width} x {height} page of handwriting to {Path.GetFullPath(page)}");

        var lines = ReadPage(page, o.ModelPath, show: false, report: pages == 1);
        var expected = text.Replace("\r", "").Split('\n');
        if (pages == 1)
            Console.WriteLine();
        int pageErrors = 0, pageTotal = 0;
        for (int i = 0; i < Math.Max(expected.Length, lines.Count); i++)
        {
            string want = i < expected.Length ? expected[i] : "", got = i < lines.Count ? lines[i].Text : "";
            int e = Levenshtein(Comparable(want), Comparable(got));
            var script = Characters.IsRightToLeft(want) ? Script.Arabic : Script.Latin;
            var (se, st) = errorsByScript.GetValueOrDefault(script);
            errorsByScript[script] = (se + e, st + Comparable(want).Length);
            (pageErrors, pageTotal) = (pageErrors + e, pageTotal + Comparable(want).Length);
            if (pages == 1 || e > 0)
            {
                Console.WriteLine($"{(pages == 1 ? "" : $"page {p + 1} ")}line {i + 1} ({(i < lines.Count ? lines[i].Script.ToString() : "-")})");
                Console.WriteLine($"  expected: {want}");
                Console.WriteLine($"  read:     {got}");
                Console.WriteLine($"  {e} edit{(e == 1 ? "" : "s")}");
            }
        }
    }

    Console.WriteLine();
    int errors = errorsByScript.Values.Sum(v => v.Errors), total = errorsByScript.Values.Sum(v => v.Total);
    if (pages > 1)
    {
        Console.WriteLine($"{pages} pages of random words (seeds {seed}-{seed + pages - 1}), each from other test characters:");
        foreach (var (script, (e, t)) in errorsByScript.OrderBy(kv => kv.Key))
            Console.WriteLine($"  {script,-6} {e / (double)Math.Max(t, 1):P1} character error rate ({e} edits over {t:N0} characters)");
    }
    Console.WriteLine($"Character error rate: {errors / (double)Math.Max(total, 1):P1} ({errors} edit{(errors == 1 ? "" : "s")} over {total:N0} characters)");
    Console.WriteLine($"The text read{(pages > 1 ? " from the last page" : "")} is also in {Path.GetFullPath(Path.ChangeExtension(page, ".txt"))} (an editor shows Arabic joined and right to left).");
}

// A page of random words: a Latin line and an Arabic line, each with two words of letters and one number, and the
// same again. Random words have no spelling to lean on, so every character must be read on its own.
static string RandomText(CharacterClass[] classes, Random random)
{
    string Word(Script script, bool digits, int min, int max)
    {
        var pick = classes.Where(c => c.Script == script && c.IsDigit == digits).ToArray();
        return string.Concat(Enumerable.Range(0, random.Next(min, max + 1)).Select(_ => pick[random.Next(pick.Length)].Text));
    }

    string Line(Script script) => string.Join(' ', Word(script, false, 2, 6), Word(script, true, 2, 4), Word(script, false, 2, 6));
    return string.Join('\n', Line(Script.Latin), Line(Script.Arabic), Line(Script.Latin), Line(Script.Arabic));
}

void Ocr(SampleOptions o)
{
    if (o.ImagePath is null)
        throw new ArgumentException("ocr needs an image of a page: dotnet run -- ocr page.png (or run 'demo' to make one)");
    var crop = o.Crop is { } c ? PhotoPage.ParseCrop(c) : (PixelBox?)null;
    foreach (var line in ReadPage(o.ImagePath, o.ModelPath, show: true, crop: crop, photo: o.Photo))
        Console.WriteLine($"[{(line.Script == Script.Arabic ? "ar" : "en")}] {line.Text}");
    Console.WriteLine($"Also written to {Path.GetFullPath(Path.ChangeExtension(o.ImagePath, ".txt"))}");
}

// The lines of a page image, each with its script and its text in reading order. Also writes them to a .txt file.
// Idrak's RegionClassifier separates the ink, frames each character the layout finds and classifies them in batches.
List<(Script Script, string Text)> ReadPage(string path, string modelPath, bool show, bool report = true, PixelBox? crop = null, bool? photo = null)
{
    if (!File.Exists(modelPath))
        throw new FileNotFoundException($"No model at '{modelPath}'. Run 'dotnet run -- train' first.");
    var clock = Stopwatch.StartNew();
    var times = new List<(string Stage, double Ms)>();
    void Lap(string stage) { times.Add((stage, clock.Elapsed.TotalMilliseconds)); clock.Restart(); }

    using var classifier = RegionClassifier.Load(modelPath).Build();
    var byText = Characters.FromStored(classifier.Classes).ToDictionary(c => c.Text);
    Lap("load the model");
    var image = ImageCodecs.Decode(ImageConversion.Readable(path));
    Lap("open the image");
    var (page, prepared) = PhotoPage.Foreground(image, classifier, crop, photo);
    Lap(prepared.Photo ? "ink (photo)" : "ink");
    if (report && prepared.Photo)
        Console.WriteLine($"Read as a photo: one threshold took {prepared.GlobalInk:P0} of the pixels as ink; a local threshold " +
            $"took {prepared.Ink:P1}, after removing {prepared.RuledPixels:N0} ruled-line pixels and {prepared.SurroundingRegions} regions around the page");
    var glyphs = new TextLineProposer().Find(page);
    Lap("lines and characters");
    var found = classifier.Classify(page, [.. glyphs.Select(g => g.Box)]);
    Lap("classify (first run)");
    if (report && glyphs.Count > 0)
    {
        classifier.Classify(page, [.. glyphs.Select(g => g.Box)]);           // again, warm: the speed of every later page
        Lap("classify (warm)");
    }
    var answers = Enumerable.Range(0, found.Count).Select(i => found.Top(i, found.Classes.Count)).ToArray();
    if (report)
        Console.WriteLine($"Found {glyphs.Count} characters on {glyphs.Select(g => g.Line).Distinct().Count()} lines");

    var lines = new List<(Script, string)>();
    foreach (var line in Enumerable.Range(0, glyphs.Count).GroupBy(i => glyphs[i].Line))
    {
        var indices = line.ToList();
        // A dash (a short flat mark standing between spaces, no character of the model) is its own word, read without
        // the model. The flat strokes joining Arabic letters (هـنـا) touch their letters, so they are no dashes.
        var heights = indices.Select(i => glyphs[i].Box.Height).Order().ToArray();
        int typical = heights[heights.Length / 2];
        bool Dash(int i)
        {
            int at = indices.IndexOf(i);
            bool spaced = glyphs[i].SpaceBefore && (at == indices.Count - 1 || glyphs[indices[at + 1]].SpaceBefore);
            return spaced && glyphs[i].Box.Height * 4 <= typical && glyphs[i].Box.Width >= 2 * glyphs[i].Box.Height;
        }

        // The line's script: the one most of its probability is on.
        var script = Enum.GetValues<Script>().MaxBy(s =>
            indices.Where(i => !Dash(i)).Sum(i => answers[i].Where(c => byText[c.Class].Script == s).Sum(c => c.Score)));

        var words = new List<List<int>>();
        foreach (int i in indices)
        {
            if (words.Count == 0 || glyphs[i].SpaceBefore || Dash(i) || Dash(words[^1][^1]))
                words.Add([]);
            words[^1].Add(i);
        }

        // A run of one-character words (0 1 2 4 5) is digits or letters as a whole: the clear ones decide for the
        // look-alikes (a lone 1 is as likely an L).
        var kind = new bool?[words.Count];
        for (int start = 0; start < words.Count;)
        {
            int end = start;
            while (end < words.Count && words[end].Count == 1 && !Dash(words[end][0]))
                end++;
            if (end - start >= 2)
            {
                bool number = Words.IsNumber(words[start..end].Select(w => answers[w[0]]), script, byText);
                for (int k = start; k < end; k++)
                    kind[k] = number;
            }
            start = Math.Max(end, start + 1);
        }

        var visual = words.Select((w, k) => Dash(w[0]) ? "-" : Words.InContext([.. w.Select(i => answers[i])], script, byText, kind[k])).ToList();
        lines.Add((script, TextOrder.Logical(visual, rightToLeft: script == Script.Arabic)));

        if (show)
            foreach (int i in indices.OrderBy(i => found.Confidence(i)).Take(1))
            {
                var frame = new float[Datasets.Size * Datasets.Size];
                ContentFrame.Extract(page, glyphs[i].Box, frame, Datasets.Size);   // the character as the model saw it
                ConsoleReport.PrintImage(frame, Datasets.Size, Datasets.Size);
                Console.WriteLine($"least certain on line {glyphs[i].Line + 1}: {answers[i][0].Class} ({answers[i][0].Score:P0}), " +
                    $"else {string.Join(", ", answers[i].Skip(1).Take(2).Select(s => $"{s.Class} {s.Score:P0}"))}");
            }
    }

    File.WriteAllLines(Path.ChangeExtension(path, ".txt"), lines.Select(l => l.Item2), Encoding.UTF8);
    if (report)
    {
        Lap(show ? "words, order (and printing)" : "words and reading order");
        PrintTimes(times, glyphs.Count, image.Width, image.Height, classifier);
    }
    return lines;
}

// Where the time went, the characters per second of the model (warm), and the process's peak memory.
static void PrintTimes(List<(string Stage, double Ms)> times, int characters, int width, int height, RegionClassifier classifier)
{
    Console.WriteLine();
    Console.WriteLine($"Time ({width} x {height} image, {characters} characters, on {Device.Default}):");
    foreach (var (stage, ms) in times)
        Console.WriteLine($"  {stage,-26} {ms,9:F1} ms");
    double page = times.Where(t => t.Stage != "load the model" && t.Stage != "classify (warm)").Sum(t => t.Ms);
    Console.WriteLine($"  {"page, model loaded",-26} {page,9:F1} ms (all but loading the model and the warm repeat)");
    if (times.FirstOrDefault(t => t.Stage == "classify (warm)") is { Ms: > 0 } warm && characters > 0)
        Console.WriteLine($"  model, warm: {characters / (warm.Ms / 1000):N0} characters/s ({warm.Ms * 1000 / characters:F0} µs each)");
    using var process = Process.GetCurrentProcess();
    Console.WriteLine($"  memory: {process.PeakWorkingSet64 / (1024.0 * 1024):N0} MB peak working set, {GC.GetTotalMemory(false) / (1024.0 * 1024):N0} MB managed now");
}

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
