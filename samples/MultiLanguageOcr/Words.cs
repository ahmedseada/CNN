namespace MultiLanguageOcr;

/// <summary>A character's probability for every class, in the model's class order.</summary>
internal delegate ReadOnlySpan<float> Scores(int character);

/// <summary>
/// Reads a word within its line's script. Each character becomes its likeliest character of that script. Then the
/// word is letters or digits by the model's probabilities summed over the word: a word whose characters put more
/// probability on letters than on digits is a word of letters, even when look-alikes (L read as 1, O as 0) make most
/// of its likeliest characters digits. In a word of letters a digit becomes the likeliest of the letters it looks like
/// (1 -> I or L, 0 -> O or D, ١ -> ا), or else its likeliest letter, and in a word of digits the other way round. The
/// model's probabilities choose among look-alikes too: a 1 among letters is as often an L as an I. A Latin word then
/// takes the case of most of its letters.
/// </summary>
/// <remarks>
/// Everything is read straight from each character's probabilities (one pass over the classes per question, the
/// lowest class index winning a tie as a stable sort by probability would): no list of classes is sorted or built.
/// </remarks>
internal sealed class Words
{
    // Characters that handwriting makes hard to tell apart, for words whose kind (letters or digits) is known.
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

    private readonly CharacterClass[] _classes;
    private readonly int[]?[] _letterOptions, _digitOptions;     // by class index: the look-alikes' class indices

    /// <summary>The word reader for a model's classes (in its output order).</summary>
    public Words(CharacterClass[] classes)
    {
        _classes = classes;
        var byText = classes.ToDictionary(c => c.Text);
        int[]? Options(Dictionary<string, string[]> table, CharacterClass c) =>
            table.TryGetValue(c.Text, out var texts) ? [.. texts.Where(byText.ContainsKey).Select(t => byText[t].Index)] : null;
        _letterOptions = [.. classes.Select(c => Options(LooksLikeLetter, c))];
        _digitOptions = [.. classes.Select(c => Options(LooksLikeDigit, c))];
    }

    /// <summary>
    /// The word's text, its characters given by index into <paramref name="scores"/>, in page order. The word is
    /// digits or letters as <paramref name="digits"/> says, or else by its own probabilities (<see cref="IsNumber"/>).
    /// </summary>
    public string InContext(ReadOnlySpan<int> word, Scores scores, Script script, bool? digits = null)
    {
        Span<int> chars = word.Length <= 64 ? stackalloc int[64] : new int[word.Length];   // small words stay on the stack
        chars = chars[..word.Length];
        for (int i = 0; i < word.Length; i++)
            chars[i] = Best(scores(word[i]), script, kind: null);
        bool number = digits ?? IsNumber(word, scores, script);

        for (int i = 0; i < chars.Length; i++)
        {
            var c = _classes[chars[i]];
            if (!number && c.IsDigit)
                chars[i] = LikeliestOf(scores(word[i]), _letterOptions[c.Index]) ?? Best(scores(word[i]), script, kind: false);
            else if (number && !c.IsDigit)
                chars[i] = LikeliestOf(scores(word[i]), _digitOptions[c.Index]) ?? Best(scores(word[i]), script, kind: true);
        }

        int length = 0, letters = 0, upper = 0;
        foreach (int k in chars)
            length += _classes[k].Text.Length;
        Span<char> text = length <= 256 ? stackalloc char[256] : new char[length];
        text = text[..length];
        int at = 0;
        foreach (int k in chars)
        {
            _classes[k].Text.CopyTo(text[at..]);
            at += _classes[k].Text.Length;
        }
        if (script != Script.Latin)
            return new string(text);
        foreach (char ch in text)
            if (char.IsLetter(ch))
                (letters, upper) = (letters + 1, upper + (char.IsUpper(ch) ? 1 : 0));
        bool toUpper = upper * 2 >= letters;
        for (int i = 0; i < text.Length; i++)
            text[i] = toUpper ? char.ToUpperInvariant(text[i]) : char.ToLowerInvariant(text[i]);
        return new string(text);
    }

    /// <summary>
    /// Whether characters (one word, or a run of one-character words such as 0 1 2 4 5) are digits: the probability
    /// they put on digits, within the script, beats the probability on letters.
    /// </summary>
    public bool IsNumber(ReadOnlySpan<int> characters, Scores scores, Script script)
    {
        double letterMass = 0, digitMass = 0;
        foreach (int character in characters)
        {
            var p = scores(character);
            for (int k = 0; k < p.Length; k++)
                if (_classes[k].Script == script)
                {
                    if (_classes[k].IsDigit)
                        digitMass += p[k];
                    else
                        letterMass += p[k];
                }
        }
        return digitMass > letterMass;
    }

    /// <summary>The probability a character puts on a script's classes.</summary>
    public double Mass(ReadOnlySpan<float> p, Script script)
    {
        double mass = 0;
        for (int k = 0; k < p.Length; k++)
            if (_classes[k].Script == script)
                mass += p[k];
        return mass;
    }

    // The likeliest class of a script (and, when `kind` is given, of digits or of letters); the lowest index wins a tie.
    private int Best(ReadOnlySpan<float> p, Script script, bool? kind)
    {
        int best = -1;
        for (int k = 0; k < p.Length; k++)
            if (_classes[k].Script == script && (kind is null || _classes[k].IsDigit == kind) && (best < 0 || p[k] > p[best]))
                best = k;
        return best;
    }

    // The likeliest of some classes (look-alikes), or null when there are none; the lowest index wins a tie.
    private static int? LikeliestOf(ReadOnlySpan<float> p, int[]? options)
    {
        if (options is null || options.Length == 0)
            return null;
        int best = -1;
        foreach (int k in options)
            if (best < 0 || p[k] > p[best] || (p[k] == p[best] && k < best))
                best = k;
        return best;
    }
}
