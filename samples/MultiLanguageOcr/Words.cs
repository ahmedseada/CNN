using Idrak.Inference;

namespace MultiLanguageOcr;

/// <summary>
/// Reads a word within its line's script. Each character becomes its likeliest character of that script. Then the
/// word is letters or digits by the model's probabilities summed over the word: a word whose characters put more
/// probability on letters than on digits is a word of letters, even when look-alikes (L read as 1, O as 0) make most
/// of its likeliest characters digits. In a word of letters a digit becomes the likeliest of the letters it looks like
/// (1 -> I or L, 0 -> O or D, ١ -> ا), or else its likeliest letter, and in a word of digits the other way round. The
/// model's probabilities choose among look-alikes too: a 1 among letters is as often an L as an I. A Latin word then
/// takes the case of most of its letters.
/// </summary>
internal static class Words
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

    /// <summary>
    /// The word's text, its characters given as every class's probability (best first), in page order. The word is
    /// digits or letters as <paramref name="digits"/> says, or else by its own probabilities (<see cref="IsNumber"/>).
    /// </summary>
    public static string InContext(IReadOnlyList<IReadOnlyList<ClassScore>> word, Script script, Dictionary<string, CharacterClass> byText, bool? digits = null)
    {
        IEnumerable<CharacterClass> Candidates(IReadOnlyList<ClassScore> a) => a.Select(s => byText[s.Class]).Where(c => c.Script == script);
        var chars = word.Select(a => Candidates(a).First()).ToArray();
        bool number = digits ?? IsNumber(word, script, byText);

        for (int i = 0; i < chars.Length; i++)
        {
            if (!number && chars[i].IsDigit)
                chars[i] = LikeliestOf(word[i], LooksLikeLetter.GetValueOrDefault(chars[i].Text), byText) ?? Candidates(word[i]).First(c => !c.IsDigit);
            else if (number && !chars[i].IsDigit)
                chars[i] = LikeliestOf(word[i], LooksLikeDigit.GetValueOrDefault(chars[i].Text), byText) ?? Candidates(word[i]).First(c => c.IsDigit);
        }

        string text = string.Concat(chars.Select(c => c.Text));
        if (script != Script.Latin)
            return text;
        var letters = text.Where(char.IsLetter).ToArray();
        bool upper = letters.Count(char.IsUpper) * 2 >= letters.Length;
        return upper ? text.ToUpperInvariant() : text.ToLowerInvariant();
    }

    /// <summary>
    /// Whether characters (one word, or a run of one-character words such as 0 1 2 4 5) are digits: the probability
    /// they put on digits, within the script, beats the probability on letters.
    /// </summary>
    public static bool IsNumber(IEnumerable<IReadOnlyList<ClassScore>> characters, Script script, Dictionary<string, CharacterClass> byText)
    {
        double letterMass = 0, digitMass = 0;
        foreach (var answer in characters)
            foreach (var s in answer)
                if (byText[s.Class] is { } c && c.Script == script)
                    (letterMass, digitMass) = c.IsDigit ? (letterMass, digitMass + s.Score) : (letterMass + s.Score, digitMass);
        return digitMass > letterMass;
    }

    // The likeliest of some characters (look-alikes) by the model's probabilities, or null when none is a class.
    private static CharacterClass? LikeliestOf(IReadOnlyList<ClassScore> answer, string[]? options, Dictionary<string, CharacterClass> byText) =>
        options is null ? null
            : answer.Where(s => options.Contains(s.Class) && byText.ContainsKey(s.Class)).Select(s => byText[s.Class]).FirstOrDefault();
}
