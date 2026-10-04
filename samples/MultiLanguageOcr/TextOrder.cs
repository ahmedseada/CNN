namespace MultiLanguageOcr;

/// <summary>
/// The order characters take on a page (left to right) against the order they are read in. A right-to-left line
/// (Arabic) runs its words from right to left and each word's letters from right to left, but a number keeps its digits
/// left to right, as Arabic writes them. This is the core of Unicode's bidirectional rules for one script per line.
/// </summary>
internal static class TextOrder
{
    /// <summary>The words of a line as they appear on the page from left to right, each spelled left to right.</summary>
    public static List<string> Visual(string line, bool rightToLeft)
    {
        var words = line.Split(' ', StringSplitOptions.RemoveEmptyEntries).ToList();
        if (!rightToLeft)
            return words;
        words.Reverse();
        return [.. words.Select(w => IsNumber(w) ? w : new string([.. w.Reverse()]))];
    }

    /// <summary>A line read from the page (its words left to right, each spelled left to right) in reading order.</summary>
    public static string Logical(IReadOnlyList<string> visualWords, bool rightToLeft) =>
        // The same reversal as Visual: applied twice it gives back the original.
        string.Join(' ', rightToLeft ? Visual(string.Join(' ', visualWords), rightToLeft: true) : visualWords);

    /// <summary>Whether a word is a number (digits only, Latin or Arabic-Indic).</summary>
    public static bool IsNumber(string word) => word.Length > 0 && word.All(char.IsDigit);
}
