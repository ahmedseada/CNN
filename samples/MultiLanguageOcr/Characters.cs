namespace MultiLanguageOcr;

/// <summary>A writing system the model reads.</summary>
internal enum Script { Latin, Arabic }

/// <summary>One class of the model: its character, script and kind.</summary>
internal sealed record CharacterClass(int Index, string Text, Script Script, bool IsDigit);

/// <summary>
/// The model's 85 classes, in label order: EMNIST Balanced's 47 (digits, capitals and 11 lower-case letters), AHCD's 28
/// Arabic letters (isolated forms) and MADBase's 10 Arabic-Indic digits.
/// </summary>
internal static class Characters
{
    /// <summary>AHCD's letters in its label order (1 = alef ... 28 = yeh).</summary>
    public const string ArabicLetters = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي";

    /// <summary>MADBase's digits in its label order (0-9).</summary>
    public const string ArabicDigits = "٠١٢٣٤٥٦٧٨٩";

    /// <summary>The classes, given EMNIST Balanced's characters in its label order.</summary>
    public static CharacterClass[] Classes(IReadOnlyList<string> emnist) =>
    [
        .. emnist.Select((c, i) => new CharacterClass(i, c, Script.Latin, char.IsDigit(c[0]))),
        .. ArabicLetters.Select((c, i) => new CharacterClass(emnist.Count + i, c.ToString(), Script.Arabic, IsDigit: false)),
        .. ArabicDigits.Select((c, i) => new CharacterClass(emnist.Count + ArabicLetters.Length + i, c.ToString(), Script.Arabic, IsDigit: true)),
    ];

    /// <summary>The classes back from a saved model's class names (EMNIST's come first, the 38 Arabic ones last).</summary>
    public static CharacterClass[] FromStored(IReadOnlyList<string> names) =>
        Classes([.. names.Take(names.Count - ArabicLetters.Length - ArabicDigits.Length)]);

    /// <summary>Whether a line of text is written right to left (it has Arabic characters).</summary>
    public static bool IsRightToLeft(string text) => text.Any(c => c is >= '؀' and <= 'ۿ');
}
