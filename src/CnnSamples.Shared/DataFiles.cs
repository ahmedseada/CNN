namespace CnnSamples.Shared;

/// <summary>Finds files that <c>dotnet idrak data download</c> fetched, by name.</summary>
public static class DataFiles
{
    /// <summary>
    /// The sample's data folder first, then Idrak's own download cache (what <c>idrak data download</c> uses without
    /// --cache): IDRAK_CACHE, otherwise ~/.cache/idrak.
    /// </summary>
    public static string[] SearchFolders(string dataFolder)
    {
        var cache = Environment.GetEnvironmentVariable("IDRAK_CACHE");
        return
        [
            dataFolder,
            string.IsNullOrEmpty(cache)
                ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "idrak")
                : cache,
        ];
    }

    /// <summary>The first file named <paramref name="file"/> under any of the folders, or null.</summary>
    public static string? TryFind(string file, IEnumerable<string> folders) =>
        folders.Where(Directory.Exists)
            .Select(folder => Directory.EnumerateFiles(folder, file, SearchOption.AllDirectories).FirstOrDefault())
            .FirstOrDefault(path => path is not null);

    /// <summary>The first file named <paramref name="file"/> under any of the folders.</summary>
    /// <exception cref="FileNotFoundException">No folder has it; the message says how to download it.</exception>
    public static string Find(string file, IReadOnlyList<string> folders, string downloadHint) =>
        TryFind(file, folders) ?? throw new FileNotFoundException(
            $"'{file}' not found under {string.Join(" or ", folders.Select(f => $"'{f}'"))}. {downloadHint}");
}
