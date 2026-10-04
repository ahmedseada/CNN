using Idrak.Diagnostics;
using Idrak.Layers;

namespace CnnSamples.Shared;

/// <summary>
/// Keeps a copy of the weights of the best epoch so far and puts them back after training.
/// </summary>
/// <remarks>
/// Idrak 0.2.0's RestoreBestWeights restores them only when early stopping ends the run, not when the run reaches its
/// last epoch (fixed for 0.2.1 in https://github.com/ahmedseada/Idrak/pull/1). Use it as
/// <c>OnEpoch = e => best.Track(e)</c>, then call <see cref="Restore"/> after Fit.
/// </remarks>
public sealed class BestWeights(Module model) : IDisposable
{
    private readonly MemoryStream _weights = new();

    /// <summary>The epoch whose weights are kept (0 before the first).</summary>
    public int Epoch { get; private set; }

    /// <summary>Copies the weights when <paramref name="epoch"/> is the best so far.</summary>
    public void Track(EpochCompleted epoch)
    {
        if (!epoch.IsBest)
            return;
        _weights.SetLength(0);
        model.Save(_weights);
        Epoch = epoch.Epoch;
    }

    /// <summary>Loads the best epoch's weights back into the model.</summary>
    public void Restore()
    {
        if (Epoch == 0)
            return;
        _weights.Position = 0;
        model.Load(_weights);
    }

    public void Dispose() => _weights.Dispose();
}
