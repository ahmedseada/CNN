using Idrak;
using Idrak.Diagnostics;

namespace MultiLanguageOcr;

/// <summary>
/// Collects Idrak's inference telemetry (<see cref="TelemetryLevel.Inference"/>: one <see cref="InferenceCompleted"/>
/// per batch the model runs, timed with the device synchronized) so a page's model time and throughput come from the
/// library's own measurement. With <c>--telemetry</c> Idrak's console logger prints every event as well.
/// </summary>
internal sealed class InferenceLog : ITelemetryHook
{
    private readonly List<InferenceCompleted> _events = [];

    public TelemetryLevel Levels => TelemetryLevel.Inference;

    public void OnInference(in InferenceCompleted e)
    {
        lock (_events)
            _events.Add(e);
    }

    /// <summary>Subscribes a log (and, when asked, Idrak's console logger); dispose the session to stop.</summary>
    public static (InferenceLog Log, TelemetrySession Session) Start(bool console)
    {
        var log = new InferenceLog();
        var builder = Telemetry.Configure().Hook(log);
        if (console)
            builder = builder.Console(TelemetryLevel.Inference);
        return (log, builder.Start());
    }

    /// <summary>The events logged since <paramref name="from"/> (an earlier <see cref="Count"/>): batches, samples and model time.</summary>
    public (int Batches, int Samples, TimeSpan Latency) Since(int from)
    {
        lock (_events)
        {
            var part = _events.Skip(from).ToArray();
            return (part.Length, part.Sum(e => e.Samples), TimeSpan.FromTicks(part.Sum(e => e.Latency.Ticks)));
        }
    }

    /// <summary>The number of events so far.</summary>
    public int Count
    {
        get
        {
            lock (_events)
                return _events.Count;
        }
    }

    /// <summary>The device's memory as Idrak counts it (tensors in use, cached for reuse, the limit).</summary>
    public static string Memory(Device device)
    {
        var m = ComputeResources.GetMemoryUsage(device);
        static string Mb(long bytes) => $"{bytes / (1024.0 * 1024):N0} MB";
        return $"{Mb(m.InUse)} in use, {Mb(m.Cached)} cached{(m.Limit is { } limit ? $", limit {Mb(limit)}" : "")}";
    }
}
