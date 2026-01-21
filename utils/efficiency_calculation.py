import torch.nn as nn
import torch
import time
import numpy as np
def count_trainable_params(model: nn.Module):
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    a = sum(p.numel() for p in model.parameters())
    return t, a

@torch.no_grad()
def measure_inference_latency(
    model: nn.Module,
    example_batch,
    device: torch.device = None,
    warmup: int = 10,
    runs: int = 100,
):
  
    if device is None:
        device = _device_of(model)
    model.eval()

    x = example_batch[0] if isinstance(example_batch, (list, tuple)) else example_batch
    x = x.to(device, non_blocking=True)

    # warmup
    for _ in range(max(0, warmup)):
        _ = model(x)

    times = []
    if device.type == "cuda":
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(runs):
            torch.cuda.synchronize(device)
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize(device)
            times.append(starter.elapsed_time(ender) / 1000.0)  # s
    else:
        for _ in range(runs):
            t0 = time.perf_counter()
            _  = model(x)
            times.append(time.perf_counter() - t0)

    times = np.array(times)
    batch_size = x.shape[0] if hasattr(x, "shape") else len(x)
    mean_s = float(times.mean())

    return {
        "latency_ms_mean": mean_s * 1000.0,
        "latency_ms_p50": float(np.percentile(times, 50)) * 1000.0,
        "latency_ms_p95": float(np.percentile(times, 95)) * 1000.0,
        "throughput_samples_per_s": batch_size / (mean_s + 1e-12),
        "batch_size": batch_size,
    }
