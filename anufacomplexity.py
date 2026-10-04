# -*- coding: utf-8 -*-
"""
PASAN_DG_complexity.py
======================
Computes, for every model in the ablation (imported from PASAN_DG_main.py):

  Model | Params (M) | bb (M) | Head (M) | CPU latency (ms) | Mem b1 (MB) | Mem b16 (MB)

Definitions
-----------
  Params (M)  : ALL parameters (trainable + frozen) of the deployed network.
                For PASAN_DG the frozen ConvNeXt-Small teacher is NOT counted
                (it is not used at inference); it is reported separately in
                the 'Teacher (M)' column.
  bb (M)      : backbone parameters
                  Baseline : ConvNeXt-Tiny without its classifier head
                  Others   : the FeatureListNet backbone (model.bb)
  Head (M)    : Params - bb   (spectral / mixstyle / attention / bottleneck /
                classifier — everything that is not the backbone)
  CPU latency : mean +/- std of single-image (batch 1, 224x224) forward pass,
                eval mode, no_grad, after warm-up.
  Mem b1/b16  : peak inference memory in MB (weights + activations) for
                batch 1 / batch 16.  Measured on CUDA if available
                (torch.cuda.max_memory_allocated), otherwise peak process RSS
                on CPU.

Usage:  python PASAN_DG_complexity.py
Output: PASAN_DG_COMPLEXITY_RESULTS/complexity_table.csv
"""

import gc, time, threading, warnings
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from VaitheeANUFA import create_model, MODELS, IMG_SIZE

# Separate output folder -> never touches the training results folder
OUT_DIR = Path("PASAN_DG_COMPLEXITY_RESULTS")

warnings.filterwarnings("ignore")

NUM_CLASSES   = 4          # glioma / meningioma / notumor / pituitary
CPU_THREADS   = None       # None -> torch default; set an int to pin threads
WARMUP_RUNS   = 10
TIMED_RUNS    = 50
MEM_DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
MEM_CUDA_IDX  = 0          # which GPU to use for memory measurement

if CPU_THREADS is not None:
    torch.set_num_threads(CPU_THREADS)


# -----------------------------------------------------------------------------
# Parameter counting
# -----------------------------------------------------------------------------
def n_params(module):
    return sum(p.numel() for p in module.parameters())


def count_parameters(model_name, model):
    """Returns (total_M, backbone_M, head_M, teacher_M)."""
    teacher_p = 0
    if hasattr(model, "teacher"):
        teacher_p = n_params(model.teacher)

    total_p = n_params(model) - teacher_p          # deployed network only

    if model_name == "Baseline":
        base   = model.model                       # full timm ConvNeXt-Tiny
        head_p = n_params(base.head)               # norm + fc classifier head
        bb_p   = total_p - head_p
    else:
        bb_p   = n_params(model.bb)
        head_p = total_p - bb_p

    return total_p / 1e6, bb_p / 1e6, head_p / 1e6, teacher_p / 1e6


# -----------------------------------------------------------------------------
# CPU latency
# -----------------------------------------------------------------------------
@torch.no_grad()
def cpu_latency_ms(model):
    model = model.to("cpu").eval()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    for _ in range(WARMUP_RUNS):
        model(x)
    times = []
    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


# -----------------------------------------------------------------------------
# Peak memory
# -----------------------------------------------------------------------------
def _rss_mb():
    import psutil
    return psutil.Process().memory_info().rss / (1024 ** 2)


class _RSSPeak:
    """Background sampler of process RSS (CPU memory fallback)."""
    def __init__(self, interval=0.005):
        self.interval = interval
        self.peak     = 0.0
        self._stop    = threading.Event()
        self._t       = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, _rss_mb())
            time.sleep(self.interval)

    def __enter__(self):
        self.peak = _rss_mb()
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, _rss_mb())


@torch.no_grad()
def peak_memory_mb(model_name, batch_size):
    """Peak (weights + activations) inference memory in MB for one batch size."""
    gc.collect()
    if MEM_DEVICE == "cuda":
        dev = torch.device(f"cuda:{MEM_CUDA_IDX}")
        torch.cuda.empty_cache()
        torch.cuda.synchronize(dev)
        base = torch.cuda.memory_allocated(dev)
        torch.cuda.reset_peak_memory_stats(dev)

        model = create_model(model_name, NUM_CLASSES)
        if hasattr(model, "teacher"):
            del model.teacher                       # not used at inference
        model = model.to(dev).eval()
        x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE, device=dev)
        model(x)
        torch.cuda.synchronize(dev)
        peak = (torch.cuda.max_memory_allocated(dev) - base) / (1024 ** 2)
        del model, x
        torch.cuda.empty_cache()
        return peak

    # CPU fallback: peak RSS delta (weights + activations)
    rss0  = _rss_mb()
    model = create_model(model_name, NUM_CLASSES)
    if hasattr(model, "teacher"):
        del model.teacher
    gc.collect()
    model = model.eval()
    x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)
    with _RSSPeak() as pk:
        model(x)
    peak = pk.peak - rss0
    del model, x
    gc.collect()
    return peak


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\nMemory measured on: {MEM_DEVICE.upper()}  |  "
          f"CPU threads: {torch.get_num_threads()}")
    rows = []

    for name in MODELS:
        print(f"\n--- {name} ---")
        model = create_model(name, NUM_CLASSES)

        tot, bb, head, teacher = count_parameters(name, model)
        lat_mean, lat_std      = cpu_latency_ms(model)
        del model
        gc.collect()

        mem1  = peak_memory_mb(name, 1)
        mem16 = peak_memory_mb(name, 16)

        rows.append({
            "Model":                name,
            "Params (M)":           round(tot,  3),
            "bb (M)":               round(bb,   3),
            "Head (M)":             round(head, 3),
            "Teacher (M, train-only)": round(teacher, 3),
            "CPU latency (ms)":     f"{lat_mean:.2f} +/- {lat_std:.2f}",
            "Mem b1 (MB)":          round(mem1,  1),
            "Mem b16 (MB)":         round(mem16, 1),
        })
        print(rows[-1])

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "complexity_table.csv", index=False)

    print("\n" + "=" * 110)
    print(out.to_string(index=False))
    print("=" * 110)
    print(f"Saved: {(OUT_DIR / 'complexity_table.csv').resolve()}")
