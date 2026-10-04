# -*- coding: utf-8 -*-
"""
PASAN_DG_flops.py
=================
Computes FLOPs for every model in the ablation (imported from PASAN_DG_main.py).

  Model | FLOPs (G) | MACs (G)

Method
------
  torch.utils.flop_counter.FlopCounterMode (no extra dependency), one forward
  pass of a single 224x224 RGB image, eval mode, no_grad, on CPU.
    FLOPs = 2 x MACs  (convolution / matmul / linear / bmm operators)
  Notes:
    * PASAN_DG: the frozen ConvNeXt-Small teacher is NOT part of the deployed
      forward pass, so it is excluded (it is only used during training).
    * FFT (rfft2/irfft2), normalisation and element-wise ops are not counted
      by FlopCounterMode; their contribution is negligible next to the
      convolutions.

Usage:  python PASAN_DG_flops.py
Output: PASAN_DG_COMPLEXITY_RESULTS/flops_table.csv
"""

import gc, warnings
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from pathlib import Path
from VaitheeANUFA import create_model, MODELS, IMG_SIZE

# Separate output folder -> never touches the training results folder
OUT_DIR = Path("PASAN_DG_COMPLEXITY_RESULTS")

warnings.filterwarnings("ignore")

NUM_CLASSES = 4


@torch.no_grad()
def count_flops(model, batch_size=1):
    model = model.to("cpu").eval()
    x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)
    with FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops()


if __name__ == "__main__":
    rows = []
    for name in MODELS:
        model = create_model(name, NUM_CLASSES)
        if hasattr(model, "teacher"):
            del model.teacher                       # training-only
        flops = count_flops(model, batch_size=1)
        rows.append({
            "Model":      name,
            "FLOPs (G)":  round(flops / 1e9, 3),
            "MACs (G)":   round(flops / 2e9, 3),
        })
        print(rows[-1])
        del model
        gc.collect()

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "flops_table.csv", index=False)

    print("\n" + "=" * 50)
    print(out.to_string(index=False))
    print("=" * 50)
    print(f"Saved: {(OUT_DIR / 'flops_table.csv').resolve()}")
