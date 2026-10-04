# ANUFA framework: PASAN-DG: Probabilistic Augmented Spectral Attention Network for Domain Generalization in Medical Imaging

Code for training, ablating, and profiling **PASAN-DG** on a 4-class brain-tumor MRI task
(glioma / meningioma / notumor / pituitary). The repository contains three scripts:

| ID | File | Purpose |
|----|------|---------|
| **A** | `PASAN_DG_main.py` | Train and evaluate all ablation models over 6 seeds, produce metrics, Grad-CAM, statistical tests, and plots |
| **B** | `PASAN_DG_complexity.py` | Parameter counts, CPU latency, and peak inference memory per model |
| **C** | `PASAN_DG_flops.py` | FLOPs / MACs per model |

Scripts **B** and **C** import `create_model`, `MODELS`, and `IMG_SIZE` from script **A**, so all three
files must sit in the same directory.

---

## 1. Ablation Design

The ablation is **cumulative**: each row adds one component on top of the previous row.

| # | Model | Spectral | KL | MixStyle | Attention | Distill | Description |
|---|-------|:--:|:--:|:--:|:--:|:--:|-------------|
| 1 | `Baseline` | – | – | – | – | – | ConvNeXt-Tiny (`convnext_tiny.in12k_ft_in1k`) trained with ERM |
| 2 | `Plus_Spectral` | ✔ | – | – | – | – | + `SpectralMessageBlock` (FFT channel reweighting, SE gate, residual) |
| 3 | `Plus_KL` | ✔ | ✔ | – | – | – | + `VariationalBottleneck` (latent dim 512, KL regularization) |
| 4 | `Plus_MixStyle` | ✔ | ✔ | ✔ | – | – | + `MixStyleLayer` (p=0.5, α=0.1) |
| 5 | `Plus_Attention` | ✔ | ✔ | ✔ | ✔ | – | + `DomainAttentionGate` (channel recalibration, reduction 8) |
| 6 | `PASAN_DG` | ✔ | ✔ | ✔ | ✔ | ✔ | + Knowledge distillation from a frozen ConvNeXt-Small teacher (`convnext_small.in12k_ft_in1k`) and MixUp |

> Note: the code does **not** enforce that each step improves performance. Whether the progression is
> monotonic is an empirical result, so check `ablation_table.csv` and `paired_ttest_ablation.csv`.

---

## 2. Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.1 (needed for `torch.utils.flop_counter`)
- torchvision ≥ 0.16 (needed for `torchvision.transforms.v2`)
- `timm`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`
- `psutil` (only used by script B for the CPU memory fallback)

```bash
pip install torch torchvision timm numpy pandas scipy scikit-learn matplotlib seaborn psutil
```

Pretrained weights (`*.in12k_ft_in1k`) are downloaded by `timm` on first use, so internet access to the
Hugging Face Hub is required for all three scripts unless the weights are already cached.

---

## 3. Dataset Layout

Scripts use `torchvision.datasets.ImageFolder`, so each split needs one sub-folder per class:

```
Brain tumor dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Test/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Edit these constants at the top of **A** to match your machine:

```python
TRAIN_PATH = "/path/to/Brain tumor dataset/Training/"  (Kaggle training dataset)
TEST_PATH  = "/path/to/Brain tumor dataset/Test/"      (Mendeley testing dataset)
SAVE_DIR   = Path("PASAN_DG_RESULTS_stnv")
DEVICE     = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")   # change the GPU index as needed
```

---

## 4. Script A: `PASAN_DG_main.py` (Training, Evaluation, Statistics)

```bash
python PASAN_DG_main.py
```

### Training protocol

| Item | Value |
|------|-------|
| Seeds | 42, 43, 44, 45, 46, 47 (n = 6) |
| Validation split | 20% of `Training/`, stratified, re-drawn per seed |
| Optimizer | AdamW, lr = 1e-4, weight decay = 1e-2 |
| Scheduler | CosineAnnealingWarmRestarts (T_0 = 20, η_min = 1e-6) |
| Epochs / early stopping | 60 max, patience 12 on validation loss |
| Batch size | 16 |
| Loss | Cross-entropy, label smoothing 0.10 |
| KL weight | β = 1e-3 × min(1, epoch / 20) (linear warm-up), used for `Plus_KL` onward |
| MixUp | α = 0.4, **`PASAN_DG` only** |
| Distillation | T = 4.0, weight 0.4, **`PASAN_DG` only**; loss = 0.6·task + 0.4·KD |
| Gradient clipping | 1.0 |
| Input size | 224 × 224 |
| Model selection | Best validation-loss checkpoint is restored before testing |

### Behaviour to be aware of

- **Resumable.** `all_results.csv` is rewritten after every run, and any (model, seed) pair already in it is skipped.
- **Grad-CAM is generated for every test image** (test batch size = 1), which is slow and uses a lot of disk space. Target layers:
  - Baseline: `model.stages[-1].blocks[-1].conv_dw`
  - Other models: `bb.stages_3.blocks[-1].conv_dw`
- Grad-CAM is computed for the **ground-truth class**, not the predicted class.
- Metrics are computed on the **Test** folder only.

### Output structure

```
PASAN_DG_RESULTS_stnv/
├── <Model>/                         # one per model (6 total)
│   ├── metrics/                     # test_metrics, cm (csv+png), roc (csv+png),
│   │                                # calibration png, history csv  (per seed)
│   ├── curves/                      # training curves per seed
│   ├── gradcam/<class>/img_XXXXX.png
│   └── model_note.txt
├── all_results.csv                  # one row per (model, seed)
├── summary_aggregated.csv           # mean/std/min/max per model
├── ablation_table.csv
├── paired_ttest_ablation.csv        # step-wise paired t-tests (each model vs. the previous one)
├── wilcoxon_n6.csv                  # all pairwise Wilcoxon signed-rank tests
├── mcnemar_pooled.csv               # all pairwise McNemar tests, pooled over seeds
├── superiority_table.csv            # PASAN-DG vs. Baseline
├── cbge_estimates.csv               # concentration-based generalization estimate
└── boxplot_all_metrics.png, barplot_<metric>.png, ablation_progression.png,
    mcnemar_pvalue_heatmap.png, cohens_d_heatmap.png, ece_calibration_bar.png
```

### Reported metrics

Accuracy, Macro-F1, Macro-Precision, Macro-Recall, Macro-AUC (one-vs-rest), and ECE (10 bins, lower is better).

### Statistical analysis

- **Paired t-test** (step-wise): each model against the one before it in the ablation.
- **Wilcoxon signed-rank** (all pairs, α = 0.05): with only 6 paired seeds, the smallest possible two-sided
  p-value is 0.03125, so significance is only reachable when all six differences have the same sign.
- **McNemar's test** with continuity correction: discordant counts are pooled across seeds. Because the same
  test images are reused across seeds, the pooled test treats those repeated predictions as independent;
  interpret it as descriptive.
- **Cohen's d** (pooled SD) for effect sizes.
- **CBGE**: empirical risk + a concentration-based slack term (`n_train = 5712`, δ = 0.05). Change
  `n_train` in `compute_cbge` if your training set size differs.

---

## 5. Script B: `PASAN_DG_complexity.py` (Parameters, Latency, Memory)

```bash
python PASAN_DG_complexity.py
```

Output: `PASAN_DG_COMPLEXITY_RESULTS/complexity_table.csv`

| Column | Definition |
|--------|------------|
| `Params (M)` | All parameters of the **deployed** network (the teacher is excluded) |
| `bb (M)` | Backbone parameters (Baseline: ConvNeXt-Tiny without its `head`; others: `model.bb`) |
| `Head (M)` | `Params − bb` (spectral, MixStyle, attention, bottleneck, classifier) |
| `Teacher (M, train-only)` | ConvNeXt-Small teacher parameters, training only |
| `CPU latency (ms)` | Mean ± std over 50 timed runs after 10 warm-up runs, batch 1, 224×224, eval mode, `no_grad` |
| `Mem b1 (MB)` / `Mem b16 (MB)` | Peak inference memory (weights + activations) for batch 1 / 16 |

Memory is measured with `torch.cuda.max_memory_allocated` on GPU `MEM_CUDA_IDX` (default 0) when CUDA is
available, otherwise as the peak-RSS delta on CPU (via `psutil`, sampled every 5 ms). Latency is always on CPU.
Set `CPU_THREADS` at the top of the script to pin the thread count for reproducible latency numbers.

---

## 6. Script C: `PASAN_DG_flops.py` (FLOPs / MACs)

```bash
python PASAN_DG_flops.py
```

Output: `PASAN_DG_COMPLEXITY_RESULTS/flops_table.csv` with columns `Model`, `FLOPs (G)`, `MACs (G)`.

- Uses `torch.utils.flop_counter.FlopCounterMode`: one forward pass, a single 224×224 RGB image, CPU, eval mode.
- `FLOPs = 2 × MACs`.
- The frozen teacher is removed before counting, since it is not part of the inference path.
- FFT operations (`rfft2` / `irfft2`), normalization layers, and element-wise ops are **not** counted by
  `FlopCounterMode`. Their contribution is small relative to the convolutions, but the reported FLOPs for
  the spectral variants are therefore a slight lower bound.

---

## 7. Recommended Run Order

```bash
python PASAN_DG_main.py          # A: train and evaluate (long-running)
python PASAN_DG_complexity.py    # B: params, latency, memory
python PASAN_DG_flops.py         # C: FLOPs
```

B and C do not depend on A's results, only on its model definitions, so they can be run before or after training.

---

## 8. Reproducibility Notes

- Seeds are set for `random`, `numpy`, and `torch` (CPU and CUDA), with `cudnn.deterministic = True` and `benchmark = False`.
- Bit-exact results across different hardware or library versions are not guaranteed.
- Latency and memory in B depend on the CPU, thread count, and GPU used, so report the hardware alongside the numbers.

---

## 9. Known Limitations

- Dataset paths and GPU index are hard-coded and must be edited.
- Per-image Grad-CAM for the full test set across 6 models × 6 seeds is time- and storage-intensive.
- Some console messages contain `?` where Unicode symbols were intended; save the files as UTF-8 to avoid this.
- Also the code took 4 days to run completely, the implementation environment is specified in the manuscript, if you have other gpus, the time for completion of the code might vary.

## Citation

Cite the github link and once the manuscript gets accepted the citation link shall be provided.
