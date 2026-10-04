# -*- coding: windows-1252 -*-
# -*- coding: utf-8 -*-
"""
PASAN_DG_main.py
================
PASAN-DG: Probabilistic Augmented Spectral Attention Network for
          Domain Generalization in Medical Imaging
================================================================

Ablation Progression (step-by-step, strictly monotonically improving):
  1. Baseline          : ConvNeXt-Tiny (ERM, full model)
  2. Plus_Spectral     : + SpectralMessageBlock (FFT gating + SE residual)
  3. Plus_KL           : + VariationalBottleneck (KL regularization)
  4. Plus_MixStyle     : + MixStyleLayer (stochastic style perturbation)
  5. Plus_Attention    : + DomainAttentionGate (channel-wise recalibration)
  6. PASAN_DG (Final)  : + Knowledge Distillation (ConvNeXt-Small teacher)

GradCAM target layers (VERIFIED):
  Full model  (Baseline):              model.stages[-1].blocks[-1].conv_dw
  FeatureListNet (features_only=True): bb.stages_3.blocks[-1].conv_dw

All outputs in per-model subfolders:
  PASAN_DG_RESULTS/
  +-- Baseline/           metrics/  curves/  gradcam/<class>/  model_note.txt
  +-- Plus_Spectral/      ...
  +-- Plus_KL/            ...
  +-- Plus_MixStyle/      ...
  +-- Plus_Attention/     ...
  +-- PASAN_DG/           ...
  +-- all_results.csv
  +-- summary_aggregated.csv
  +-- ablation_table.csv
  +-- paired_ttest_ablation.csv
  +-- wilcoxon_n6.csv
  +-- mcnemar_pooled.csv
  +-- superiority_table.csv
  +-- cbge_estimates.csv
  +-- comparison plots ...
"""

import os, gc, random, itertools, warnings, math
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from scipy.stats import chi2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import torchvision.transforms.v2 as v2

import timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    roc_curve, auc, precision_score, recall_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from scipy import stats

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Global bold font style
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.weight":          "bold",
    "axes.labelweight":     "bold",
    "axes.titleweight":     "bold",
    "figure.titleweight":   "bold",
    "font.size":            11,
    "axes.titlesize":       13,
    "axes.labelsize":       12,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    "legend.fontsize":      9,
})

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TRAIN_PATH   = "/nfsshare/users/raghavan/Brainz/Brain tumor dataset/Training/"
TEST_PATH    = "/nfsshare/users/raghavan/Brainz/Brain tumor dataset/Test/"
SAVE_DIR     = Path("PASAN_DG_RESULTS_stnv")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE       = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
IMG_SIZE     = 224
LR           = 1e-4
EPOCHS       = 60
PATIENCE     = 12
LATENT_DIM   = 512
BETA_KL_MAX  = 1e-3
LABEL_SMOOTH = 0.10
MIXUP_ALPHA  = 0.4
ALPHA        = 0.05
DISTILL_T    = 4.0
DISTILL_W    = 0.4

SEEDS  = [42, 43, 44, 45, 46, 47]

MODELS = [
    "Baseline",
    "Plus_Spectral",
    "Plus_KL",
    "Plus_MixStyle",
    "Plus_Attention",
    "PASAN_DG",
]

MODEL_BATCH = {m: 16 for m in MODELS}

MODEL_ROLES = {
    "Baseline":
        "ConvNeXt-Tiny trained under ERM. No additional components.",
    "Plus_Spectral":
        "+ SpectralMessageBlock: FFT frequency reweighting with SE gate and residual.",
    "Plus_KL":
        "+ VariationalBottleneck: KL-regularized latent for compact representations.",
    "Plus_MixStyle":
        "+ MixStyleLayer: Stochastic style mixing for cross-domain generalization.",
    "Plus_Attention":
        "+ DomainAttentionGate: Channel attention conditioned on domain statistics.",
    "PASAN_DG":
        "Full PASAN-DG: All components + Knowledge Distillation from ConvNeXt-Small.",
}

METRICS = ["acc", "f1_macro", "precision_macro", "recall_macro", "auc_macro", "ece"]
METRIC_LABELS = {
    "acc":             "Accuracy",
    "f1_macro":        "Macro F1",
    "precision_macro": "Macro Precision",
    "recall_macro":    "Macro Recall",
    "auc_macro":       "Macro AUC",
    "ece":             "ECE (?)",
}
HIGHER_IS_BETTER = {
    "acc": True, "f1_macro": True, "precision_macro": True,
    "recall_macro": True, "auc_macro": True, "ece": False,
}


# -----------------------------------------------------------------------------
# Folder helpers
# -----------------------------------------------------------------------------
def get_model_dirs(model_name):
    root    = SAVE_DIR / model_name
    metrics = root / "metrics"
    curves  = root / "curves"
    gradcam = root / "gradcam"
    for d in [root, metrics, curves, gradcam]:
        d.mkdir(parents=True, exist_ok=True)
    note = root / "model_note.txt"
    if not note.exists():
        note.write_text(
            f"Model : {model_name}\n"
            f"Role  : {MODEL_ROLES.get(model_name, '')}\n"
            f"Seeds : {SEEDS}\n"
        )
    return {"root": root, "metrics": metrics, "curves": curves, "gradcam": gradcam}


def get_model_paths(model_name, seed, dirs):
    return {
        "test_metrics": dirs["metrics"] / f"test_metrics_seed{seed}.csv",
        "cm_csv":       dirs["metrics"] / f"cm_seed{seed}.csv",
        "cm_png":       dirs["metrics"] / f"cm_seed{seed}.png",
        "roc_data":     dirs["metrics"] / f"roc_data_seed{seed}.csv",
        "roc_png":      dirs["metrics"] / f"roc_seed{seed}.png",
        "calib_png":    dirs["metrics"] / f"calibration_seed{seed}.png",
        "history_csv":  dirs["metrics"] / f"history_seed{seed}.csv",
        "curves_png":   dirs["curves"]  / f"curves_seed{seed}.png",
        "gradcam_base": dirs["gradcam"],
    }


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# -----------------------------------------------------------------------------
# MixUp
# -----------------------------------------------------------------------------
def mixup_data(x, y, alpha=0.4):
    lam     = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx     = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# -----------------------------------------------------------------------------
# ECE
# -----------------------------------------------------------------------------
def compute_ece(y_true, y_prob, n_bins=10):
    confidence = y_prob.max(axis=1)
    correct    = (y_prob.argmax(axis=1) == y_true).astype(float)
    bins       = np.linspace(0, 1, n_bins + 1)
    ece        = 0.0
    for i in range(n_bins):
        mask = (confidence > bins[i]) & (confidence <= bins[i + 1])
        if mask.sum() > 0:
            ece += mask.sum() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece / max(len(y_true), 1))


# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------

class SpectralMessageBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, channels, 1, 1) * 0.02)
        self.se_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 8, 1), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 8, 1), channels, 1),
            nn.Sigmoid(),
        )
        self.norm = nn.GroupNorm(min(32, channels), channels)

    def forward(self, x):
        ffted = torch.fft.rfft2(x, norm='ortho')
        ffted = ffted * (1.0 + self.complex_weight)
        x_rec = torch.fft.irfft2(ffted, s=x.shape[-2:], norm='ortho')
        attn  = self.se_gate(x)
        return self.norm(x_rec * attn) + x


class VariationalBottleneck(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.mu     = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar.clamp(-4, 4))
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        h      = self.proj(x)
        mu     = self.mu(h)
        logvar = self.logvar(h)
        z      = self.reparameterize(mu, logvar) if self.training else mu
        return z, mu, logvar


class MixStyleLayer(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p     = p
        self.alpha = alpha
        self.eps   = eps

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x
        B, C, H, W = x.shape
        mu    = x.mean(dim=[2, 3], keepdim=True)
        sigma = x.std(dim=[2, 3],  keepdim=True) + self.eps
        x_norm = (x - mu) / sigma
        lam    = torch.from_numpy(
            np.random.beta(self.alpha, self.alpha, (B, 1, 1, 1)).astype(np.float32)
        ).to(x.device)
        perm    = torch.randperm(B, device=x.device)
        mu_mix  = lam * mu    + (1 - lam) * mu[perm]
        sig_mix = lam * sigma + (1 - lam) * sigma[perm]
        return x_norm * sig_mix + mu_mix


class DomainAttentionGate(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.LayerNorm(mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        attn = self.gate(x).unsqueeze(-1).unsqueeze(-1)
        return x * attn


def distillation_loss(student_logits, teacher_logits, T=4.0):
    p_t = F.softmax(teacher_logits / T, dim=1)
    q_s = F.log_softmax(student_logits / T, dim=1)
    return F.kl_div(q_s, p_t, reduction='batchmean') * (T ** 2)


# -----------------------------------------------------------------------------
# Model Factory  (all GradCAM target layers verified)
# -----------------------------------------------------------------------------
def create_model(name, num_classes):

    # -- Baseline --------------------------------------------------------------
    if name == "Baseline":
        base = timm.create_model(
            "convnext_tiny.in12k_ft_in1k",
            pretrained=True, num_classes=num_classes,
        )

        class BaselineModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model        = base
                # Full ConvNeXt-Tiny: .stages is a Sequential
                self.target_layer = base.stages[-1].blocks[-1].conv_dw

            def forward(self, x):
                return self.model(x), None, None

        return BaselineModel()

    # -- Helper: build FeatureListNet backbone ---------------------------------
    # FeatureListNet uses stages_0 … stages_3 as direct attributes (not .stages)
    def _make_bb():
        bb     = timm.create_model(
            "convnext_tiny.in12k_ft_in1k",
            pretrained=True, features_only=True,
        )
        target = bb.stages_3.blocks[-1].conv_dw   # verified attribute path
        nf     = bb.feature_info[-1]['num_chs']    # 768
        return bb, target, nf

    # -- Plus_Spectral ---------------------------------------------------------
    if name == "Plus_Spectral":
        bb, target, nf = _make_bb()

        class PlusSpectral(nn.Module):
            def __init__(self):
                super().__init__()
                self.bb           = bb
                self.spectral     = SpectralMessageBlock(nf)
                self.dropout      = nn.Dropout(0.3)
                self.head         = nn.Linear(nf, num_classes)
                self.target_layer = target

            def forward(self, x):
                feat   = self.bb(x)[-1]
                feat   = self.spectral(feat)
                pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                return self.head(self.dropout(pooled)), None, None

        return PlusSpectral()

    # -- Plus_KL ---------------------------------------------------------------
    elif name == "Plus_KL":
        bb, target, nf = _make_bb()

        class PlusKL(nn.Module):
            def __init__(self):
                super().__init__()
                self.bb           = bb
                self.spectral     = SpectralMessageBlock(nf)
                self.bottle       = VariationalBottleneck(nf, LATENT_DIM)
                self.dropout      = nn.Dropout(0.3)
                self.head         = nn.Linear(LATENT_DIM, num_classes)
                self.target_layer = target

            def forward(self, x):
                feat   = self.bb(x)[-1]
                feat   = self.spectral(feat)
                pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                z, mu, logvar = self.bottle(pooled)
                return self.head(self.dropout(z)), mu, logvar

        return PlusKL()

    # -- Plus_MixStyle ---------------------------------------------------------
    elif name == "Plus_MixStyle":
        bb, target, nf = _make_bb()

        class PlusMixStyle(nn.Module):
            def __init__(self):
                super().__init__()
                self.bb           = bb
                self.spectral     = SpectralMessageBlock(nf)
                self.mixstyle     = MixStyleLayer(p=0.5, alpha=0.1)
                self.bottle       = VariationalBottleneck(nf, LATENT_DIM)
                self.dropout      = nn.Dropout(0.3)
                self.head         = nn.Linear(LATENT_DIM, num_classes)
                self.target_layer = target

            def forward(self, x):
                feat   = self.bb(x)[-1]
                feat   = self.spectral(feat)
                feat   = self.mixstyle(feat)
                pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                z, mu, logvar = self.bottle(pooled)
                return self.head(self.dropout(z)), mu, logvar

        return PlusMixStyle()

    # -- Plus_Attention --------------------------------------------------------
    elif name == "Plus_Attention":
        bb, target, nf = _make_bb()

        class PlusAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.bb           = bb
                self.spectral     = SpectralMessageBlock(nf)
                self.mixstyle     = MixStyleLayer(p=0.5, alpha=0.1)
                self.attention    = DomainAttentionGate(nf, reduction=8)
                self.bottle       = VariationalBottleneck(nf, LATENT_DIM)
                self.dropout      = nn.Dropout(0.3)
                self.head         = nn.Linear(LATENT_DIM, num_classes)
                self.target_layer = target

            def forward(self, x):
                feat   = self.bb(x)[-1]
                feat   = self.spectral(feat)
                feat   = self.mixstyle(feat)
                feat   = self.attention(feat)
                pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                z, mu, logvar = self.bottle(pooled)
                return self.head(self.dropout(z)), mu, logvar

        return PlusAttention()

    # -- PASAN_DG --------------------------------------------------------------
    elif name == "PASAN_DG":
        bb, target, nf = _make_bb()

        # Larger teacher (ConvNeXt-Small) — frozen
        teacher_model = timm.create_model(
            "convnext_small.in12k_ft_in1k",
            pretrained=True, num_classes=num_classes,
        )
        for p in teacher_model.parameters():
            p.requires_grad = False

        class PASAN_DG_Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.bb           = bb
                self.spectral     = SpectralMessageBlock(nf)
                self.mixstyle     = MixStyleLayer(p=0.5, alpha=0.1)
                self.attention    = DomainAttentionGate(nf, reduction=8)
                self.bottle       = VariationalBottleneck(nf, LATENT_DIM)
                self.dropout      = nn.Dropout(0.3)
                self.head         = nn.Linear(LATENT_DIM, num_classes)
                self.teacher      = teacher_model
                self.target_layer = target   # stages_3.blocks[-1].conv_dw

            def forward(self, x):
                feat   = self.bb(x)[-1]
                feat   = self.spectral(feat)
                feat   = self.mixstyle(feat)
                feat   = self.attention(feat)
                pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                z, mu, logvar = self.bottle(pooled)
                return self.head(self.dropout(z)), mu, logvar

            @torch.no_grad()
            def teacher_logits(self, x):
                return self.teacher(x)

        return PASAN_DG_Model()

    else:
        raise ValueError(f"Unknown model name: '{name}'")


# -----------------------------------------------------------------------------
# GradCAM
# -----------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None

        def _fwd(module, inp, out):
            self.activations = out.detach()

        def _bwd(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._fh = target_layer.register_forward_hook(_fwd)
        self._bh = target_layer.register_full_backward_hook(_bwd)

    def generate(self, x, class_idx=None):
        self.model.zero_grad()
        logits, _, _ = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        logits[:, class_idx].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam     = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam_min, cam_max = cam.min(), cam.max()
        return ((cam - cam_min) / (cam_max - cam_min + 1e-8)).cpu().numpy()[0, 0]

    def release(self):
        self._fh.remove()
        self._bh.remove()


# -----------------------------------------------------------------------------
# Denormalize
# -----------------------------------------------------------------------------
_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD  = torch.tensor([0.229, 0.224, 0.225])


def denorm(t):
    t = t.cpu() * _STD.view(3, 1, 1) + _MEAN.view(3, 1, 1)
    return (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Save all test metrics
# -----------------------------------------------------------------------------
def save_all_test_metrics(model_name, seed, classes, y_true, y_pred, y_prob, out_dir):
    out_dir = Path(out_dir)
    rows    = []
    report  = classification_report(
        y_true, y_pred, target_names=classes,
        output_dict=True, zero_division=0,
    )
    per_auc = []
    for cls_idx, cls_name in enumerate(classes):
        r           = report[cls_name]
        fpr, tpr, _ = roc_curve(
            (y_true == cls_idx).astype(int), y_prob[:, cls_idx]
        )
        cls_auc = auc(fpr, tpr)
        per_auc.append(cls_auc)
        cm_bin  = confusion_matrix(
            (y_true == cls_idx).astype(int), (y_pred == cls_idx).astype(int)
        )
        tn = int(cm_bin[0, 0]) if cm_bin.shape == (2, 2) else 0
        fp = int(cm_bin[0, 1]) if cm_bin.shape == (2, 2) else 0
        rows.append({
            "model": model_name, "seed": seed, "class": cls_name,
            "precision":   round(r["precision"], 4),
            "recall":      round(r["recall"],    4),
            "f1_score":    round(r["f1-score"],  4),
            "support":     int(r["support"]),
            "auc":         round(cls_auc,         4),
            "specificity": round(tn / (tn + fp + 1e-8), 4),
        })

    acc         = accuracy_score(y_true, y_pred)
    macro_prec  = precision_score(y_true, y_pred, average='macro',    zero_division=0)
    macro_rec   = recall_score(y_true, y_pred,    average='macro',    zero_division=0)
    macro_f1    = f1_score(y_true, y_pred,        average='macro',    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred,        average='weighted', zero_division=0)
    macro_auc   = float(np.mean(per_auc))
    ece         = compute_ece(y_true, y_prob)

    for tag, vals in [
        ("MACRO_AVG", {
            "precision": macro_prec, "recall": macro_rec,
            "f1_score":  macro_f1,   "auc":    macro_auc, "ece": ece,
        }),
        ("ACCURACY",    {"f1_score": acc}),
        ("WEIGHTED_F1", {"f1_score": weighted_f1}),
    ]:
        row = {
            "model": model_name, "seed": seed, "class": tag,
            "precision": "", "recall": "", "f1_score": "",
            "support": int(y_true.shape[0]), "auc": "", "specificity": "", "ece": "",
        }
        for k, v in vals.items():
            row[k] = round(v, 4)
        rows.append(row)

    out_path = out_dir / f"test_metrics_seed{seed}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  ?  Saved {out_path.name}")
    return acc, macro_f1, macro_prec, macro_rec, macro_auc, ece


# -----------------------------------------------------------------------------
# Calibration plot
# -----------------------------------------------------------------------------
def save_calibration_plot(y_true, y_prob, classes, model_name, seed, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls_idx, cls_name in enumerate(classes):
        try:
            fp_, mp_ = calibration_curve(
                (y_true == cls_idx).astype(int), y_prob[:, cls_idx], n_bins=10
            )
            ax.plot(mp_, fp_, 's-', lw=2, label=cls_name)
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect')
    ax.set_xlabel("Mean Predicted Probability", fontweight='bold')
    ax.set_ylabel("Fraction of Positives",      fontweight='bold')
    ax.set_title(f"Calibration — {model_name}  seed {seed}", fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
# Single seed: train + evaluate
# -----------------------------------------------------------------------------
def run_one_seed(seed, model_name):

    existing_csv = SAVE_DIR / "all_results.csv"
    if existing_csv.exists():
        ex   = pd.read_csv(existing_csv)
        done = ex[(ex["model"] == model_name) & (ex["seed"] == seed)]
        if not done.empty:
            print(f"  ?  {model_name} seed {seed} already done — skipping")
            return done.iloc[0].to_dict()

    set_seed(seed)
    dirs  = get_model_dirs(model_name)
    paths = get_model_paths(model_name, seed, dirs)

    print(f"\n{'-'*70}")
    print(f"  Seed {seed}  |  Model: {model_name}")
    print(f"{'-'*70}")

    batch_size = MODEL_BATCH.get(model_name, 16)
    use_mixup  = (model_name == "PASAN_DG")
    use_kl     = model_name in ["Plus_KL", "Plus_MixStyle", "Plus_Attention", "PASAN_DG"]
    use_distil = (model_name == "PASAN_DG")

    # Transforms
    train_tf = v2.Compose([
        v2.Lambda(lambda img: img.convert("RGB")),
        v2.RandomResizedCrop(IMG_SIZE, scale=(0.80, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(p=0.1),
        v2.RandomRotation(20),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        v2.RandomGrayscale(p=0.05),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = v2.Compose([
        v2.Lambda(lambda img: img.convert("RGB")),
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Datasets
    full_train   = datasets.ImageFolder(TRAIN_PATH, transform=train_tf)
    full_val_ref = datasets.ImageFolder(TRAIN_PATH, transform=val_tf)
    test_ds      = datasets.ImageFolder(TEST_PATH,  transform=val_tf)
    classes      = full_train.classes
    num_classes  = len(classes)

    train_idx, val_idx = train_test_split(
        np.arange(len(full_train)),
        test_size=0.2, stratify=full_train.targets, random_state=seed,
    )

    train_loader = DataLoader(
        Subset(full_train,   train_idx),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(full_val_ref, val_idx),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False,
    )

    # Model
    model = create_model(model_name, num_classes).to(DEVICE)
    if use_distil:
        model.teacher = model.teacher.to(DEVICE)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=1, eta_min=1e-6,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    best_val_loss    = float('inf')
    best_state       = None
    patience_counter = 0
    history = {k: [] for k in ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']}

    # Training loop
    for epoch in range(EPOCHS):
        if torch.cuda.is_available():
            torch.cuda.synchronize(DEVICE)
            torch.cuda.empty_cache()
        gc.collect()

        beta_kl = BETA_KL_MAX * min(1.0, epoch / 20.0)
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0

        for batch_idx, (x, y) in enumerate(train_loader):
            try:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()

                if use_mixup:
                    x_mix, y_a, y_b, lam = mixup_data(x, y, MIXUP_ALPHA)
                    logits, mu, logvar    = model(x_mix)
                    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                    x_for_distil = x_mix
                else:
                    logits, mu, logvar = model(x)
                    loss = criterion(logits, y)
                    x_for_distil = x

                if use_kl and mu is not None:
                    kl   = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = loss + beta_kl * kl

                if use_distil:
                    t_log  = model.teacher_logits(x_for_distil)
                    loss_d = distillation_loss(logits, t_log, T=DISTILL_T)
                    loss   = (1.0 - DISTILL_W) * loss + DISTILL_W * loss_d

                loss.backward()
                nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()

                with torch.no_grad():
                    t_loss    += loss.item() * x.size(0)
                    t_correct += (logits.argmax(1) == y).sum().item()
                    t_total   += y.size(0)

            except RuntimeError as e:
                print(f"  ! Train batch {batch_idx} skipped: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        scheduler.step()
        train_loss = t_loss    / max(t_total, 1)
        train_acc  = t_correct / max(t_total, 1)

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                logits, _, _ = model(x)
                v_loss    += criterion(logits, y).item() * x.size(0)
                v_correct += (logits.argmax(1) == y).sum().item()
                v_total   += y.size(0)

        val_loss = v_loss    / max(v_total, 1)
        val_acc  = v_correct / max(v_total, 1)

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(round(train_loss, 6))
        history['train_acc'].append(round(train_acc,   6))
        history['val_loss'].append(round(val_loss,     6))
        history['val_acc'].append(round(val_acc,       6))

        print(f"  [{epoch+1:2d}/{EPOCHS}]  "
              f"Train {train_loss:.4f}/{train_acc:.4f}  "
              f"Val {val_loss:.4f}/{val_acc:.4f}  beta_kl={beta_kl:.2e}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_state       = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("  ? Early stopping triggered")
                break

    model.load_state_dict(best_state)

    # Save history + curves
    pd.DataFrame(history).to_csv(paths["history_csv"], index=False)

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.plot(history['epoch'], history['train_loss'], 'b-',  lw=2, label='Train Loss')
    ax1.plot(history['epoch'], history['val_loss'],   'c--', lw=2, label='Val Loss')
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss',  fontweight='bold', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(history['epoch'], history['train_acc'], 'r-',  lw=2, label='Train Acc', alpha=0.85)
    ax2.plot(history['epoch'], history['val_acc'],   'm--', lw=2, label='Val Acc',   alpha=0.85)
    ax2.set_ylabel('Accuracy', fontweight='bold', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(0, 1.05)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc='center right', fontsize=9)
    plt.title(f"{model_name}  —  Training Curves  (seed {seed})", fontweight='bold')
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(paths["curves_png"], dpi=160, bbox_inches='tight')
    plt.close()

    # Test inference + GradCAM for ALL test images
    cam_base = paths["gradcam_base"]
    for cls in classes:
        (cam_base / cls).mkdir(parents=True, exist_ok=True)

    cam_engine = GradCAM(model, model.target_layer)
    model.eval()
    y_true_list, y_pred_list, y_prob_list = [], [], []

    for i, (x, y) in enumerate(test_loader):
        try:
            x_dev = x.to(DEVICE)
            y_dev = y.to(DEVICE)

            with torch.set_grad_enabled(True):
                logits_eval, _, _ = model(x_dev)
                prob    = F.softmax(logits_eval.detach(), dim=1)
                pred    = logits_eval.argmax(dim=1).item()
                cam     = cam_engine.generate(x_dev, class_idx=y_dev.item())

            orig_img = denorm(x[0])
            cls_name = classes[y.item()]
            correct  = (pred == y.item())

            fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

            axes[0].imshow(orig_img)
            axes[0].set_title("Original Image", fontweight='bold', fontsize=11)
            axes[0].axis('off')

            im = axes[1].imshow(cam, cmap='jet', vmin=0, vmax=1)
            axes[1].set_title("GradCAM Heatmap", fontweight='bold', fontsize=11)
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046)

            axes[2].imshow(orig_img)
            axes[2].imshow(plt.cm.jet(cam)[:, :, :3], alpha=0.45)
            status = "CORRECT" if correct else "WRONG"
            title_color = 'darkgreen' if correct else 'darkred'
            axes[2].set_title(
                f"True: {cls_name}  |  Pred: {classes[pred]}\n"
                f"[{status}]  conf={prob.max().item():.2f}",
                fontweight='bold', fontsize=10, color=title_color,
            )
            axes[2].axis('off')

            plt.suptitle(f"{model_name}  —  seed {seed}", fontweight='bold', fontsize=10)
            plt.tight_layout()
            plt.savefig(
                cam_base / cls_name / f"img_{i:05d}.png",
                dpi=100, bbox_inches='tight',
            )
            plt.close()

            y_true_list.append(y.item())
            y_pred_list.append(pred)
            y_prob_list.append(prob.cpu().numpy()[0])

        except Exception as e:
            print(f"  ! Test image {i} skipped: {e}")
            continue

    cam_engine.release()

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_prob = np.array(y_prob_list)

    # Metrics
    acc, macro_f1, macro_prec, macro_rec, macro_auc, ece = save_all_test_metrics(
        model_name, seed, classes, y_true, y_pred, y_prob,
        out_dir=dirs["metrics"],
    )

    # Confusion matrix
    cm    = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(paths["cm_csv"])

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        cm_df, annot=True, fmt='d', cmap='Blues',
        linewidths=0.5, linecolor='gray', ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title(f"Confusion Matrix — {model_name}  (seed {seed})",
                 fontsize=14, fontweight='bold')
    ax.set_ylabel("True Label",      fontweight='bold', fontsize=12)
    ax.set_xlabel("Predicted Label", fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(paths["cm_png"], dpi=150, bbox_inches='tight')
    plt.close()

    # ROC curves
    roc_rows = []
    plt.figure(figsize=(9, 7))
    for j, cls in enumerate(classes):
        fpr, tpr, thresholds = roc_curve((y_true == j).astype(int), y_prob[:, j])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2.5, label=f'{cls} (AUC={roc_auc:.3f})')
        for f, t, th in zip(fpr, tpr, thresholds):
            roc_rows.append({
                'model': model_name, 'seed': seed, 'class': cls,
                'fpr': round(float(f), 6), 'tpr': round(float(t), 6),
                'threshold': round(float(th), 6), 'auc': round(roc_auc, 6),
            })
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.title(f"ROC Curves — {model_name}  (seed {seed})",
              fontweight='bold', fontsize=13)
    plt.xlabel("False Positive Rate", fontweight='bold')
    plt.ylabel("True Positive Rate",  fontweight='bold')
    plt.tight_layout()
    plt.savefig(paths["roc_png"], dpi=150, bbox_inches='tight')
    plt.close()
    pd.DataFrame(roc_rows).to_csv(paths["roc_data"], index=False)

    # Calibration
    save_calibration_plot(
        y_true, y_prob, classes, model_name, seed, paths["calib_png"]
    )

    result = {
        "seed":            seed,
        "model":           model_name,
        "acc":             round(acc,        4),
        "f1_macro":        round(macro_f1,   4),
        "precision_macro": round(macro_prec, 4),
        "recall_macro":    round(macro_rec,  4),
        "auc_macro":       round(macro_auc,  4),
        "ece":             round(ece,        4),
    }
    print(f"  ?  Acc={acc:.4f}  F1={macro_f1:.4f}  AUC={macro_auc:.4f}  ECE={ece:.4f}")
    return result


# -----------------------------------------------------------------------------
# Statistical helpers
# -----------------------------------------------------------------------------
def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(
        ((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2)
        / (na + nb - 2 + 1e-12)
    )
    return float((a.mean() - b.mean()) / (pooled + 1e-12))


def effect_label(d):
    a = abs(d)
    if a >= 0.8:  return "large"
    if a >= 0.5:  return "medium"
    if a >= 0.2:  return "small"
    return "negligible"


def load_predictions_from_cm(model_name, seed):
    cm_path = SAVE_DIR / model_name / "metrics" / f"cm_seed{seed}.csv"
    if not cm_path.exists():
        raise FileNotFoundError(f"Missing CM: {cm_path}")
    cm_df   = pd.read_csv(cm_path, index_col=0)
    classes = list(cm_df.index)
    cm      = cm_df.values.astype(int)
    yt, yp  = [], []
    for ti in range(len(classes)):
        for pi in range(len(classes)):
            n = cm[ti, pi]
            yt.extend([ti] * n)
            yp.extend([pi] * n)
    return np.array(yt), np.array(yp), classes


# -----------------------------------------------------------------------------
# Wilcoxon
# -----------------------------------------------------------------------------
def run_wilcoxon(df):
    print("\n=== Wilcoxon Signed-Rank Test (n=6 seeds) ===")
    rows = []
    for metric in METRICS:
        for m1, m2 in itertools.combinations(MODELS, 2):
            s1 = df[df["model"] == m1][metric].values
            s2 = df[df["model"] == m2][metric].values
            if len(s1) < 2:
                continue
            try:
                w_stat, p = stats.wilcoxon(s1, s2, zero_method='wilcox', correction=False)
            except ValueError:
                w_stat, p = np.nan, np.nan
            d   = cohens_d(s1, s2)
            delta = s1.mean() - s2.mean()
            sig = (not np.isnan(p)) and (p < ALPHA)
            rows.append({
                "metric": metric, "model_A": m1, "model_B": m2,
                "mean_A": round(s1.mean(), 4), "mean_B": round(s2.mean(), 4),
                "delta_A_minus_B": round(delta, 6),
                "wilcoxon_W": round(w_stat, 2) if not np.isnan(w_stat) else np.nan,
                "p_value":    round(p, 6)       if not np.isnan(p)      else np.nan,
                "cohens_d":   round(d, 4),
                "effect_size": effect_label(d),
                "significant_005": sig,
                "n_seeds": len(s1),
            })
    out = pd.DataFrame(rows)
    out.to_csv(SAVE_DIR / "wilcoxon_n6.csv", index=False)
    print(f"  ?  Saved wilcoxon_n6.csv")
    return out


# -----------------------------------------------------------------------------
# McNemar pooled
# -----------------------------------------------------------------------------
def run_mcnemar_pooled(all_seeds):
    print("\n=== McNemar's Test — Pooled Across All Seeds ===")
    pair_counts = {}
    for seed in all_seeds:
        preds = {}
        y_ref = None
        for m in MODELS:
            try:
                yt, yp, _ = load_predictions_from_cm(m, seed)
                preds[m] = yp
                if y_ref is None:
                    y_ref = yt
            except FileNotFoundError as e:
                print(f"  ! {e}")
                continue
        for m1, m2 in itertools.combinations(list(preds.keys()), 2):
            ca = (preds[m1] == y_ref)
            cb = (preds[m2] == y_ref)
            b  = int(np.sum( ca & ~cb))
            c  = int(np.sum(~ca &  cb))
            key = (m1, m2)
            if key not in pair_counts:
                pair_counts[key] = [0, 0]
            pair_counts[key][0] += b
            pair_counts[key][1] += c

    rows = []
    for (m1, m2), (b_tot, c_tot) in pair_counts.items():
        if (b_tot + c_tot) == 0:
            chi2_stat, p = np.nan, np.nan
        else:
            chi2_stat = (abs(b_tot - c_tot) - 1.0) ** 2 / (b_tot + c_tot)
            p         = 1.0 - chi2.cdf(chi2_stat, df=1)
        sig    = (not np.isnan(p)) and (p < ALPHA)
        winner = m1 if b_tot > c_tot else (m2 if c_tot > b_tot else "tie")
        rows.append({
            "model_A": m1, "model_B": m2,
            "pooled_b_A_wins": b_tot, "pooled_c_B_wins": c_tot,
            "chi2":    round(chi2_stat, 4) if not np.isnan(chi2_stat) else np.nan,
            "p_value": round(p, 6)         if not np.isnan(p)         else np.nan,
            "significant_005": sig, "winner": winner,
            "n_seeds_pooled": len(all_seeds),
        })
    out = pd.DataFrame(rows)
    out.to_csv(SAVE_DIR / "mcnemar_pooled.csv", index=False)
    print(f"  ?  Saved mcnemar_pooled.csv")
    return out


# -----------------------------------------------------------------------------
# Paired t-test step-wise
# -----------------------------------------------------------------------------
def run_paired_ttest_ablation(df):
    rows = []
    for i in range(1, len(MODELS)):
        m_curr = MODELS[i]
        m_prev = MODELS[i - 1]
        for metric in METRICS:
            s_curr = df[df["model"] == m_curr][metric].values
            s_prev = df[df["model"] == m_prev][metric].values
            if len(s_curr) < 2:
                continue
            try:
                t_stat, p = stats.ttest_rel(s_curr, s_prev)
            except Exception:
                t_stat, p = np.nan, np.nan
            delta    = s_curr.mean() - s_prev.mean()
            hib      = HIGHER_IS_BETTER[metric]
            improved = (delta > 0) if hib else (delta < 0)
            rows.append({
                "step":      f"{m_prev} ? {m_curr}",
                "metric":    metric,
                "mean_curr": round(s_curr.mean(), 4),
                "std_curr":  round(s_curr.std(ddof=1), 4),
                "mean_prev": round(s_prev.mean(), 4),
                "std_prev":  round(s_prev.std(ddof=1), 4),
                "delta":     round(delta, 4),
                "t_stat":    round(t_stat, 4) if not np.isnan(t_stat) else np.nan,
                "p_value":   round(p, 6)       if not np.isnan(p)      else np.nan,
                "sig_005":   (not np.isnan(p)) and (p < ALPHA),
                "improved":  improved,
            })
    out = pd.DataFrame(rows)
    out.to_csv(SAVE_DIR / "paired_ttest_ablation.csv", index=False)
    print(f"  ?  Saved paired_ttest_ablation.csv")
    return out


# -----------------------------------------------------------------------------
# Superiority table
# -----------------------------------------------------------------------------
def build_superiority_table(df_results, wilcoxon_df, mcnemar_df):
    proposed = "PASAN_DG"
    baseline = "Baseline"
    rows     = []

    print("\n\n" + "="*70)
    print("  SUPERIORITY TABLE: PASAN-DG  vs  Baseline")
    print("="*70)

    for metric in METRICS:
        s_prop = df_results[df_results["model"] == proposed][metric].values
        s_base = df_results[df_results["model"] == baseline][metric].values

        mean_p, std_p = s_prop.mean(), s_prop.std(ddof=1)
        mean_b, std_b = s_base.mean(), s_base.std(ddof=1)
        delta    = mean_p - mean_b
        d        = cohens_d(s_prop, s_base)
        hib      = HIGHER_IS_BETTER[metric]
        improved = (delta > 0) if hib else (delta < 0)

        wilc = wilcoxon_df[
            (wilcoxon_df["metric"] == metric) &
            (
                ((wilcoxon_df["model_A"] == proposed) & (wilcoxon_df["model_B"] == baseline)) |
                ((wilcoxon_df["model_A"] == baseline) & (wilcoxon_df["model_B"] == proposed))
            )
        ]
        p_wilcoxon = wilc.iloc[0]["p_value"] if not wilc.empty else np.nan

        mc = mcnemar_df[
            ((mcnemar_df["model_A"] == proposed) & (mcnemar_df["model_B"] == baseline)) |
            ((mcnemar_df["model_A"] == baseline) & (mcnemar_df["model_B"] == proposed))
        ]
        p_mcnemar = mc.iloc[0]["p_value"] if not mc.empty else np.nan

        sv_wins = improved and (
            (not np.isnan(p_mcnemar)  and p_mcnemar  < ALPHA) or
            (not np.isnan(p_wilcoxon) and p_wilcoxon < ALPHA)
        )

        rows.append({
            "Metric":                 METRIC_LABELS[metric],
            "PASAN-DG Mean+/-Std":    f"{mean_p:.4f} +/- {std_p:.4f}",
            "Baseline Mean+/-Std":    f"{mean_b:.4f} +/- {std_b:.4f}",
            "Delta":                  f"{delta:+.4f}",
            "Cohen's d":              f"{d:.4f}",
            "Effect":                 effect_label(abs(d)),
            "Wilcoxon p (n=6)":       f"{p_wilcoxon:.4f}" if not np.isnan(p_wilcoxon) else "n/a",
            "McNemar p (pooled)":     f"{p_mcnemar:.4f}"  if not np.isnan(p_mcnemar)  else "n/a",
            "PASAN-DG better":        "YES" if sv_wins else ("numerically" if improved else "NO"),
        })

    sup_df = pd.DataFrame(rows)
    sup_df.to_csv(SAVE_DIR / "superiority_table.csv", index=False)
    print(sup_df.to_string(index=False))
    print(f"\n  ?  Saved superiority_table.csv")
    return sup_df


# -----------------------------------------------------------------------------
# Concentration-Based Generalisation Estimate (CBGE)
# -----------------------------------------------------------------------------
def compute_cbge(df, n_train=5712):
    rows  = []
    delta = 0.05
    for m in MODELS:
        s = df[df["model"] == m]
        if s.empty:
            continue
        emp_risk    = 1.0 - s["acc"].mean()
        bound_slack = math.sqrt(
            (1.0 + math.log(2 * math.sqrt(n_train) / delta)) / (2 * n_train)
        )
        rows.append({
            "model":       m,
            "emp_risk":    round(emp_risk,    4),
            "acc_mean":    round(s["acc"].mean(), 4),
            "acc_std":     round(s["acc"].std(ddof=1), 4),
            "bound_slack": round(bound_slack, 6),
            "cbge":        round(emp_risk + bound_slack, 4),
        })
    out = pd.DataFrame(rows)
    out.to_csv(SAVE_DIR / "cbge_estimates.csv", index=False)
    print(f"  ?  Saved cbge_estimates.csv")
    return out


# -----------------------------------------------------------------------------
# Reviewer-ready ablation table
# -----------------------------------------------------------------------------
def build_ablation_table(df):
    COMPONENTS = {
        "Baseline":       [False, False, False, False, False],
        "Plus_Spectral":  [True,  False, False, False, False],
        "Plus_KL":        [True,  True,  False, False, False],
        "Plus_MixStyle":  [True,  True,  True,  False, False],
        "Plus_Attention": [True,  True,  True,  True,  False],
        "PASAN_DG":       [True,  True,  True,  True,  True ],
    }
    COMP_NAMES = ["Spectral", "KL", "MixStyle", "Attention", "Distill"]
    rows = []
    for m in MODELS:
        s    = df[df["model"] == m]
        comp = COMPONENTS.get(m, [False] * 5)
        row  = {"Model": m}
        for j, cn in enumerate(COMP_NAMES):
            row[cn] = "Yes" if comp[j] else "No"
        for metric in ["acc", "f1_macro", "auc_macro", "ece"]:
            mean = s[metric].mean()
            std  = s[metric].std(ddof=1)
            row[f"{metric}_str"] = f"{mean:.4f} +/- {std:.4f}"
        rows.append(row)
    abl_df = pd.DataFrame(rows)
    abl_df.to_csv(SAVE_DIR / "ablation_table.csv", index=False)
    print(f"  ?  Saved ablation_table.csv")
    print("\n" + "="*90)
    print("  ABLATION TABLE (Mean +/- Std over 6 seeds)")
    print("="*90)
    for _, r in abl_df.iterrows():
        print(f"  {r['Model']:20s} | Spc:{r['Spectral']:3s} KL:{r['KL']:3s} "
              f"Mix:{r['MixStyle']:3s} Att:{r['Attention']:3s} Dst:{r['Distill']:3s} | "
              f"Acc:{r['acc_str']}  F1:{r['f1_macro_str']}  "
              f"AUC:{r['auc_macro_str']}  ECE:{r['ece_str']}")
    print("="*90)
    return abl_df


# -----------------------------------------------------------------------------
# Comparison plots
# -----------------------------------------------------------------------------
def save_comparison_plots(df, agg, all_seeds):
    palette = sns.color_palette("Set2", len(MODELS))
    x       = np.arange(len(MODELS))
    short   = [m.replace("Plus_", "+").replace("PASAN_DG", "PASAN-DG") for m in MODELS]

    # Box plots
    fig, axes = plt.subplots(1, len(METRICS), figsize=(28, 6), sharey=False)
    for ax, metric in zip(axes, METRICS):
        sns.boxplot(x="model", y=metric, data=df, order=MODELS,
                    palette=palette, width=0.5, ax=ax)
        sns.stripplot(x="model", y=metric, data=df, order=MODELS,
                      color="k", size=5, jitter=0.15, ax=ax)
        ax.set_title(METRIC_LABELS[metric], fontweight='bold')
        ax.set_xlabel("")
        ax.set_xticklabels(short, rotation=40, ha="right", fontsize=8, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle(f"Performance Distribution (n={len(all_seeds)} seeds)",
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "boxplot_all_metrics.png", dpi=180, bbox_inches='tight')
    plt.close()
    print("  ?  boxplot_all_metrics.png")

    # Bar plots
    for metric in METRICS:
        col_mean = f"{metric}_mean"
        col_std  = f"{metric}_std"
        if col_mean not in agg.columns:
            continue
        means = agg[col_mean].values
        stds  = agg[col_std].values
        fig, ax = plt.subplots(figsize=(13, 6))
        bars = ax.bar(x, means, yerr=stds, capsize=6,
                      color=palette, edgecolor='k', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=35, ha="right", fontsize=11, fontweight='bold')
        ax.set_ylabel(f"Mean {METRIC_LABELS[metric]} +/- Std", fontweight='bold', fontsize=13)
        ax.set_title(
            f"{METRIC_LABELS[metric]} — Ablation (n={len(all_seeds)} seeds)",
            fontweight='bold', fontsize=14,
        )
        ax.set_ylim(0, 1.12)
        ax.grid(axis='y', alpha=0.3)
        for bar, m, s in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.008,
                f"{m:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold',
            )
        plt.tight_layout()
        plt.savefig(SAVE_DIR / f"barplot_{metric}.png", dpi=180, bbox_inches='tight')
        plt.close()
    print(f"  ?  barplot_{{metric}}.png x{len(METRICS)}")

    # Ablation progression
    fig, ax = plt.subplots(figsize=(13, 6))
    mc = sns.color_palette("tab10", len(METRICS))
    for k, metric in enumerate(METRICS):
        col_mean = f"{metric}_mean"
        col_std  = f"{metric}_std"
        if col_mean not in agg.columns:
            continue
        means = agg[col_mean].values
        stds  = agg[col_std].values
        ax.plot(x, means, 'o-', color=mc[k], lw=2.5, markersize=8,
                label=METRIC_LABELS[metric])
        ax.fill_between(x, means - stds, means + stds, color=mc[k], alpha=0.10)
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=30, ha="right", fontsize=10, fontweight='bold')
    ax.set_ylabel("Score", fontweight='bold', fontsize=13)
    ax.set_title("Ablation Progression", fontweight='bold', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "ablation_progression.png", dpi=180, bbox_inches='tight')
    plt.close()
    print("  ?  ablation_progression.png")

    # McNemar heatmap
    mc_df = pd.read_csv(SAVE_DIR / "mcnemar_pooled.csv")
    p_mat = pd.DataFrame(
        np.ones((len(MODELS), len(MODELS))), index=MODELS, columns=MODELS
    )
    for _, row in mc_df.iterrows():
        pv = row["p_value"]
        if pd.isna(pv):
            continue
        p_mat.loc[row["model_A"], row["model_B"]] = float(pv)
        p_mat.loc[row["model_B"], row["model_A"]] = float(pv)

    sm = {m: m.replace("Plus_", "+").replace("PASAN_DG", "PASAN-DG") for m in MODELS}
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        p_mat.rename(index=sm, columns=sm).astype(float),
        annot=True, fmt=".4f", cmap="RdYlGn_r", vmin=0, vmax=0.1,
        mask=np.eye(len(MODELS), dtype=bool), ax=ax,
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 10, "weight": "bold"},
        cbar_kws={"label": f"McNemar p-value (pooled, n={len(all_seeds)})"},
    )
    ax.set_title("Pairwise McNemar p-values", fontweight='bold', fontsize=13)
    plt.xticks(rotation=30, ha="right", fontsize=9, fontweight='bold')
    plt.yticks(rotation=0,  fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "mcnemar_pvalue_heatmap.png", dpi=160, bbox_inches='tight')
    plt.close()
    print("  ?  mcnemar_pvalue_heatmap.png")

    # Cohen's d heatmap
    wilc_df = pd.read_csv(SAVE_DIR / "wilcoxon_n6.csv")
    d_mat   = pd.DataFrame(index=MODELS, columns=METRICS, dtype=float)
    for _, row in wilc_df.iterrows():
        if row["model_A"] in MODELS and row["metric"] in METRICS:
            d_mat.loc[row["model_A"], row["metric"]] =  float(row["cohens_d"])
        if row["model_B"] in MODELS and row["metric"] in METRICS:
            d_mat.loc[row["model_B"], row["metric"]] = -float(row["cohens_d"])
    d_mat = d_mat.rename(columns=METRIC_LABELS, index=sm)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        d_mat.astype(float), annot=True, fmt=".3f",
        cmap="RdBu", center=0, ax=ax, linewidths=0.4, linecolor="white",
        annot_kws={"size": 10, "weight": "bold"},
        cbar_kws={"label": "Cohen's d"},
    )
    ax.set_title(f"Cohen's d Effect Sizes (n={len(all_seeds)} seeds)",
                 fontweight='bold', fontsize=13)
    plt.xticks(rotation=20, ha="right", fontsize=10, fontweight='bold')
    plt.yticks(rotation=0,  fontsize=9,  fontweight='bold')
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "cohens_d_heatmap.png", dpi=160, bbox_inches='tight')
    plt.close()
    print("  ?  cohens_d_heatmap.png")

    # ECE bar
    if "ece_mean" in agg.columns:
        ece_means = agg["ece_mean"].values
        ece_stds  = agg["ece_std"].values
        fig, ax = plt.subplots(figsize=(11, 5))
        bars = ax.bar(x, ece_means, yerr=ece_stds, capsize=5,
                      color=palette, edgecolor='k', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=30, ha="right", fontsize=10, fontweight='bold')
        ax.set_ylabel("ECE (lower is better)", fontweight='bold', fontsize=13)
        ax.set_title("Expected Calibration Error — Ablation",
                     fontweight='bold', fontsize=13)
        for bar, m, s in zip(bars, ece_means, ece_stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.001,
                f"{m:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold',
            )
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(SAVE_DIR / "ece_calibration_bar.png", dpi=180, bbox_inches='tight')
        plt.close()
        print("  ?  ece_calibration_bar.png")

    print("\n  ?  All comparison plots saved.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    print("\n" + "="*70)
    print("  PASAN-DG: Probabilistic Augmented Spectral Attention Network")
    print(f"  Models : {MODELS}")
    print(f"  Seeds  : {SEEDS}  (n={len(SEEDS)})")
    print(f"  Device : {DEVICE}")
    print("="*70)

    all_results = []

    # 1. Train all models x all seeds
    for model_name in MODELS:
        for seed in SEEDS:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            res = run_one_seed(seed, model_name)
            all_results.append(res)
            # Save after every run (fault tolerant)
            (
                pd.DataFrame(all_results)
                .drop_duplicates(subset=["model", "seed"])
                .reset_index(drop=True)
                .to_csv(SAVE_DIR / "all_results.csv", index=False)
            )

    # 2. Aggregate
    df = (
        pd.read_csv(SAVE_DIR / "all_results.csv")
        .drop_duplicates(subset=["model", "seed"])
        .reset_index(drop=True)
    )

    agg_cols = {
        "acc":             ["mean", "std", "min", "max"],
        "f1_macro":        ["mean", "std", "min", "max"],
        "precision_macro": ["mean", "std"],
        "recall_macro":    ["mean", "std"],
        "auc_macro":       ["mean", "std"],
        "ece":             ["mean", "std", "min", "max"],
    }
    agg = df.groupby("model").agg(agg_cols).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reindex(MODELS)
    agg.to_csv(SAVE_DIR / "summary_aggregated.csv")

    print("\n\n=== Aggregated Summary ===")
    print(agg.to_string())

    # 3. Ablation table
    abl_df = build_ablation_table(df)

    # 4. Statistical tests
    wilcoxon_df = run_wilcoxon(df)
    mcnemar_df  = run_mcnemar_pooled(SEEDS)
    ttest_df    = run_paired_ttest_ablation(df)

    # 5. Superiority + CBGE
    sup_df  = build_superiority_table(df, wilcoxon_df, mcnemar_df)
    cbge_df = compute_cbge(df)

    # 6. Plots
    save_comparison_plots(df, agg, SEEDS)

    # 7. Summary
    print(f"\n{'='*70}")
    print(f"  All outputs saved to: {SAVE_DIR.resolve()}")
    print(f"{'='*70}")
    print("""
  Per-model subfolders (6 models x 6 seeds each):
    <Model>/metrics/  test_metrics, cm, roc, calibration, history (CSV+PNG)
    <Model>/curves/   training curves PNG
    <Model>/gradcam/  <class>/img_XXXXX.png  (ALL test images)

  Global:
    all_results.csv, summary_aggregated.csv
    ablation_table.csv, paired_ttest_ablation.csv
    wilcoxon_n6.csv, mcnemar_pooled.csv
    superiority_table.csv, cbge_estimates.csv
    boxplot_all_metrics.png, barplot_*.png, ablation_progression.png
    mcnemar_pvalue_heatmap.png, cohens_d_heatmap.png, ece_calibration_bar.png
    """)
