# ANUFA-PASAN_DG

PASAN-DG
Probabilistic Augmented Spectral Attention Network for Domain Generalization in Medical Imaging
1. Overview
This repository contains the complete experimental implementation of PASAN-DG, a domain-generalization framework for medical image classification.

The experimental design evaluates PASAN-DG through a strictly ordered ablation study:

Baseline — ConvNeXt-Tiny with standard Empirical Risk Minimization (ERM)

Plus_Spectral — Baseline + SpectralMessageBlock

Plus_KL — Plus_Spectral + VariationalBottleneck with KL regularization

Plus_MixStyle — Plus_KL + MixStyleLayer

Plus_Attention — Plus_MixStyle + DomainAttentionGate

PASAN_DG — Full model + Knowledge Distillation from ConvNeXt-Small

The experiments use multiple random seeds to assess the stability of the results.

Experimental seed set
42, 43, 44, 45, 46, 47

Thus, each ablation model is evaluated over 6 independent seeds.

The complete experiment consists of:

6 models × 6 seeds = 36 training/evaluation runs

2. Dataset Protocol
2.1 Training Dataset
The training data are expected in an ImageFolder-compatible directory structure:

Training/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/

The training dataset is used for:

model training

validation split generation

model selection through validation loss

The training dataset is split independently for every random seed using:

80% training
20% validation

The split is stratified by class.

The random state is equal to the corresponding experimental seed.

2.2 Independent Test Dataset
The test dataset must be maintained separately from the training dataset.

Expected structure:

Test/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/

The test dataset is not used for model selection or early stopping.

Testing is performed only after the best validation-loss checkpoint has been selected.

3. Kaggle Training Protocol
3.1 Purpose
The Kaggle environment is used for model training and validation.

The training pipeline should be executed using the training dataset only.

The test dataset must not influence:

hyperparameter selection

model architecture selection

early stopping

checkpoint selection

augmentation selection

training decisions

3.2 Environment
The main training script is:

PASAN_DG_main.py

The expected execution environment is a GPU-enabled PyTorch environment.

The code currently specifies:

DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

Important Kaggle requirement
Kaggle GPU device numbering must be checked before execution.

If Kaggle exposes only:

cuda:0

then the device configuration should be changed to:

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Do not assume that cuda:2 exists on Kaggle.

4. Kaggle Dataset Paths
The original script contains cluster-specific paths:

TRAIN_PATH = "/nfsshare/users/raghavan/Brainz/Brain tumor dataset/Training/"
TEST_PATH  = "/nfsshare/users/raghavan/Brainz/Brain tumor dataset/Test/"

These paths are not portable to Kaggle.

They must be replaced by the Kaggle dataset mount paths.

For example:

TRAIN_PATH = "/kaggle/input/<dataset-name>/Training/"
TEST_PATH  = "/kaggle/input/<dataset-name>/Test/"

The exact path must match the uploaded Kaggle dataset.

No hard-coded /nfsshare/... path should remain in the Kaggle version.

5. Image Preprocessing
All images are converted to RGB.

Training augmentation
Training images use:

RandomResizedCrop: 224 × 224
scale: 0.80–1.00
RandomHorizontalFlip
RandomVerticalFlip: p = 0.10
RandomRotation: ±20°
ColorJitter:
    brightness = 0.2
    contrast   = 0.2
    saturation = 0.1
RandomGrayscale: p = 0.05

Images are then converted to floating-point tensors and normalized using ImageNet statistics:

Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]

Validation/Test preprocessing
Validation and test images use:

Resize: 224 × 224
RGB conversion
Tensor conversion
ImageNet normalization

No stochastic augmentation is applied during validation or testing.

6. Base Architecture
The principal backbone is:

ConvNeXt-Tiny

implemented through timm:

convnext_tiny.in12k_ft_in1k

The backbone uses pretrained weights.

The final classification problem contains:

4 classes

7. Ablation Architecture
The ablation study is cumulative.

7.1 Baseline
ConvNeXt-Tiny
+ classification head

Training objective:

Cross-Entropy Loss
+ label smoothing

No PASAN-DG-specific modules are added.

7.2 Plus_Spectral
Adds:

SpectralMessageBlock

The block performs:

2-D FFT

learned frequency reweighting

inverse FFT

squeeze-and-excitation gating

GroupNorm

residual connection

Conceptually:

Feature
  ↓
FFT
  ↓
Frequency Reweighting
  ↓
Inverse FFT
  ↓
SE Gate
  ↓
GroupNorm
  ↓
Residual Addition

7.3 Plus_KL
Adds the:

VariationalBottleneck

The feature representation is transformed into:

μ
log σ²
z

using the reparameterization mechanism during training.

The objective additionally includes KL divergence:

KL = -0.5 × mean(1 + logvar - μ² - exp(logvar))

The KL coefficient is gradually increased:

β_KL = 0 → BETA_KL_MAX

with:

BETA_KL_MAX = 1e-3

over the first 20 epochs.

7.4 Plus_MixStyle
Adds:

MixStyleLayer

with:

p = 0.5
alpha = 0.1

During training, feature statistics can be mixed between samples.

This is intended to perturb style-related feature statistics and improve robustness to domain-specific appearance variation.

MixStyle is disabled during evaluation.

7.5 Plus_Attention
Adds:

DomainAttentionGate

after MixStyle.

The gate performs channel-wise recalibration using global feature statistics.

The default reduction factor is:

reduction = 8

7.6 PASAN_DG
The final PASAN-DG model combines:

ConvNeXt-Tiny
      ↓
SpectralMessageBlock
      ↓
MixStyleLayer
      ↓
DomainAttentionGate
      ↓
VariationalBottleneck
      ↓
Classifier

and additionally uses:

ConvNeXt-Small

as a frozen teacher during training.

The teacher architecture is:

convnext_small.in12k_ft_in1k

8. Knowledge Distillation
Knowledge distillation is used only in:

PASAN_DG

The teacher is frozen:

for p in teacher_model.parameters():
    p.requires_grad = False

The distillation temperature is:

T = 4.0

The distillation weight is:

0.4

The distillation loss is:

KL(student || teacher)

with temperature scaling:

KL × T²

The final PASAN-DG training loss combines classification and distillation:

Loss =
    (1 - 0.4) × classification_loss
    + 0.4 × distillation_loss

When applicable, the KL regularization term is also included.

9. MixUp
MixUp is enabled only for:

PASAN_DG

with:

MIXUP_ALPHA = 0.4

A mixed input is generated as:

x_mix = λx + (1 − λ)x_perm

The classification loss is correspondingly mixed between the original and permuted labels.

The teacher receives the same mixed input during knowledge distillation.

10. Optimization
All models use:

Optimizer: AdamW
Learning rate: 1e-4
Weight decay: 1e-2

Classification loss:

CrossEntropyLoss
label_smoothing = 0.10

Gradient clipping:

max_norm = 1.0

Learning-rate scheduler:

CosineAnnealingWarmRestarts
T_0 = 20
T_mult = 1
eta_min = 1e-6

Maximum training duration:

60 epochs

Early stopping:

patience = 12 epochs

The selected checkpoint is the model state with the lowest validation loss.

11. Batch Size
The current configuration uses:

Batch size = 16

for all six models.

If GPU memory is insufficient, the batch size may be changed only before beginning a formally reported experiment.

Any change must be documented because it changes the experimental protocol.

12. Reproducibility
Every experiment explicitly sets:

Python random seed
NumPy seed
PyTorch CPU seed
PyTorch CUDA seed

The code also enables deterministic cuDNN behavior:

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

The six reported seeds are:

42
43
44
45
46
47

Results must therefore be reported as:

mean ± standard deviation

over the six seeds.

13. Mendeley Testing Protocol
13.1 Purpose
The Mendeley dataset is treated as an independent external test dataset.

It must not be used during:

training

validation

early stopping

hyperparameter tuning

architecture selection

teacher selection

threshold selection

The purpose of the Mendeley evaluation is to assess external/generalization performance.

13.2 Required Dataset Structure
The external Mendeley test set should be organized as:

Mendeley_Test/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/

The class names must correspond to the four classes used during training.

The class-to-index mapping must be checked carefully.

For ImageFolder, alphabetical ordering is normally used.

Therefore the final experiment must verify:

classes = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

Do not silently rename or reorder classes between training and testing.

14. External Testing Rule
The Mendeley dataset must be loaded with the test/validation transform only.

Use:

Resize 224 × 224
RGB
ToTensor
ImageNet normalization

Do not use:

RandomResizedCrop
RandomRotation
ColorJitter
RandomHorizontalFlip
RandomVerticalFlip
RandomGrayscale
MixUp
MixStyle

during external testing.

15. Model Checkpoint Selection
For each seed, the checkpoint selected using the internal validation set must be evaluated on the independent Mendeley test set.

The Mendeley test results must not be used to select the best seed.

The recommended reporting procedure is:

Seed 42 checkpoint → Mendeley
Seed 43 checkpoint → Mendeley
Seed 44 checkpoint → Mendeley
Seed 45 checkpoint → Mendeley
Seed 46 checkpoint → Mendeley
Seed 47 checkpoint → Mendeley

Results should then be summarized over the six independently trained models.

16. Testing Metrics
The following metrics are required:

Accuracy
Macro Precision
Macro Recall
Macro F1
Macro AUC
ECE

Additionally, the following should be retained:

Per-class Precision
Per-class Recall
Per-class F1
Per-class AUC
Per-class Specificity
Confusion Matrix
ROC curves
Calibration curves

17. Expected Output Structure
The principal training output directory is:

PASAN_DG_RESULTS_stnv/

Expected structure:

PASAN_DG_RESULTS_stnv/
│
├── Baseline/
│   ├── metrics/
│   ├── curves/
│   ├── gradcam/
│   └── model_note.txt
│
├── Plus_Spectral/
│   ├── metrics/
│   ├── curves/
│   ├── gradcam/
│   └── model_note.txt
│
├── Plus_KL/
│   ├── metrics/
│   ├── curves/
│   ├── gradcam/
│   └── model_note.txt
│
├── Plus_MixStyle/
│   ├── metrics/
│   ├── curves/
│   ├── gradcam/
│   └── model_note.txt
│
├── Plus_Attention/
│   ├── metrics/
│   ├── curves/
│   ├── gradcam/
│   └── model_note.txt
│
├── PASAN_DG/
│   ├── metrics/
│   ├── curves/
│   ├── gradcam/
│   └── model_note.txt
│
├── all_results.csv
├── summary_aggregated.csv
├── ablation_table.csv
├── paired_ttest_ablation.csv
├── wilcoxon_n6.csv
├── mcnemar_pooled.csv
├── superiority_table.csv
├── cbge_estimates.csv
│
├── boxplot_all_metrics.png
├── barplot_acc.png
├── barplot_f1_macro.png
├── barplot_precision_macro.png
├── barplot_recall_macro.png
├── barplot_auc_macro.png
├── barplot_ece.png
├── ablation_progression.png
├── mcnemar_pvalue_heatmap.png
├── cohens_d_heatmap.png
└── ece_calibration_bar.png

18. Per-Seed Outputs
For each model and seed, the following are generated:

test_metrics_seedXX.csv
cm_seedXX.csv
cm_seedXX.png
roc_data_seedXX.csv
roc_seedXX.png
calibration_seedXX.png
history_seedXX.csv
curves_seedXX.png

Grad-CAM outputs are organized by class:

gradcam/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/

Each test image receives a Grad-CAM visualization.

19. Grad-CAM
Grad-CAM is generated for all test images.

The target layers are:

Baseline
model.stages[-1].blocks[-1].conv_dw

FeatureListNet-based models
bb.stages_3.blocks[-1].conv_dw

The Grad-CAM output contains:

Original image

Grad-CAM heatmap

Original image with Grad-CAM overlay

The title records:

True class
Predicted class
Correct/Wrong status
Prediction confidence

20. Statistical Analysis
The ablation study uses multiple statistical analyses.

20.1 Paired t-test
Adjacent ablation stages are compared:

Baseline → Plus_Spectral
Plus_Spectral → Plus_KL
Plus_KL → Plus_MixStyle
Plus_MixStyle → Plus_Attention
Plus_Attention → PASAN_DG

The comparison is performed independently for each metric.

20.2 Wilcoxon Signed-Rank Test
All model pairs are compared using the Wilcoxon signed-rank test across the six seeds.

The output is:

wilcoxon_n6.csv

The significance threshold is:

α = 0.05

20.3 Cohen's d
Effect size is calculated using Cohen's d.

Interpretation:

|d| < 0.20       negligible
0.20–0.49       small
0.50–0.79       medium
≥ 0.80          large

20.4 McNemar Test
McNemar's test is applied to paired prediction outcomes.

The current implementation pools discordant prediction counts across the six seeds.

The output is:

mcnemar_pooled.csv

21. Important Statistical Interpretation
Statistical significance must not be interpreted as proof that PASAN-DG is universally superior.

The following should be reported separately:

Numerical improvement
Effect size
Statistical significance
External-dataset performance

A result should not be described as "significantly better" unless the corresponding statistical test supports that claim.

22. Complexity Analysis
Two separate scripts are provided.

PASAN_DG_complexity.py
PASAN_DG_flops.py

These scripts are intended for architectural complexity reporting and should be run separately from model training.

23. Parameter Analysis
PASAN_DG_complexity.py reports:

Model
Params (M)
bb (M)
Head (M)
Teacher (M, train-only)
CPU latency (ms)
Mem b1 (MB)
Mem b16 (MB)

For PASAN-DG, the frozen ConvNeXt-Small teacher is reported separately.

It is not included in deployed inference parameters, because the teacher is used only during training.

24. FLOPs Analysis
PASAN_DG_flops.py reports:

Model
FLOPs (G)
MACs (G)

The measurement uses a:

single 224 × 224 RGB image
batch size = 1

The teacher is explicitly excluded from PASAN-DG inference FLOPs.

The resulting file is:

PASAN_DG_COMPLEXITY_RESULTS/flops_table.csv

25. Complexity Output
Expected structure:

PASAN_DG_COMPLEXITY_RESULTS/
├── complexity_table.csv
└── flops_table.csv

26. Inference Definition
For complexity reporting, the deployed PASAN-DG inference network consists of:

ConvNeXt-Tiny backbone
+
SpectralMessageBlock
+
MixStyleLayer
+
DomainAttentionGate
+
VariationalBottleneck
+
Classifier

The ConvNeXt-Small teacher is excluded because it is a training-only component.

27. Strict Train/Test Separation
The following rule must be maintained throughout the study:

TRAINING DATA
     │
     ├── 80% Training
     │
     └── 20% Validation
              │
              └── Model selection / early stopping

Mendeley TEST DATA
     │
     └── Independent final evaluation

The external test set must never be used to:

tune hyperparameters

select an architecture

choose a seed

choose an epoch

select a threshold

modify augmentation

determine the final model configuration

28. Recommended Execution Order
Step 1 — Prepare Kaggle dataset
Upload/mount the training dataset in Kaggle.

Verify:

Training/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/

Step 2 — Verify environment
Confirm:

Python
PyTorch
Torchvision
timm
NumPy
Pandas
SciPy
scikit-learn
Matplotlib
Seaborn

and verify that the selected CUDA device exists.

Step 3 — Update paths
Modify:

TRAIN_PATH
TEST_PATH
DEVICE

for the Kaggle environment.

Step 4 — Run training
Execute:

python PASAN_DG_main.py

The script trains:

6 models × 6 seeds

and saves results after every completed run.

This fault-tolerant behavior allows completed experiments to remain available if a later run fails.

Step 5 — Verify training outputs
Before external testing, verify:

all_results.csv
summary_aggregated.csv
ablation_table.csv

and confirm that all expected:

6 models × 6 seeds

are present.

Step 6 — Freeze the experimental configuration
After the Kaggle training results are finalized:

Do not modify:
- architecture
- learning rate
- augmentation
- seed list
- loss weights
- test preprocessing
- model selection criterion

The external test should then be performed using the frozen protocol.

Step 7 — Run Mendeley external testing
Evaluate the six seed-specific checkpoints for each model on the Mendeley test dataset.

Save:

per-seed predictions
per-class metrics
confusion matrices
ROC curves
calibration results
Grad-CAM visualizations

Step 8 — Complexity analysis
Run:

python PASAN_DG_complexity.py

followed by:

python PASAN_DG_flops.py

Save the resulting tables separately from the training results.

29. Reproducibility Checklist
Before claiming that the experiment is reproducible, verify all of the following:

 Same dataset version

 Same four class names

 Same class ordering

 Same six random seeds

 Same train/validation split procedure

 Same image size

 Same normalization

 Same augmentations

 Same optimizer

 Same learning rate

 Same weight decay

 Same scheduler

 Same label smoothing

 Same KL coefficient

 Same MixStyle parameters

 Same MixUp parameters

 Same distillation temperature

 Same distillation weight

 Same early-stopping criterion

 Same patience

 Same backbone versions

 Same pretrained weights

 Same PyTorch/timm environment where possible

 External Mendeley data excluded from training decisions

 Test results computed only after checkpoint selection

30. Reporting Standard
For each model, report:

Accuracy       = mean ± SD
Macro F1       = mean ± SD
Macro AUC      = mean ± SD
Macro Precision= mean ± SD
Macro Recall   = mean ± SD
ECE            = mean ± SD

For the primary comparison between PASAN-DG and Baseline, additionally report:

Absolute difference
Cohen's d
Wilcoxon p-value
McNemar p-value
External Mendeley performance

31. Recommended Main Results Table
The final manuscript should contain a table with the following structure:

Model	Accuracy	Macro F1	Macro Precision	Macro Recall	Macro AUC	ECE
Baseline	mean ± SD	mean ± SD	mean ± SD	mean ± SD	mean ± SD	mean ± SD
Plus_Spectral	mean ± SD	mean ± SD	mean ± SD	mean ± SD	mean ± SD	mean ± SD
Plus_KL	mean ± SD	mean ± SD	mean ± SD	mean ± SD	mean ± SD	mean ± SD
Plus_MixStyle	mean ± SD	mean ± SD	mean ± SD	mean ± SD	mean ± SD	mean ± SD
Plus_Attention	mean ± SD	mean ± SD	mean ± SD	mean ± SD	mean ± SD	mean ± SD
PASAN-DG	mean ± SD	mean ± SD	mean ± SD	mean ± SD	mean ± SD	mean ± SD

32. Recommended External Validation Table
For the independent Mendeley dataset:

Model	Accuracy	Macro F1	Macro Precision	Macro Recall	Macro AUC	ECE
Baseline	—	—	—	—	—	—
Plus_Spectral	—	—	—	—	—	—
Plus_KL	—	—	—	—	—	—
Plus_MixStyle	—	—	—	—	—	—
Plus_Attention	—	—	—	—	—	—
PASAN-DG	—	—	—	—	—	—

Values must be populated only after the independent Mendeley evaluation has been completed.

33. Recommended Complexity Table
Model	Params (M)	Backbone (M)	Head/Modules (M)	Teacher (M)	CPU Latency (ms)	Mem B1 (MB)	Mem B16 (MB)	FLOPs (G)
Baseline	—	—	—	—	—	—	—	—
Plus_Spectral	—	—	—	—	—	—	—	—
Plus_KL	—	—	—	—	—	—	—	—
Plus_MixStyle	—	—	—	—	—	—	—	—
Plus_Attention	—	—	—	—	—	—	—	—
PASAN-DG	—	—	—	—	—	—	—	—

34. Important Implementation Notes
34.1 Do not claim monotonic improvement before observing the results
The architecture is designed as a cumulative ablation progression.

However, the statement:

"strictly monotonically improving"

is a hypothesis/design expectation, not a result that can be guaranteed by the code.

The final manuscript must report the actual measured performance.

If an intermediate component decreases a metric, that decrease must be reported honestly.

34.2 ECE interpretation
ECE is a lower-is-better metric.

Therefore:

Lower ECE = better calibration

while:

Higher Accuracy/F1/AUC/Precision/Recall = better

34.3 Complexity interpretation
Training complexity and deployment complexity are different.

The PASAN-DG teacher:

ConvNeXt-Small

is required during training but is not part of the deployed inference network.

Therefore it should not be counted as deployment parameters or deployment FLOPs.

It should nevertheless be reported separately as:

Teacher (M, train-only)

35. Directory Separation
The recommended project organization is:

PASAN-DG/
│
├── PASAN_DG_main.py
├── PASAN_DG_complexity.py
├── PASAN_DG_flops.py
├── README.md
│
├── training/
│   └── Kaggle-related configuration
│
├── external_testing/
│   └── Mendeley evaluation scripts
│
├── PASAN_DG_RESULTS_stnv/
│   └── Training/ablation results
│
└── PASAN_DG_COMPLEXITY_RESULTS/
    └── Complexity and FLOPs results

The exact folder names may be changed, but training results and complexity results should remain logically separated.

36. Final Experimental Workflow
The complete study follows this sequence:

                 ┌──────────────────────┐
                 │  Training Dataset    │
                 └──────────┬───────────┘
                            │
                    Stratified 80/20
                            │
             ┌──────────────┴──────────────┐
             │                             │
          Training                     Validation
             │                             │
             └──────────────┬──────────────┘
                            │
                    Model Selection
                            │
              ┌─────────────┴─────────────┐
              │                           │
       6 Ablation Models             6 Seeds
              │                           │
              └─────────────┬─────────────┘
                            │
                     Frozen Checkpoints
                            │
                            ▼
                  Independent Mendeley
                       Test Set
                            │
              ┌─────────────┼─────────────┐
              │             │             │
           Metrics       ROC/ECE       Grad-CAM
              │             │             │
              └─────────────┴─────────────┘
                            │
                            ▼
                 Statistical Analysis
                            │
                            ▼
                 Complexity + FLOPs
                            │
                            ▼
                 Final Manuscript Tables

37. Publication Integrity Statement
The experimental protocol is designed so that the independent Mendeley dataset is used exclusively for external evaluation.

All architecture and training decisions must be finalized before inspecting Mendeley test performance.

No result should be selectively removed because it is unfavorable.

All six seeds should be retained unless a run fails for a documented technical reason.

Any excluded run must be explicitly documented.

The final reported results should distinguish between:

Internal validation performance
Internal test performance, if applicable
External Mendeley performance
Statistical significance
Effect size
Computational complexity

This separation is essential for a defensible domain-generalization study.

38. Software and Reproducibility Information
The experiment depends on:

Python
PyTorch
Torchvision
timm
NumPy
Pandas
SciPy
scikit-learn
Matplotlib
Seaborn

The exact package versions used for the final reported experiment should be recorded and included with the submission.

Recommended command:

pip freeze > requirements-freeze.txt

The CUDA version, GPU model, PyTorch version, torchvision version, and timm version should also be recorded.

39. Citation and Research Use
When reporting this implementation in a manuscript, describe the experiment as a multi-seed cumulative ablation with independent external validation.

Do not report only the best seed.

The primary reported value should be:

Mean ± standard deviation across six seeds

and the independent Mendeley evaluation should be reported separately.

40. Final Checklist Before Submission
Training
 Six models trained

 Six seeds completed for each model

 36 runs accounted for

 Best checkpoint selected using validation loss only

 Training results saved

 Training curves saved

Internal Evaluation
 Accuracy

 Macro F1

 Macro Precision

 Macro Recall

 Macro AUC

 ECE

 Confusion matrices

 ROC curves

 Calibration curves

 Grad-CAM

Ablation
 Ablation table

 Paired t-tests

 Wilcoxon tests

 Cohen's d

 McNemar analysis

 PASAN-DG vs Baseline superiority analysis

External Mendeley Evaluation
 Independent dataset

 No training-data leakage

 No hyperparameter tuning on Mendeley

 Same preprocessing

 Six seed checkpoints evaluated

 Per-class results retained

 Aggregate results reported as mean ± SD

Complexity
 Parameter count

 Backbone parameter count

 Additional module parameter count

 Teacher parameter count separately

 CPU latency

 Batch-1 memory

 Batch-16 memory

 FLOPs

 MACs

Reproducibility
 Dataset version recorded

 Package versions recorded

 CUDA version recorded

 GPU recorded

 Random seeds recorded

 Final code archived

 Final configuration archived

 No undocumented experimental changes

End of README
PASAN-DG — Probabilistic Augmented Spectral Attention Network for Domain Generalization in Medical Imaging

Experimental protocol:

Kaggle
  ↓
Training + Internal Validation
  ↓
6 Models × 6 Seeds
  ↓
Frozen Checkpoints
  ↓
Independent Mendeley Testing
  ↓
Statistical Analysis
  ↓
Complexity/FLOPs
  ↓
Publication Results
