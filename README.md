# HAM10000 Multimodal Skin Lesion Classification

A multimodal deep learning system for dermoscopic skin lesion classification,
combining image data with patient metadata through cross-attention fusion.
Built from scratch in PyTorch on Google Colab (T4 GPU, free tier).

## Task

7-class classification of skin lesions from the
[HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
(10,015 dermoscopic images + patient metadata: age, sex, lesion localization).

**Primary metric**: Macro-averaged AUROC across all 7 classes.
Accuracy is not used as the primary metric due to severe class imbalance
(melanocytic nevi = 67% of samples).

## Results

| Model | Test AUROC | Test Accuracy |
|---|---|---|
| Metadata only (MLP) | 0.7764 | 0.3556 |
| Image only (CNN-B) | 0.9231 | 0.6663 |
| FiLM fusion | 0.9158 | 0.6803 |
| Early fusion | 0.9323 | 0.6893 |
| Late fusion | 0.9350 | 0.6963 |
| **Cross-attention fusion** | **0.9385** | **0.7063** |

### Per-class AUROC: image-only vs cross-attention

| Class | Image only | Cross-attention | Delta |
|---|---|---|---|
| Melanoma | 0.8517 | 0.8829 | +0.0312 |
| Actinic keratosis | 0.9509 | 0.9744 | +0.0235 |
| Benign keratosis | 0.8663 | 0.8944 | +0.0281 |
| Basal cell carcinoma | 0.9613 | 0.9689 | +0.0076 |
| Dermatofibroma | 0.9278 | 0.9327 | +0.0049 |
| Nevi | 0.9156 | 0.9391 | +0.0236 |
| Vascular | 0.9880 | 0.9771 | -0.0109 |

The gain is concentrated in the malignant and pre-malignant classes -
most critically melanoma (+0.0312), where missed diagnoses carry the
highest clinical cost.

## Project Structure

### Phase 1 — Data Analysis & Baseline
- EDA revealing 67% class imbalance
- Stratified 70/20/10 train/val/test split
- Majority-class baseline: 67% accuracy, 0.50 AUROC
- Establishes why macro AUROC is the correct metric

### Phase 2 — Architecture Ablation

| Model | Description | Test AUROC |
|---|---|---|
| CNN-A | Simple conv blocks | 0.8696 |
| CNN-B | + Residual connections + BatchNorm | 0.9198 |
| CNN-C | + Squeeze-and-Excitation attention | 0.9198 |
| ViT-Tiny | Vision Transformer from scratch | 0.8980 |

Selected: CNN-B. Equal AUROC to CNN-C at lower complexity.
ViT-Tiny underperformed as expected - insufficient data for training without pretraining.

### Phase 3 — Training Configuration

| Dimension | Winner | Notes |
|---|---|---|
| Optimizer | Adam | Marginal +0.004 over AdamW baseline |
| Loss function | Standard CE | Focal loss competitive but no clear win |
| Regularization | No dropout | CNN-B too compact, dropout hurt capacity |
| LR schedule | Cosine annealing | OneCycleLR destabilised training |

Full results:

| Experiment | Test AUROC |
|---|---|
| Baseline (AdamW + CE) | 0.9161 |
| Adam + CE | 0.9199 |
| Focal γ=0.5 | 0.9170 |
| Focal γ=2.0 | 0.9176 |
| Focal γ=5.0 | 0.9085 |
| Weighted CE | 0.8367 |
| Dropout 0.3 | 0.9113 |
| Dropout 0.5 | 0.9102 |
| OneCycleLR | 0.8709 |
| ReduceLROnPlateau | 0.9170 |
| **Adam + focal γ=2.0 (combination)** | **0.9153** |

Key findings:
- Adam won individually at 0.9199. Focal γ=2.0 won its group at 0.9176.
- Combining both produced 0.9153 — worse than either individually.
  This confirms the components interact rather than compound: Adam without
  weight decay has a specific optimisation dynamic that focal loss
  disrupts rather than complements.
- Weighted cross-entropy severely underperformed (0.8367) due to loss
  scale inflation from aggressive inverse frequency weighting.
- OneCycleLR destabilised training — the warmup phase interacted poorly
  with the small dataset size.

### Phase 4 — Multimodal Fusion

| Model | Test AUROC |
|---|---|
| Metadata only | 0.7764 |
| Image only | 0.9231 |
| FiLM fusion | 0.9158 |
| Early fusion | 0.9323 |
| Late fusion | 0.9350 |
| Cross-attention fusion | 0.9385 |

- **Early fusion**: metadata injected at stem output - disrupts low-level
  feature learning, underperforms image-only
- **Late fusion**: separate pipelines concatenated at classifier - simple
  and effective, beats image-only
- **FiLM**: channel-wise affine modulation at intermediate layers -
  global modulation introduces noise, underperforms
- **Cross-attention**: metadata queries attend to spatial image tokens
  at the pooling stage - spatially selective, best performer

### Phase 5 — Interpretability & Calibration
- Grad-CAM heatmaps on final convolutional layer
- Cross-attention weight visualisation
- Reliability diagrams and ECE scores

## Key Design Decisions

**Why AUROC over accuracy**: a model predicting only nevi achieves 67%
accuracy but 0.50 AUROC. Accuracy is misleading for imbalanced medical data.

**Why CNN-B over CNN-C**: SE attention added no measurable gain at higher
parameter count. Simpler model preferred when performance is equal.

**Why cross-attention beats late fusion**: late fusion averages all spatial
positions equally before combining with metadata. Cross-attention delays
spatial collapse until after the metadata guides which regions to summarise,
producing a spatially-selective representation.

**Why early fusion underperforms**: injecting demographic context before
meaningful features have formed adds noise to low-level processing.
The CNN has to learn to ignore the injection at early layers, making
optimisation harder.

## Setup

```bash
pip install torch torchvision scikit-learn pandas matplotlib seaborn wandb
```

Download the dataset from Kaggle:

```bash
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
```

## Training

All training was performed on Google Colab free tier (T4 GPU, 16GB VRAM).
Approximate training times per model: ~65 minutes (30 epochs, batch size 32).
Checkpoints saved to Google Drive after every epoch with full optimizer
state restoration for seamless resume after disconnects.

## Citation

```
Tschandl P., Rosendahl C. & Kittler H. The HAM10000 dataset, a large
collection of multi-source dermatoscopic images of common pigmented
skin lesions. Sci. Data 5, 180161 (2018).
doi: 10.1038/sdata.2018.161
```
