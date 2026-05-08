# Technical Specification
## Brain Tumor Segmentation on MRI using 3D U-Net

**Project:** Automatic segmentation of brain tumor regions on multimodal MRI scans
**Architecture:** 3D U-Net (MONAI implementation)
**Dataset:** BraTS 2023 (Adult Glioma — BraTS-GLI 2023)
**Target audience:** Implementation by Claude Code in IDE terminal

> **Version note:** This specification was updated from BraTS 2021 to BraTS 2023 after dataset verification. The unique label values in the segmentation masks are `{0, 1, 2, 3}` (BraTS 2023 convention), **not** `{0, 1, 2, 4}` (BraTS 2021 convention). All label conversion logic, file naming patterns, and folder scanning has been updated accordingly.

---

## 1. Project Overview

The goal of this project is to implement a deep learning system that automatically segments brain tumor sub-regions from multimodal MRI scans. The system takes 4 MRI modalities (T1, T1ce, T2, FLAIR) as input and produces 3 binary segmentation masks corresponding to the standard BraTS tumor sub-regions:

- **WT (Whole Tumor)** — entire tumor including edema
- **TC (Tumor Core)** — necrotic core + enhancing tumor
- **ET (Enhancing Tumor)** — only the contrast-enhancing region

The model is trained on the BraTS 2023 dataset and evaluated using Dice Score, IoU, and 95% Hausdorff Distance metrics.

---

## 2. Technology Stack

### 2.1 Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.10+ | Main programming language |
| PyTorch | 2.1+ | Deep learning framework (backend for MONAI) |
| MONAI | 1.3+ | Medical imaging framework — model, transforms, metrics, data loaders |
| nibabel | 5.1+ | Reading/writing NIfTI medical image files |
| SimpleITK | 2.3+ | Additional medical image I/O and processing |
| NumPy | 1.24+ | Numerical operations |
| SciPy | 1.11+ | Scientific computing (post-processing, connected components) |

### 2.2 Training & Evaluation

| Library | Version | Purpose |
|---------|---------|---------|
| scikit-learn | 1.3+ | Train/val/test splitting, additional metrics |
| tqdm | 4.66+ | Progress bars during training |
| tensorboard | 2.15+ | Training metrics visualization |
| matplotlib | 3.8+ | Plotting and visualization of segmentation results |
| pandas | 2.1+ | Storing evaluation results in CSV |

### 2.3 Optional (recommended)

| Library | Purpose |
|---------|---------|
| wandb | Alternative to TensorBoard for experiment tracking |
| einops | Cleaner tensor reshape operations |

### 2.4 Installation Command

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "monai[all]==1.3.0"
pip install nibabel SimpleITK numpy scipy scikit-learn tqdm tensorboard matplotlib pandas
```

**Note:** Install the CUDA version of PyTorch matching the system's CUDA toolkit. The `monai[all]` extra installs all optional MONAI dependencies.

---

## 3. Dataset: BraTS 2023 (Adult Glioma)

### 3.1 Description

BraTS 2023 (Brain Tumor Segmentation Challenge 2023) is the current standard benchmark dataset for brain tumor segmentation research. The 2023 edition is organized as a cluster of sub-challenges; this project uses the **Adult Glioma (BraTS-GLI)** subset, which is the direct successor to BraTS 2021.

- **Number of cases:** ~1,251 training cases (with labels)
- **Modalities per case:** 4 (T1, T1ce, T2, FLAIR)
- **Image format:** NIfTI (`.nii.gz`)
- **Image size:** 240 × 240 × 155 voxels
- **Voxel spacing:** 1 × 1 × 1 mm (isotropic, already resampled)
- **Preprocessing applied by dataset authors:** skull-stripping, co-registration to anatomical template, resampling to 1 mm isotropic

### 3.2 Label Convention (BraTS 2023) — IMPORTANT CHANGE FROM 2021

The segmentation mask contains the following voxel values:

| Value | Meaning |
|-------|---------|
| 0 | Background |
| 1 | Necrotic and non-enhancing tumor core (NCR/NET) |
| 2 | Peritumoral edema (ED) |
| 3 | GD-enhancing tumor (ET) |

**Critical difference from BraTS 2021:** In BraTS 2021 the enhancing tumor label was `4` (with `3` skipped). In **BraTS 2023 the enhancing tumor label is `3`**, and there is no gap in the label values. Any code copied from BraTS 2021 tutorials must be updated.

The conversion to the 3 standard sub-regions for BraTS 2023:

- **WT (Whole Tumor)** = labels {1, 2, 3}
- **TC (Tumor Core)** = labels {1, 3}
- **ET (Enhancing Tumor)** = label {3}

This conversion must be implemented in the dataset loader.

### 3.3 Expected Folder Structure

BraTS 2023 uses a different file naming convention than BraTS 2021. Each patient folder is named like `BraTS-GLI-00000-000` (where the trailing `-000` is the timepoint/session ID), and modality files use the suffixes `t1n` (T1 native), `t1c` (T1 contrast), `t2w` (T2 weighted), and `t2f` (T2 FLAIR):

```
data/
└── BraTS2023/
    ├── BraTS-GLI-00000-000/
    │   ├── BraTS-GLI-00000-000-t1n.nii.gz
    │   ├── BraTS-GLI-00000-000-t1c.nii.gz
    │   ├── BraTS-GLI-00000-000-t2w.nii.gz
    │   ├── BraTS-GLI-00000-000-t2f.nii.gz
    │   └── BraTS-GLI-00000-000-seg.nii.gz
    ├── BraTS-GLI-00002-000/
    │   ├── BraTS-GLI-00002-000-t1n.nii.gz
    │   ├── ...
    └── ...
```

**Modality suffix mapping (for `dataset.py`):**

| BraTS 2023 suffix | Logical name (in `config.MODALITIES`) | Description |
|-------------------|----------------------------------------|-------------|
| `t1n` | `t1` | T1-weighted native |
| `t1c` | `t1ce` | T1-weighted post-contrast |
| `t2w` | `t2` | T2-weighted |
| `t2f` | `flair` | T2 FLAIR |
| `seg` | (label) | Ground-truth segmentation mask |

### 3.4 Data Split

Use a fixed random seed (e.g., 42) and split the cases as follows:

- **Training set:** 80% (~1000 cases)
- **Validation set:** 10% (~125 cases)
- **Test set:** 10% (~125 cases)

The split must be done at the patient level, not at the slice level. Save the split as a JSON file (`splits.json`) for reproducibility.

---

## 4. Project Structure

```
brain_tumor_segmentation/
│
├── data/
│   └── BraTS2023/                    # Raw dataset (not committed to git)
│
├── outputs/
│   ├── checkpoints/                  # Saved model weights
│   ├── logs/                         # TensorBoard logs
│   ├── predictions/                  # Inference outputs (NIfTI files)
│   └── results/                      # CSV files with evaluation metrics
│
├── splits.json                       # Fixed train/val/test split
│
├── config.py                         # Central configuration file
├── dataset.py                        # Data loading and BraTS-specific logic
├── transforms.py                     # MONAI transforms pipelines
├── model.py                          # Model architecture definitions
├── train.py                          # Training loop
├── evaluate.py                       # Evaluation on test set
├── predict.py                        # Inference on new data
│
├── requirements.txt                  # Python dependencies
└── README.md                         # Setup and usage instructions
```

---

## 5. File-by-File Specification

### 5.1 `config.py`

Central configuration with all hyperparameters and paths. No magic numbers anywhere else in the project.

**Required parameters:**

```python
# Paths
DATA_ROOT = "data/BraTS2023"
OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = "outputs/checkpoints"
LOG_DIR = "outputs/logs"
SPLITS_FILE = "splits.json"

# Dataset
BRATS_VERSION = "2023"
MODALITIES = ["t1", "t1ce", "t2", "flair"]   # Logical names used internally
# Mapping from logical names to BraTS 2023 file suffixes
MODALITY_SUFFIXES = {
    "t1":    "t1n",
    "t1ce":  "t1c",
    "t2":    "t2w",
    "flair": "t2f",
}
LABEL_SUFFIX = "seg"
NUM_INPUT_CHANNELS = 4
NUM_OUTPUT_CHANNELS = 3  # WT, TC, ET

# BraTS 2023 label values in the raw segmentation mask
LABEL_BACKGROUND = 0
LABEL_NCR = 1   # Necrotic / non-enhancing tumor core
LABEL_ED  = 2   # Peritumoral edema
LABEL_ET  = 3   # GD-enhancing tumor (was 4 in BraTS 2021!)

# Data split
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

# Model
MODEL_NAME = "UNet"  # Options: "UNet", "AttentionUNet", "SegResNet"
CHANNELS = (32, 64, 128, 256, 512)
STRIDES = (2, 2, 2, 2)
NUM_RES_UNITS = 2

# Training
PATCH_SIZE = (128, 128, 128)
BATCH_SIZE = 2
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
OPTIMIZER = "AdamW"
SCHEDULER = "CosineAnnealingLR"

# Loss
LOSS_FUNCTION = "DiceCELoss"
INCLUDE_BACKGROUND = False
SIGMOID = True

# Validation
VAL_INTERVAL = 5  # Validate every N epochs
SLIDING_WINDOW_OVERLAP = 0.5

# Early stopping
EARLY_STOPPING_PATIENCE = 30

# Hardware
DEVICE = "cuda"  # or "cpu"
NUM_WORKERS = 4
PIN_MEMORY = True
```

### 5.2 `dataset.py`

Responsible for scanning the BraTS 2023 data directory, creating data dictionaries, splitting into train/val/test, and converting labels.

**Required functions:**

1. `scan_brats_directory(data_root)` — scans the BraTS 2023 folder. For each patient subfolder (e.g., `BraTS-GLI-00000-000`), constructs file paths using the BraTS 2023 naming convention and the `MODALITY_SUFFIXES` mapping from `config.py`. Returns a list of dictionaries:
   ```python
   [
       {
           "image": [
               "data/BraTS2023/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1n.nii.gz",
               "data/BraTS2023/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1c.nii.gz",
               "data/BraTS2023/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2w.nii.gz",
               "data/BraTS2023/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2f.nii.gz",
           ],
           "label": "data/BraTS2023/BraTS-GLI-00000-000/BraTS-GLI-00000-000-seg.nii.gz",
           "patient_id": "BraTS-GLI-00000-000"
       },
       ...
   ]
   ```
   The function must verify that all 5 files (4 modalities + segmentation) exist for each patient and skip incomplete cases with a warning.

2. `create_splits(data_list, train_ratio, val_ratio, test_ratio, seed)` — deterministic split based on `RANDOM_SEED`. Saves to `splits.json` if not exists, loads from it if exists. The split is at the patient level.

3. `ConvertBratsLabelsd(MapTransform)` — MONAI MapTransform that converts the raw BraTS 2023 mask (values 0/1/2/3) into a 3-channel binary tensor:
   - **Channel 0 (WT):** `(label == 1) | (label == 2) | (label == 3)`
   - **Channel 1 (TC):** `(label == 1) | (label == 3)`
   - **Channel 2 (ET):** `(label == 3)`

   Implementation sketch:
   ```python
   from monai.transforms import MapTransform
   import torch

   class ConvertBratsLabelsd(MapTransform):
       """Convert BraTS 2023 labels (0,1,2,3) to 3-channel WT/TC/ET binary mask."""
       def __call__(self, data):
           d = dict(data)
           for key in self.keys:
               label = d[key]
               result = [
                   (label == 1) | (label == 2) | (label == 3),  # WT
                   (label == 1) | (label == 3),                  # TC
                   (label == 3),                                  # ET
               ]
               d[key] = torch.stack(result, dim=0).float() if isinstance(label, torch.Tensor) \
                        else np.stack(result, axis=0).astype("float32")
               # Squeeze the original channel dim if present
               if d[key].ndim == 5:
                   d[key] = d[key].squeeze(1)
           return d
   ```
   **Note on the inverse mapping:** when converting predictions back to a single-label mask in `predict.py`, the inverse for BraTS 2023 is: `ET → 3`, `(TC and not ET) → 1`, `(WT and not TC) → 2`. Do **not** use `4` anywhere in the output.

4. `get_dataloaders(config)` — returns `train_loader`, `val_loader`, `test_loader` using MONAI `Dataset` (or `CacheDataset` if RAM allows) and PyTorch `DataLoader`.

### 5.3 `transforms.py`

Two MONAI transform pipelines: one for training (with augmentations) and one for validation/test (without augmentations). The `ConvertBratsLabelsd` transform must be the BraTS 2023 version (mapping 0/1/2/3 → WT/TC/ET).

**Training pipeline (`get_train_transforms()`):**

1. `LoadImaged(keys=["image", "label"])` — load NIfTI files
2. `EnsureChannelFirstd(keys=["image", "label"])` — channel-first format
3. `ConvertBratsLabelsd(keys="label")` — BraTS 2023 conversion (0/1/2/3 → 3 binary channels)
4. `Orientationd(keys=["image", "label"], axcodes="RAS")` — standard orientation
5. `NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)` — normalize per modality, only on non-zero (brain) voxels
6. `RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=PATCH_SIZE, pos=1, neg=1, num_samples=2)` — sample patches with positive/negative balance
7. `RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0)`
8. `RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1)`
9. `RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2)`
10. `RandRotate90d(keys=["image", "label"], prob=0.5)`
11. `RandScaleIntensityd(keys="image", factors=0.1, prob=0.5)`
12. `RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5)`
13. `RandGaussianNoised(keys="image", prob=0.2, mean=0.0, std=0.1)`
14. `EnsureTyped(keys=["image", "label"])`

**Validation/test pipeline (`get_val_transforms()`):**

1. `LoadImaged`
2. `EnsureChannelFirstd`
3. `ConvertBratsLabelsd` (BraTS 2023 version)
4. `Orientationd(axcodes="RAS")`
5. `NormalizeIntensityd(nonzero=True, channel_wise=True)`
6. `EnsureTyped`

No cropping or augmentation — full volumes are passed through sliding window inference at validation time.

### 5.4 `model.py`

Defines the model architecture(s). **No changes needed for the BraTS 2023 switch** — the model interface (4 input channels, 3 output channels) is identical.

**Required function:**

```python
def get_model(config):
    """
    Returns a PyTorch model based on config.MODEL_NAME.
    Supported: "UNet", "AttentionUNet", "SegResNet"
    """
```

**Primary model — 3D U-Net via MONAI:**

```python
from monai.networks.nets import UNet

model = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=3,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm="INSTANCE",
)
```

**Alternative architectures (for comparison experiments):**

- `AttentionUnet` from `monai.networks.nets`
- `SegResNet` from `monai.networks.nets`

All alternatives must accept the same `in_channels=4`, `out_channels=3` interface.

### 5.5 `train.py`

Training loop with logging, checkpointing, and early stopping. **No changes needed for the BraTS 2023 switch** — the loss, metrics, and inference logic operate on the 3-channel binary representation, which is independent of the underlying raw label values.

**Required logic:**

1. Load config, set random seeds (`torch.manual_seed`, `np.random.seed`)
2. Initialize model via `get_model(config)`, move to device
3. Initialize loss: `DiceCELoss(sigmoid=True, include_background=False)`
4. Initialize optimizer: `AdamW(lr=1e-4, weight_decay=1e-5)`
5. Initialize scheduler: `CosineAnnealingLR(T_max=NUM_EPOCHS)`
6. Initialize TensorBoard SummaryWriter
7. Initialize MONAI metrics: `DiceMetric(include_background=False, reduction="mean_batch")`
8. For each epoch:
   - Training step: forward pass, loss, backward, optimizer.step
   - Log loss to TensorBoard
   - Every `VAL_INTERVAL` epochs: run validation using `sliding_window_inference` from `monai.inferers`
   - Compute mean Dice over validation set (separate for WT, TC, ET)
   - If best mean Dice → save checkpoint to `outputs/checkpoints/best_model.pth`
   - Always save `last_model.pth`
   - Check early stopping
9. At the end, save final model and training history

**Sliding window inference parameters:**

```python
from monai.inferers import sliding_window_inference

val_outputs = sliding_window_inference(
    inputs=val_images,
    roi_size=PATCH_SIZE,
    sw_batch_size=1,
    predictor=model,
    overlap=0.5,
    mode="gaussian",
)
```

### 5.6 `evaluate.py`

Loads the best checkpoint and evaluates on the test set. The metrics operate on the 3-channel binary representation, so this file does not need BraTS-version-specific logic — it only depends on `dataset.py` and `transforms.py` to deliver correctly converted labels.

**Required logic:**

1. Load config and best model checkpoint
2. Get test dataloader
3. For each test patient:
   - Run sliding window inference
   - Apply sigmoid + threshold (0.5)
   - Compute per-patient metrics:
     - `DiceMetric` (WT, TC, ET)
     - `MeanIoU`
     - `HausdorffDistanceMetric(percentile=95)` (WT, TC, ET)
4. Aggregate results: mean ± std for each metric and each region
5. Save per-patient results to `outputs/results/test_results.csv`
6. Save summary statistics to `outputs/results/test_summary.csv`
7. Print results to console

### 5.7 `predict.py`

Inference on a single patient (a folder containing 4 modality NIfTI files following the BraTS 2023 naming convention).

**Required logic:**

1. Parse command-line arguments: `--input_dir`, `--output_dir`, `--checkpoint`
2. Load model from checkpoint
3. Load 4 modalities using the BraTS 2023 suffixes (`t1n`, `t1c`, `t2w`, `t2f`), apply validation transforms
4. Run sliding window inference
5. Apply sigmoid + threshold (0.5)
6. Post-processing:
   - Remove small connected components (< 50 voxels) using `scipy.ndimage.label`
   - Optional: morphological closing for smoother boundaries
7. **Convert 3-channel binary output back to a single-label NIfTI mask using the BraTS 2023 label convention (0/1/2/3):**
   - Start with an all-zero array of the same shape as one channel
   - Where WT (channel 0) is 1 → set voxel to `2` (edema by default)
   - Where TC (channel 1) is 1 → overwrite with `1` (necrotic core)
   - Where ET (channel 2) is 1 → overwrite with `3` (enhancing tumor)
   - **Do not use the value `4` anywhere — that would be the BraTS 2021 convention.**
8. Save the mask as `prediction.nii.gz` in `output_dir`
9. Generate a visualization: middle axial slice of FLAIR with overlaid predicted mask, saved as PNG

---

## 6. Training Configuration Details

### 6.1 Loss Function

`DiceCELoss` from MONAI is the recommended choice — it combines Dice loss (good for class imbalance) with Cross-Entropy (stable gradients):

```python
from monai.losses import DiceCELoss

loss_function = DiceCELoss(
    to_onehot_y=False,
    sigmoid=True,
    include_background=False,
    lambda_dice=1.0,
    lambda_ce=1.0,
)
```

`sigmoid=True` is required because the 3 output channels are not mutually exclusive (ET ⊂ TC ⊂ WT), so we use independent sigmoid activations rather than softmax.

### 6.2 Metrics

```python
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU

dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
iou_metric = MeanIoU(include_background=False, reduction="mean_batch")
hausdorff_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch")
```

All metrics report per-region values (WT, TC, ET separately), which matches the BraTS evaluation protocol.

### 6.3 Hardware Requirements

- **GPU:** Minimum 8 GB VRAM (NVIDIA), recommended 12+ GB
- **RAM:** Minimum 16 GB (32 GB if using CacheDataset)
- **Disk:** ~150 GB free space for dataset + checkpoints + logs
- **CUDA:** 11.8 or 12.1

If GPU memory is limited, reduce `BATCH_SIZE` to 1 or `PATCH_SIZE` to (96, 96, 96).

---

## 7. Implementation Order

Implement files in this strict order. Each file must be testable independently before moving to the next:

1. **`requirements.txt`** — list all dependencies
2. **`config.py`** — all configuration constants (verify `BRATS_VERSION = "2023"` and the label values)
3. **`dataset.py`** — verify by:
   - Printing the first sample's image and label paths (must contain `t1n`, `t1c`, `t2w`, `t2f`, `seg` suffixes)
   - Printing `np.unique(label)` on a raw loaded mask — **must be `[0, 1, 2, 3]`**, not `[0, 1, 2, 4]`
4. **`transforms.py`** — verify by applying transforms to one sample. After `ConvertBratsLabelsd`, the label must have shape `(3, H, W, D)` and each channel must contain only `{0.0, 1.0}`. Verify that `WT.sum() >= TC.sum() >= ET.sum()` (region nesting).
5. **`model.py`** — verify by passing a dummy tensor `torch.randn(1, 4, 128, 128, 128)` and checking output shape `(1, 3, 128, 128, 128)`
6. **`train.py`** — run for 2 epochs to verify the loop works without errors
7. **`evaluate.py`** — run on a small subset to verify metrics computation
8. **`predict.py`** — verify by running on one test patient. Open the output `prediction.nii.gz` and confirm `np.unique` is a subset of `{0, 1, 2, 3}` (BraTS 2023 convention) — **never `4`**.

---

## 8. Expected Results

Based on published BraTS 2023 results, a properly trained 3D U-Net should achieve approximately:

| Region | Dice Score | HD95 (mm) |
|--------|------------|-----------|
| Whole Tumor (WT) | 0.88 – 0.92 | 4 – 8 |
| Tumor Core (TC) | 0.83 – 0.88 | 5 – 10 |
| Enhancing Tumor (ET) | 0.78 – 0.85 | 5 – 12 |

If results are significantly below these ranges, check:
- **Label conversion (the 0/1/2/3 → WT/TC/ET mapping for BraTS 2023)** — by far the most common bug
- Normalization (must be per-channel, on non-zero voxels only)
- Number of training epochs (200 is a reasonable default)
- Loss function configuration (`include_background=False`)

---

## 9. Critical Implementation Notes

1. **Label conversion is the #1 source of bugs.** Always print `np.unique(label)` on a raw segmentation mask after loading to verify values are `{0, 1, 2, 3}` (BraTS 2023). If you see `{0, 1, 2, 4}`, the data is actually BraTS 2021 and the conversion logic must use `label == 4` instead of `label == 3` for the ET channel.

2. **The ET channel uses value 3, not 4.** This is the single most important change from BraTS 2021. Any tutorial, blog post, or GitHub repo written before mid-2023 will use `label == 4` for ET — do not copy that code blindly.

3. **File naming uses new suffixes.** BraTS 2023 files are named `*-t1n.nii.gz`, `*-t1c.nii.gz`, `*-t2w.nii.gz`, `*-t2f.nii.gz`, `*-seg.nii.gz`. The old `*_t1.nii.gz`, `*_t1ce.nii.gz`, `*_t2.nii.gz`, `*_flair.nii.gz` pattern from BraTS 2021 will not match anything.

4. **Patient folder names include a session ID.** BraTS 2023 folders are named like `BraTS-GLI-00000-000` (the trailing `-000` is the timepoint). Do not strip it — it is part of the file prefix inside the folder.

5. **Normalization must be per-modality.** Each of the 4 modalities has different intensity ranges and must be normalized independently using `channel_wise=True`.

6. **Use sliding window inference for validation, not patches.** Training on patches but validating on full volumes via sliding window is essential — otherwise validation metrics will be misleading.

7. **Save the random seed and splits.json.** Without these, results are not reproducible and you cannot fairly compare different experiments.

8. **Always set `include_background=False`** in loss and metrics. The background class dominates and would inflate metrics misleadingly.

9. **Monitor 3 separate Dice scores during training**, not just the mean. ET is typically the hardest and converges slowest — looking only at the mean hides this.

10. **Inverse label mapping in `predict.py` must use BraTS 2023 values (0/1/2/3).** Never write `4` to the output mask.

---

## 10. Deliverables

After full implementation, the project should produce:

- Trained model weights at `outputs/checkpoints/best_model.pth`
- TensorBoard logs at `outputs/logs/`
- Test set evaluation results at `outputs/results/test_results.csv` and `test_summary.csv`
- Example predictions at `outputs/predictions/` (with label values in `{0, 1, 2, 3}`)
- Reproducible data split at `splits.json`

---

## 11. Summary of Changes from the Previous (BraTS 2021) Specification

For convenience when updating an existing codebase, here is a complete list of what changed:

| Area | BraTS 2021 (old) | BraTS 2023 (new) |
|------|------------------|-------------------|
| `config.BRATS_VERSION` | `"2021"` | `"2023"` |
| `config.DATA_ROOT` | `"data/BraTS2021"` | `"data/BraTS2023"` |
| ET label value | `4` | `3` |
| Unique label values | `{0, 1, 2, 4}` | `{0, 1, 2, 3}` |
| Patient folder name | `BraTS2021_00000` | `BraTS-GLI-00000-000` |
| T1 suffix | `_t1.nii.gz` | `-t1n.nii.gz` |
| T1ce suffix | `_t1ce.nii.gz` | `-t1c.nii.gz` |
| T2 suffix | `_t2.nii.gz` | `-t2w.nii.gz` |
| FLAIR suffix | `_flair.nii.gz` | `-t2f.nii.gz` |
| Segmentation suffix | `_seg.nii.gz` | `-seg.nii.gz` |
| WT mask formula | `(L==1)\|(L==2)\|(L==4)` | `(L==1)\|(L==2)\|(L==3)` |
| TC mask formula | `(L==1)\|(L==4)` | `(L==1)\|(L==3)` |
| ET mask formula | `(L==4)` | `(L==3)` |
| Inverse mapping in `predict.py` | ET→4, TC→1, WT→2 | ET→3, TC→1, WT→2 |

**Files that need to change:** `config.py`, `dataset.py`, `transforms.py` (only the `ConvertBratsLabelsd` import/usage), `predict.py` (inverse label mapping).
**Files that do NOT change:** `model.py`, `train.py`, `evaluate.py` — they operate on the 3-channel binary representation and are agnostic to the underlying raw label values.

---

**End of Technical Specification**
