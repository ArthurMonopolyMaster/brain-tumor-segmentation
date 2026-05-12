"""
Central configuration for the Brain Tumor Segmentation project.
All hyperparameters and paths are defined here — no magic numbers elsewhere.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.environ.get(
    "BRATS_DATA_ROOT",
    "D:/BraTS_GLI_2023_diploma/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

OUTPUT_DIR = os.environ.get(
    "BRATS_OUTPUT_DIR",
    os.path.join(PROJECT_ROOT, "outputs")
)
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

SPLITS_FILE = os.environ.get(
    "BRATS_SPLITS_FILE",
    os.path.join(PROJECT_ROOT, "splits.json")
)

# ──────────────────────────────────────────────
# Debug mode (для швидкої перевірки коду перед повним тренуванням)
# ──────────────────────────────────────────────
DEBUG_MODE = os.environ.get("BRATS_DEBUG", "0") == "1"
DEBUG_NUM_PATIENTS = 8 if DEBUG_MODE else None

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
BRATS_VERSION = "2023"
MODALITIES = ["t1", "t1ce", "t2", "flair"]

# Mapping: logical name -> BraTS 2023 file suffix
MODALITY_SUFFIXES = {
    "t1":    "t1n",
    "t1ce":  "t1c",
    "t2":    "t2w",
    "flair": "t2f",
}
LABEL_SUFFIX = "seg"
# Підтримувані розширення NIfTI файлів.
# .nii.gz — стандарт BraTS-2023 (стиснутий).
# .nii — нестиснутий, з'являється коли Kaggle auto-extracts gzip при upload датасету.
# Порядок важливий: спочатку шукаємо .nii.gz (як у специфікації), потім .nii (fallback).
NIFTI_EXTENSIONS = (".nii.gz", ".nii")
NUM_INPUT_CHANNELS = 4
NUM_OUTPUT_CHANNELS = 3  # WT, TC, ET

# BraTS 2023 raw label values
LABEL_BACKGROUND = 0
LABEL_NCR = 1   # Necrotic / non-enhancing tumor core
LABEL_ED  = 2   # Peritumoral edema
LABEL_ET  = 3   # GD-enhancing tumor (was 4 in BraTS 2021!)

# ──────────────────────────────────────────────
# Data split
# ──────────────────────────────────────────────
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
MODEL_NAME = "UNet"  # Options: "UNet", "AttentionUNet", "SegResNet"
CHANNELS = (32, 64, 128, 256, 512)
STRIDES = (2, 2, 2, 2)
NUM_RES_UNITS = 2

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
PATCH_SIZE = (128, 128, 128)
BATCH_SIZE = 2
NUM_EPOCHS = 2 if DEBUG_MODE else 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
OPTIMIZER = "AdamW"
SCHEDULER = "CosineAnnealingLR"

# ──────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────
LOSS_FUNCTION = "DiceCELoss"
INCLUDE_BACKGROUND = False
SIGMOID = True

# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────
VAL_INTERVAL = 1 if DEBUG_MODE else 5  # Validate every N epochs
SLIDING_WINDOW_OVERLAP = 0.5

# ──────────────────────────────────────────────
# Early stopping
# ──────────────────────────────────────────────
EARLY_STOPPING_PATIENCE = 5 if DEBUG_MODE else 30

# ──────────────────────────────────────────────
# Hardware
# ──────────────────────────────────────────────
DEVICE = "cuda"
NUM_WORKERS = 4
PIN_MEMORY = True
