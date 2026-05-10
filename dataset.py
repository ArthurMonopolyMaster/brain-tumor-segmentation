"""
BraTS 2023 data loading: directory scanning, label conversion, train/val/test splits.
"""

import json
import os
import warnings

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import MapTransform
from sklearn.model_selection import train_test_split

import config


# ──────────────────────────────────────────────────────
# Label conversion transform
# ──────────────────────────────────────────────────────

class ConvertBratsLabelsd(MapTransform):
    """Convert BraTS 2023 raw labels (0,1,2,3) to 3-channel binary mask: WT / TC / ET."""

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]

            wt = (label == 1) | (label == 2) | (label == 3)  # Whole Tumor
            tc = (label == 1) | (label == 3)                  # Tumor Core
            et = (label == 3)                                  # Enhancing Tumor

            if isinstance(label, torch.Tensor):
                result = torch.stack([wt, tc, et], dim=0).float()
            else:
                result = np.stack([wt, tc, et], axis=0).astype(np.float32)

            # Squeeze extra channel dim that EnsureChannelFirst may have added
            if result.ndim == 5:
                result = result.squeeze(1)

            d[key] = result
        return d


# ──────────────────────────────────────────────────────
# Directory scanning
# ──────────────────────────────────────────────────────

def scan_brats_directory(data_root: str) -> list[dict]:
    """
    Scan a BraTS 2023 directory and return a list of dicts:
      {"image": [path_t1n, path_t1c, path_t2w, path_t2f], "label": path_seg, "patient_id": ...}

    Підтримує дві структури:
      A) <patient>-<suffix>.nii/<arbitrary_name>.nii  (модальність у підпапці)
      B) <patient>-<suffix>.nii                        (модальність як файл напряму)
    """
    data_list = []
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    for patient_dir in sorted(os.listdir(data_root)):
        patient_path = os.path.join(data_root, patient_dir)
        if not os.path.isdir(patient_path):
            continue

        # Збираємо шляхи до 4 модальностей
        image_paths = []
        all_exist = True
        for modality in config.MODALITIES:
            suffix = config.MODALITY_SUFFIXES[modality]
            base_name = f"{patient_dir}-{suffix}.nii"
            base_path = os.path.join(patient_path, base_name)

            # Підтримка обох структур
            if os.path.isfile(base_path):
                # Структура Б: звичайний файл
                image_paths.append(base_path)
            elif os.path.isdir(base_path):
                # Структура А: підпапка з .nii файлом всередині
                nii_files = [f for f in os.listdir(base_path) if f.endswith(".nii")]
                if not nii_files:
                    warnings.warn(f"No .nii files in: {base_path}")
                    all_exist = False
                    break
                image_paths.append(os.path.join(base_path, nii_files[0]))
            else:
                warnings.warn(f"Missing modality (neither file nor dir): {base_path}")
                all_exist = False
                break

        # Маска — завжди файл
        label_path = os.path.join(patient_path, f"{patient_dir}-{config.LABEL_SUFFIX}.nii")
        if not os.path.isfile(label_path):
            warnings.warn(f"Missing label file: {label_path}")
            all_exist = False

        if not all_exist:
            warnings.warn(f"Skipping incomplete patient: {patient_dir}")
            continue

        data_list.append({
            "image": image_paths,
            "label": label_path,
            "patient_id": patient_dir,
        })

    print(f"Found {len(data_list)} complete patients in {data_root}")
    return data_list


# ──────────────────────────────────────────────────────
# Data splits
# ──────────────────────────────────────────────────────

def create_splits(
    data_list: list[dict],
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
    seed: int = config.RANDOM_SEED,
) -> dict[str, list[dict]]:
    """
    Split data_list into train/val/test.
    Saves to / loads from config.SPLITS_FILE for reproducibility.
    """
    if os.path.isfile(config.SPLITS_FILE):
        print(f"Loading existing splits from {config.SPLITS_FILE}")
        with open(config.SPLITS_FILE, "r") as f:
            splits_ids = json.load(f)

        id_to_item = {item["patient_id"]: item for item in data_list}
        splits = {}
        for key in ("train", "val", "test"):
            splits[key] = [id_to_item[pid] for pid in splits_ids[key] if pid in id_to_item]
        return splits

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_data, val_test_data = train_test_split(
        data_list, test_size=val_test_ratio, random_state=seed,
    )

    # Second split: val vs test
    relative_test = test_ratio / val_test_ratio
    val_data, test_data = train_test_split(
        val_test_data, test_size=relative_test, random_state=seed,
    )

    print(f"Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # Save patient IDs for reproducibility
    splits_ids = {
        "train": [d["patient_id"] for d in train_data],
        "val":   [d["patient_id"] for d in val_data],
        "test":  [d["patient_id"] for d in test_data],
    }
    with open(config.SPLITS_FILE, "w") as f:
        json.dump(splits_ids, f, indent=2)
    print(f"Saved splits to {config.SPLITS_FILE}")

    return {"train": train_data, "val": val_data, "test": test_data}


# ──────────────────────────────────────────────────────
# DataLoaders
# ──────────────────────────────────────────────────────

def get_dataloaders():
    """
    Returns (train_loader, val_loader, test_loader).
    Uses CacheDataset for faster training if enough RAM is available.
    """
    from transforms import get_train_transforms, get_val_transforms

    data_list = scan_brats_directory(config.DATA_ROOT)
    splits = create_splits(data_list)
    if config.DEBUG_NUM_PATIENTS is not None:
        n = config.DEBUG_NUM_PATIENTS
        splits = {
            "train": splits["train"][:max(n - 2, 1)],
            "val": splits["val"][:1],
            "test": splits["test"][:1],
        }
        print(f"DEBUG MODE: train={len(splits['train'])}, "
              f"val={len(splits['val'])}, test={len(splits['test'])}")
    train_ds = CacheDataset(
        data=splits["train"],
        transform=get_train_transforms(),
        cache_rate=1,
        num_workers=config.NUM_WORKERS,
    )
    val_ds = CacheDataset(
        data=splits["val"],
        transform=get_val_transforms(),
        cache_rate=1,
        num_workers=config.NUM_WORKERS,
    )
    test_ds = Dataset(
        data=splits["test"],
        transform=get_val_transforms(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader
