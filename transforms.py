"""
MONAI transform pipelines for training and validation/test.
"""

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
)

import config
from dataset import ConvertBratsLabelsd


def get_train_transforms():
    """Training pipeline with augmentations and random patch cropping."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ConvertBratsLabelsd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=config.PATCH_SIZE,
            pos=1,
            neg=1,
            num_samples=2,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        RandGaussianNoised(keys="image", prob=0.2, mean=0.0, std=0.1),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms():
    """Validation/test pipeline — no augmentation, full volumes."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ConvertBratsLabelsd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ])
