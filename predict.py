"""
Inference on a single patient folder: load MRI, predict segmentation, save NIfTI + visualization.
"""

import argparse
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    EnsureType,
    LoadImage,
    NormalizeIntensity,
    Orientation,
)
from scipy import ndimage

import config
from model import get_model


def postprocess(pred: np.ndarray, min_component_size: int = 50) -> np.ndarray:
    """Remove small connected components (< min_component_size voxels) per channel."""
    cleaned = np.zeros_like(pred)
    for ch in range(pred.shape[0]):
        binary = pred[ch].astype(bool)
        labeled, num_features = ndimage.label(binary)
        for comp_id in range(1, num_features + 1):
            component = labeled == comp_id
            if component.sum() >= min_component_size:
                cleaned[ch][component] = 1.0
    return cleaned


def channels_to_brats_label(pred: np.ndarray) -> np.ndarray:
    """
    Convert 3-channel binary prediction (WT, TC, ET) back to single-label
    BraTS 2023 convention (0/1/2/3). Never uses value 4.
    """
    wt, tc, et = pred[0], pred[1], pred[2]
    output = np.zeros_like(wt, dtype=np.uint8)
    output[wt > 0.5] = config.LABEL_ED   # 2 — edema by default
    output[tc > 0.5] = config.LABEL_NCR   # 1 — necrotic core
    output[et > 0.5] = config.LABEL_ET    # 3 — enhancing tumor
    return output


def predict(input_dir: str, output_dir: str, checkpoint: str):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # ── Model ──
    model = get_model().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded model from {checkpoint}")

    # ── Load modalities ──
    patient_name = os.path.basename(input_dir)
    image_paths = []
    for modality in config.MODALITIES:
        suffix = config.MODALITY_SUFFIXES[modality]
        fpath = os.path.join(input_dir, f"{patient_name}-{suffix}.nii.gz")
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"Missing modality file: {fpath}")
        image_paths.append(fpath)

    # ── Transforms (same as validation, but for raw arrays) ──
    loader = LoadImage(image_only=True)
    transform = Compose([
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        NormalizeIntensity(nonzero=True, channel_wise=True),
        EnsureType(),
    ])

    # Load and stack modalities
    images = [loader(p) for p in image_paths]
    image = np.stack(images, axis=0)  # (4, H, W, D)
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # (1, 4, H, W, D)

    # ── Inference ──
    with torch.no_grad():
        output = sliding_window_inference(
            inputs=image,
            roi_size=config.PATCH_SIZE,
            sw_batch_size=1,
            predictor=model,
            overlap=config.SLIDING_WINDOW_OVERLAP,
            mode="gaussian",
        )
    pred = (torch.sigmoid(output) > 0.5).float().cpu().numpy()[0]  # (3, H, W, D)

    # ── Post-processing ──
    pred = postprocess(pred)

    # ── Convert to BraTS 2023 label format ──
    label_map = channels_to_brats_label(pred)

    # ── Save NIfTI ──
    os.makedirs(output_dir, exist_ok=True)

    # Use FLAIR as reference for affine/header
    flair_suffix = config.MODALITY_SUFFIXES["flair"]
    ref_path = os.path.join(input_dir, f"{patient_name}-{flair_suffix}.nii.gz")
    ref_nii = nib.load(ref_path)

    pred_nii = nib.Nifti1Image(label_map, affine=ref_nii.affine, header=ref_nii.header)
    nifti_path = os.path.join(output_dir, f"{patient_name}_prediction.nii.gz")
    nib.save(pred_nii, nifti_path)
    print(f"Prediction saved: {nifti_path}")
    print(f"Unique labels in output: {np.unique(label_map)}")

    # ── Visualization: middle axial slice ──
    mid_slice = label_map.shape[2] // 2

    flair_data = nib.load(ref_path).get_fdata()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(flair_data[:, :, mid_slice].T, cmap="gray", origin="lower")
    axes[0].set_title("FLAIR")
    axes[0].axis("off")

    axes[1].imshow(label_map[:, :, mid_slice].T, cmap="nipy_spectral", origin="lower", vmin=0, vmax=3)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    axes[2].imshow(flair_data[:, :, mid_slice].T, cmap="gray", origin="lower")
    mask = label_map[:, :, mid_slice].T
    masked = np.ma.masked_where(mask == 0, mask)
    axes[2].imshow(masked, cmap="nipy_spectral", alpha=0.5, origin="lower", vmin=0, vmax=3)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    png_path = os.path.join(output_dir, f"{patient_name}_visualization.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved: {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain tumor segmentation inference")
    parser.add_argument("--input_dir", required=True, help="Path to patient folder with 4 MRI modalities")
    parser.add_argument("--output_dir", default=config.PREDICTIONS_DIR, help="Output directory")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
        help="Path to model checkpoint",
    )
    args = parser.parse_args()
    predict(args.input_dir, args.output_dir, args.checkpoint)
