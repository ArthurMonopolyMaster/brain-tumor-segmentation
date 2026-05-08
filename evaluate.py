"""
Evaluation on the test set: Dice, IoU, Hausdorff Distance 95 per patient and per region.
"""

import os

import numpy as np
import pandas as pd
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU
from tqdm import tqdm

import config
from dataset import get_dataloaders
from model import get_model

REGION_NAMES = ["WT", "TC", "ET"]


def evaluate():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data ──
    _, _, test_loader = get_dataloaders()

    # ── Model ──
    model = get_model().to(device)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # ── Metrics ──
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    iou_metric = MeanIoU(include_background=True, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(
        include_background=True, percentile=95, reduction="mean_batch",
    )

    patient_results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            patient_id = batch["patient_id"][0]

            outputs = sliding_window_inference(
                inputs=images,
                roi_size=config.PATCH_SIZE,
                sw_batch_size=1,
                predictor=model,
                overlap=config.SLIDING_WINDOW_OVERLAP,
                mode="gaussian",
            )
            preds = (torch.sigmoid(outputs) > 0.5).float()

            # Per-patient metrics
            dice_metric(y_pred=preds, y=labels)
            iou_metric(y_pred=preds, y=labels)
            hd95_metric(y_pred=preds, y=labels)

            dice_vals = dice_metric.aggregate().cpu().numpy()
            iou_vals = iou_metric.aggregate().cpu().numpy()
            hd95_vals = hd95_metric.aggregate().cpu().numpy()

            dice_metric.reset()
            iou_metric.reset()
            hd95_metric.reset()

            row = {"patient_id": patient_id}
            for i, region in enumerate(REGION_NAMES):
                row[f"dice_{region}"] = dice_vals[i]
                row[f"iou_{region}"] = iou_vals[i]
                row[f"hd95_{region}"] = hd95_vals[i]
            patient_results.append(row)

    # ── Save per-patient results ──
    df = pd.DataFrame(patient_results)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    per_patient_path = os.path.join(config.RESULTS_DIR, "test_results.csv")
    df.to_csv(per_patient_path, index=False)
    print(f"Per-patient results saved to {per_patient_path}")

    # ── Summary statistics ──
    summary_rows = []
    metric_cols = [c for c in df.columns if c != "patient_id"]
    for col in metric_cols:
        summary_rows.append({
            "metric": col,
            "mean": df[col].mean(),
            "std": df[col].std(),
            "median": df[col].median(),
            "min": df[col].min(),
            "max": df[col].max(),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(config.RESULTS_DIR, "test_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

    # Print summary
    print("TEST SET RESULTS")
    for region in REGION_NAMES:
        dice_mean = df[f"dice_{region}"].mean()
        dice_std = df[f"dice_{region}"].std()
        iou_mean = df[f"iou_{region}"].mean()
        hd95_mean = df[f"hd95_{region}"].mean()
        print(
            f"  {region}: Dice={dice_mean:.4f}+-{dice_std:.4f}, "
            f"IoU={iou_mean:.4f}, HD95={hd95_mean:.2f}mm"
        )



if __name__ == "__main__":
    evaluate()
