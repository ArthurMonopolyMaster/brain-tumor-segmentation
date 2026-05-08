"""
Training loop with logging, checkpointing, and early stopping.
"""

import os
import time

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import get_dataloaders
from model import get_model


def train():
    # ── Reproducibility ──
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # ── Device ──
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data ──
    train_loader, val_loader, _ = get_dataloaders()

    # ── Model ──
    model = get_model().to(device)
    print(f"Model: {config.MODEL_NAME}, Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss, optimizer, scheduler ──
    loss_function = DiceCELoss(
        to_onehot_y=False,
        sigmoid=True,
        include_background=True,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS,
    )

    # ── Metrics ──
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")

    # ── TensorBoard ──
    writer = SummaryWriter(log_dir=config.LOG_DIR)

    # ── Checkpoint resume ──
    start_epoch = 0
    best_dice = -1.0
    epochs_no_improve = 0

    last_ckpt = os.path.join(config.CHECKPOINT_DIR, "last_model.pth")
    if os.path.isfile(last_ckpt):
        print(f"Resuming from {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_dice = ckpt.get("best_dice", -1.0)
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)

    # ── Training loop ──
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        step = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        avg_loss = epoch_loss / step
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
        print(f"Epoch {epoch+1} — avg loss: {avg_loss:.4f}")

        # ── Validation ──
        if (epoch + 1) % config.VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                for val_batch in val_loader:
                    val_images = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device)

                    val_outputs = sliding_window_inference(
                        inputs=val_images,
                        roi_size=config.PATCH_SIZE,
                        sw_batch_size=1,
                        predictor=model,
                        overlap=config.SLIDING_WINDOW_OVERLAP,
                        mode="gaussian",
                    )

                    val_outputs = (torch.sigmoid(val_outputs) > 0.5).float()
                    dice_metric(y_pred=val_outputs, y=val_labels)

                dice_values = dice_metric.aggregate()
                dice_metric.reset()

                dice_wt = dice_values[0].item()
                dice_tc = dice_values[1].item()
                dice_et = dice_values[2].item()
                mean_dice = dice_values.mean().item()

                writer.add_scalar("val/dice_wt", dice_wt, epoch)
                writer.add_scalar("val/dice_tc", dice_tc, epoch)
                writer.add_scalar("val/dice_et", dice_et, epoch)
                writer.add_scalar("val/dice_mean", mean_dice, epoch)

                print(
                    f"  Val Dice — WT: {dice_wt:.4f}, TC: {dice_tc:.4f}, "
                    f"ET: {dice_et:.4f}, Mean: {mean_dice:.4f}"
                )

                # ── Save best model ──
                if mean_dice > best_dice:
                    best_dice = mean_dice
                    epochs_no_improve = 0
                    torch.save(
                        model.state_dict(),
                        os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
                    )
                    print(f"  New best model saved (Dice: {best_dice:.4f})")
                else:
                    epochs_no_improve += config.VAL_INTERVAL

                # ── Early stopping ──
                if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # ── Save last checkpoint (for resume) ──
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_dice,
                "epochs_no_improve": epochs_no_improve,
            },
            last_ckpt,
        )

    writer.close()
    print(f"Training complete. Best mean Dice: {best_dice:.4f}")


if __name__ == "__main__":
    train()
