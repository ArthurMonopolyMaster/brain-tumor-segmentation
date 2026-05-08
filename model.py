"""
Model architecture definitions.
Supports UNet, AttentionUnet, and SegResNet — all with a unified interface.
"""

from monai.networks.nets import AttentionUnet, SegResNet, UNet

import config


def get_model(model_name: str = config.MODEL_NAME):
    """
    Create and return a segmentation model based on model_name.
    All models: in_channels=4 (T1, T1ce, T2, FLAIR), out_channels=3 (WT, TC, ET).
    """
    if model_name == "UNet":
        return UNet(
            spatial_dims=3,
            in_channels=config.NUM_INPUT_CHANNELS,
            out_channels=config.NUM_OUTPUT_CHANNELS,
            channels=config.CHANNELS,
            strides=config.STRIDES,
            num_res_units=config.NUM_RES_UNITS,
            norm="INSTANCE",
        )

    if model_name == "AttentionUNet":
        return AttentionUnet(
            spatial_dims=3,
            in_channels=config.NUM_INPUT_CHANNELS,
            out_channels=config.NUM_OUTPUT_CHANNELS,
            channels=config.CHANNELS,
            strides=config.STRIDES,
        )

    if model_name == "SegResNet":
        return SegResNet(
            spatial_dims=3,
            in_channels=config.NUM_INPUT_CHANNELS,
            out_channels=config.NUM_OUTPUT_CHANNELS,
            init_filters=32,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            norm="INSTANCE",
        )

    raise ValueError(f"Unknown model: {model_name}. Choose from: UNet, AttentionUNet, SegResNet")
