import torch.nn as nn
import segmentation_models_pytorch as smp

class RoadSegModel(nn.Module):
    """
    Model A: Phân vùng Mặt đường (Binary).
    Dataset: KITTI Road Segmentation.
    Output: 1 channel (Road=1, Background=0).
    """
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, x):
        return self.model(x)


class LaneSegModel(nn.Module):
    """
    Model B: Phân vùng Vạch kẻ đường (Binary).
    Dataset: Kaggle Road Lane Instance Segmentation.
    Output: 1 channel (Lane=1, Background=0).
    """
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, x):
        return self.model(x)
