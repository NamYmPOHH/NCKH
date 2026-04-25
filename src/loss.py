import torch
import torch.nn as nn

class BCEDiceLoss(nn.Module):
    """
    BCE + Dice Loss cho Binary Segmentation.
    Hiệu quả cho bài toán mất cân bằng (vùng đường/vạch kẻ nhỏ hơn nền rất nhiều).
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        pred_sigmoid = torch.sigmoid(pred)
        smooth = 1e-5
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1 - (2. * intersection + smooth) / (union + smooth)

        return bce_loss + dice_loss.mean()
