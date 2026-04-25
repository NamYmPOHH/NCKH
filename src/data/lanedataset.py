import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class RoadDataset(Dataset):
    """
    DataLoader cho KITTI Road Segmentation.
    Trích xuất Binary mask: Hồng/Tím = Road (1), còn lại = Background (0).
    """
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    def __init__(self, image_dir, mask_dir, augment=False, img_size=(640, 384)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.augment = augment
        self.img_size = img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = cv2.imread(os.path.join(self.image_dir, img_name))

        # KITTI mask naming: um_000000.png -> um_road_000000.png
        mask_name = img_name.replace("_", "_road_")
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name))

        if image is None or mask is None:
            raise FileNotFoundError(f"Không tìm thấy: {img_name} / {mask_name}")

        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Binary mask: Hồng/Tím (Road) = 1, còn lại = 0
        # KITTI: Road pixels có Blue > 200 AND Red > 200 (màu hồng/tím)
        road_mask = ((mask[:, :, 0] > 180) & (mask[:, :, 2] > 180)).astype(np.float32)

        # Augmentation
        if self.augment:
            if np.random.random() > 0.5:
                image = cv2.flip(image, 1)
                road_mask = cv2.flip(road_mask, 1)
            if np.random.random() > 0.5:
                image = np.clip(image * np.random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)

        # CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Normalize
        image = image / 255.0
        image = (image - self.MEAN) / self.STD

        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)
        road_mask = torch.tensor(road_mask, dtype=torch.float).unsqueeze(0)

        return image, road_mask
