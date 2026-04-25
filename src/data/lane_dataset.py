import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class LaneDataset(Dataset):
    """
    DataLoader cho Kaggle Road Lane Instance Segmentation.
    Trích xuất Binary mask: Trắng (255) = Lane, Đen (0) = Background.
    """
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    def __init__(self, dataset_dir, augment=False, img_size=(640, 384)):
        self.image_dir = os.path.join(dataset_dir, "train2017") 
        self.mask_dir = os.path.join(dataset_dir, "masks_train2017")
        
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Chưa convert masks! Hãy chạy python src/data/coco_to_mask.py trước.")

        # Lấy tên các file mask gốc (vì đuôi mask là .png, còn ảnh có thể là .jpg)
        self.mask_names = [f for f in os.listdir(self.mask_dir) if f.endswith('.png')]
        self.augment = augment
        self.img_size = img_size

    def __len__(self):
        return len(self.mask_names)

    def __getitem__(self, idx):
        mask_name = self.mask_names[idx]
        file_base = os.path.splitext(mask_name)[0]
        
        # Tìm file ảnh gốc (jpg hoặc png)
        img_path = os.path.join(self.image_dir, file_base + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, file_base + ".png")

        image = cv2.imread(img_path)
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise FileNotFoundError(f"Không tìm thấy ảnh hoặc mask cho: {file_base}")

        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Binary mask: 255 -> 1.0, 0 -> 0.0
        lane_mask = (mask > 127).astype(np.float32)

        # Augmentation
        if self.augment:
            if np.random.random() > 0.5:
                image = cv2.flip(image, 1)
                lane_mask = cv2.flip(lane_mask, 1)
            if np.random.random() > 0.5:
                image = np.clip(image * np.random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)

        # Normalize
        image = image / 255.0
        image = (image - self.MEAN) / self.STD

        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)
        lane_mask = torch.tensor(lane_mask, dtype=torch.float).unsqueeze(0)

        return image, lane_mask
