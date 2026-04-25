import os
import json
import cv2
import numpy as np
import argparse
from pycocotools.coco import COCO

def convert_coco_to_masks(dataset_dir):
    """
    Chuyển đổi file COCO JSON của Kaggle Lane Dataset thành Binary Masks (.png).
    Tất cả các loại vạch kẻ (solid, dotted, divider...) đều gom chung thành 1 class (Lane=255).
    """
    # Giao diện chuẩn của COCO dataset
    json_file = os.path.join(dataset_dir, "annotations", "instances_train2017.json")
        
    if not os.path.exists(json_file):
        print(f" Không tìm thấy file JSON tại: {json_file}")
        return

    # Khởi tạo COCO API
    coco = COCO(json_file)
    
    # Xác định thư mục ảnh và mask
    image_dir = os.path.join(dataset_dir, "train2017")
    mask_dir = os.path.join(dataset_dir, "masks_train2017")
    os.makedirs(mask_dir, exist_ok=True)

    img_ids = coco.getImgIds()
    print(f" Tìm thấy {len(img_ids)} ảnh. Đang tiến hành vẽ binary masks...")

    for count, img_id in enumerate(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        
        # Khởi tạo mask đen
        height = img_info['height']
        width = img_info['width']
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Lấy tất cả annotations cho ảnh này
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Vẽ polygons
        for ann in anns:
            # segmentation là list của list: [[x1, y1, x2, y2, ...]]
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                poly = np.round(poly).astype(np.int32)
                cv2.fillPoly(mask, [poly], 255) # Tô trắng (255) cho vạch kẻ
                
        # Lưu mask bằng tên file ảnh, đổi đuôi thành png (để ko mất chất lượng nén)
        mask_name = os.path.splitext(file_name)[0] + ".png"
        cv2.imwrite(os.path.join(mask_dir, mask_name), mask)

        if (count + 1) % 100 == 0:
            print(f"Đã xử lý {count + 1}/{len(img_ids)} ảnh...")

    print(f" Hoàn tất! Binary Masks được lưu tại: {mask_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Đường dẫn thư mục chứa dataset Kaggle (_annotations.coco.json)")
    args = parser.parse_args()
    
    convert_coco_to_masks(args.input)
