import cv2
import torch
import numpy as np
import os
import argparse
from src.models.unet import RoadSegModel, LaneSegModel

def predict_single_model(tensor, model):
    with torch.no_grad():
        out = model(tensor)
        pred = torch.sigmoid(out).squeeze().cpu().numpy()
    return pred

def prepare_tensor(img_resized, device, mean, std):
    rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
    rgb_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    rgb_norm = (rgb_clahe.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.tensor(rgb_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kitti_path', type=str, default="Dataset/training/image_2", help="Thư mục ảnh KITTI (Đường)")
    parser.add_argument('--kaggle_path', type=str, default="LaneDataset/train2017", help="Thư mục ảnh Kaggle (Vạch kẻ)")
    parser.add_argument('--road_weights', type=str, default="road_model.pth", help="File weights đường")
    parser.add_argument('--lane_weights', type=str, default="lane_model.pth", help="File weights vạch kẻ")
    parser.add_argument('--limit', type=int, default=0, help="Số lượng ảnh test tối đa (0 = test tất cả)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Bat dau qua trinh Du doan quy mo lon tren: {device}")
    
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
    
    # Thư mục kết quả mới
    result_road = "result/result_road"
    result_lane = "result/result_lane"
    result_combined = "result/result_train"
    os.makedirs(result_road, exist_ok=True)
    os.makedirs(result_lane, exist_ok=True)
    os.makedirs(result_combined, exist_ok=True)

    # Khởi tạo model
    road_model = RoadSegModel().to(device)
    lane_model = LaneSegModel().to(device)

    if not os.path.exists(args.road_weights) or not os.path.exists(args.lane_weights):
        print("Thieu file models. Vui long train du 2 models.")
        return

    road_model.load_state_dict(torch.load(args.road_weights, map_location=device))
    lane_model.load_state_dict(torch.load(args.lane_weights, map_location=device))
    
    road_model.eval()
    lane_model.eval()
    
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # ==========================================
    # PHẦN 1: TEST ĐƯỜNG (ROAD) TRÊN KITTI
    # ==========================================
    print(f"\n--- PHẦN 1: TẠO ẢNH PHÂN VÙNG ĐƯỜNG (TESTROAD) ---")
    if os.path.exists(args.kitti_path):
        kitti_images = sorted(os.listdir(args.kitti_path))
        if args.limit > 0: kitti_images = kitti_images[:args.limit]
        
        for idx, img_name in enumerate(kitti_images):
            img_path = os.path.join(args.kitti_path, img_name)
            img = cv2.imread(img_path)
            if img is None: continue

            original_size = img.shape[:2]
            img_resized = cv2.resize(img, (640, 384))
            
            tensor = prepare_tensor(img_resized, device, MEAN, STD)
            road_pred = predict_single_model(tensor, road_model)
            road_mask = (road_pred > 0.5).astype(np.uint8)

            img_road = img_resized.copy()
            green_overlay = img_road.copy()
            green_overlay[road_mask == 1] = [180, 255, 0] # BGR: Cyan
            mask_3ch = np.stack([road_mask]*3, axis=-1)
            img_road = np.where(mask_3ch == 1, cv2.addWeighted(img_road, 0.5, green_overlay, 0.5, 0), img_road)

            out_road = cv2.resize(img_road, (original_size[1], original_size[0]))
            cv2.imwrite(os.path.join(result_road, img_name), out_road)
            
            if (idx + 1) % 50 == 0:
                print(f"Đã xử lý {idx+1}/{len(kitti_images)} ảnh KITTI...")
    else:
        print(f"Khong tim thay thu muc {args.kitti_path}")

    # ==========================================
    # PHẦN 2 & 3: TEST VẠCH KẺ & TỔNG HỢP TRÊN KAGGLE
    # ==========================================
    print(f"\n--- PHẦN 2 & 3: TẠO ẢNH VẠCH KẺ (TESTLANE) VÀ TỔNG HỢP (TEST) ---")
    if os.path.exists(args.kaggle_path):
        kaggle_images = sorted(os.listdir(args.kaggle_path))
        
        # Chỉ lấy ảnh, loại bỏ các file .json
        kaggle_images = [img for img in kaggle_images if img.endswith(('.jpg', '.png'))]
        if args.limit > 0: kaggle_images = kaggle_images[:args.limit]
        
        for idx, img_name in enumerate(kaggle_images):
            img_path = os.path.join(args.kaggle_path, img_name)
            img = cv2.imread(img_path)
            if img is None: continue

            original_size = img.shape[:2]
            img_resized = cv2.resize(img, (640, 384))
            
            tensor = prepare_tensor(img_resized, device, MEAN, STD)
            
            # Chạy qua GIAO NGHIỆM của cả 2 mô hình chung 1 lúc để tiết kiệm tài nguyên tính toán
            road_pred = predict_single_model(tensor, road_model)
            lane_pred = predict_single_model(tensor, lane_model)
            
            road_mask = (road_pred > 0.5).astype(np.uint8)
            lane_mask = (lane_pred > 0.4).astype(np.uint8)

            # RENDER: Ảnh chỉ có Vạch kẻ (Lane Only -> testlane)
            lane_mask_dilated = cv2.dilate(lane_mask, kernel_dilate, iterations=1)
            img_lane = img_resized.copy()
            img_lane[lane_mask_dilated == 1] = [0, 0, 255] # Red
            out_lane = cv2.resize(img_lane, (original_size[1], original_size[0]))
            cv2.imwrite(os.path.join(result_lane, img_name), out_lane)

            # RENDER: Ảnh kết hợp Cả hai (Hybrid -> test)
            img_combined = img_resized.copy()
            
            # Vẽ đường trước
            green_overlay = img_combined.copy()
            green_overlay[road_mask == 1] = [180, 255, 0]
            mask_3ch = np.stack([road_mask]*3, axis=-1)
            img_combined = np.where(mask_3ch == 1, cv2.addWeighted(img_combined, 0.5, green_overlay, 0.5, 0), img_combined)
            
            # Vẽ vạch sau (Vẽ tất cả các vạch tìm được)
            img_combined[lane_mask_dilated == 1] = [0, 0, 255] # Red
            
            out_combined = cv2.resize(img_combined, (original_size[1], original_size[0]))
            cv2.imwrite(os.path.join(result_combined, img_name), out_combined)

            if (idx + 1) % 50 == 0:
                print(f"Đã ghép xong {idx+1}/{len(kaggle_images)} ảnh Kaggle...")
    else:
        print(f"Khong tim thay thu muc {args.kaggle_path}")

    print("=" * 60)
    print(f"HOAN TAT KIEM THU XUYEN DU LIEU! Cac file duoc to chuc tai:\n  - {result_road}\n  - {result_lane}\n  - {result_combined}")
    print("=" * 60)

if __name__ == "__main__":
    main()
