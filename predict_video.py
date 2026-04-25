import cv2
import torch
import numpy as np
import os
import argparse
from src.models.unet import RoadSegModel, LaneSegModel

def prepare_tensor(img_resized, device, mean, std):
    # Cân bằng ánh sáng
    rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
    rgb_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Chuẩn hóa
    rgb_norm = (rgb_clahe.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.tensor(rgb_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Duong dan file video dau vao (.mp4)")
    parser.add_argument('--output', type=str, default="result/result_video/result_video.mp4", help="Duong dan video ket qua (.mp4)")
    parser.add_argument('--road_weights', type=str, default="road_model.pth")
    parser.add_argument('--lane_weights', type=str, default="lane_model.pth")
    args = parser.parse_args()

    # Tao thu muc result neu chua co
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Khoi tao Render Video Dual-Model tren GPU: {device}")
    
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    # Nạp 2 mô hình (Đường + Vạch kẻ)
    if not os.path.exists(args.road_weights) or not os.path.exists(args.lane_weights):
        print(" Thieu model weights (road_model.pth hoac lane_model.pth)!")
        return
    
    road_model = RoadSegModel().to(device)
    lane_model = LaneSegModel().to(device)
    road_model.load_state_dict(torch.load(args.road_weights, map_location=device))
    lane_model.load_state_dict(torch.load(args.lane_weights, map_location=device))
    
    road_model.eval()
    lane_model.eval()
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Đọc Video bằng OpenCV
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f" Loi: Khong the doc video tai {args.input}")
        return

    # Lấy thông số Video gốc
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Cấu hình lưu Video xuất (Chuẩn mp4v)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    print(f" Bat dau chay FPS Inference ({width}x{height} - {fps}fps - Tong {total_frames} frames)")
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        img_resized = cv2.resize(frame, (640, 384))
        tensor = prepare_tensor(img_resized, device, MEAN, STD)
        
        with torch.no_grad():
            road_out = road_model(tensor)
            lane_out = lane_model(tensor)
            
            road_pred = torch.sigmoid(road_out).squeeze().cpu().numpy()
            lane_pred = torch.sigmoid(lane_out).squeeze().cpu().numpy()
            
        road_mask = (road_pred > 0.5).astype(np.uint8)
        lane_mask = (lane_pred > 0.4).astype(np.uint8)

        # ====== GIAI ĐOẠN RENDER VÀO FRAME ======
        lane_mask_dilated = cv2.dilate(lane_mask, kernel_dilate, iterations=1)
        img_combined = img_resized.copy()
        
        # 1. Quét đường Xanh
        green_overlay = img_combined.copy()
        green_overlay[road_mask == 1] = [180, 255, 0] # Cyan BGR
        mask_3ch = np.stack([road_mask]*3, axis=-1)
        img_combined = np.where(mask_3ch == 1, 
                                cv2.addWeighted(img_combined, 0.5, green_overlay, 0.5, 0), 
                                img_combined)
        
        # 2. Quét vạch Đỏ (Vẽ tất cả vạch tìm được)
        img_combined[lane_mask_dilated == 1] = [0, 0, 255] # Red

        # Scale lại bằng 100% video gốc
        final_frame = cv2.resize(img_combined, (width, height))
        out_video.write(final_frame)

        count += 1
        # Báo cáo tiến độ sau mỗi 30 khung hình
        if count % 30 == 0:
            print(f" Tien do: {count}/{total_frames} frames ({(count/total_frames)*100:.1f}%)")

    cap.release()
    out_video.release()
    print("=" * 50)
    print(f" Xuat Video thanh cong!\n Hay tai file '{args.output}' ve may de xem ket qua.")
    print("=" * 50)

if __name__ == "__main__":
    main()
