import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.lane_dataset import LaneDataset
from src.models.unet import LaneSegModel
from src.loss import BCEDiceLoss
import sys
import argparse
import os

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help="Đường dẫn thư mục kaggle Dataset (đã có ./masks)")
    parser.add_argument('--resume', action='store_true', help="Tiếp tục học từ file lane_model.pth đã lưu")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Binary Lane Model (Vach ke) tren: {device}")

    # Khởi tạo mô hình Lane
    model = LaneSegModel().to(device)
    
    # Nạp lại weights nếu tiếp tục học
    weights_path = "lane_model.pth"
    if args.resume and os.path.exists(weights_path):
        print("Dang nap lai mo hinh cu de hoc tiep...")
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print("Da nap thanh cong bo nao cu.")
        except Exception as e:
            print(f"Loi nap weights: {e}")

    # Khởi tạo DataLoader
    try:
        # Ảnh 640x384 cho cân bằng tốc độ
        dataset = LaneDataset(args.dataset_path, augment=True, img_size=(640, 384))
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
    except Exception as e:
        print(f"Loi DataLoader: {e}")
        sys.exit(1)

    criterion = BCEDiceLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_loss = float('inf')
    epochs = 40 # Binary hội tụ nhanh

    print("-" * 30)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            loss = criterion(pred, mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        
        with open("train_lane_log.txt", "a") as f:
            f.write(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}\n")
            
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), weights_path)
            print(f"  Best model saved! (loss={best_loss:.4f})")

    print(f"Training Done! Model saved as '{weights_path}'")

    # TU DONG GOI EXCEL EXPORT
    import subprocess
    try:
        subprocess.run([sys.executable, "export_to_excel.py"])
    except Exception as e:
        print(f"Khong the mo Excel tu dong: {e}")

if __name__ == "__main__":
    train()
