import pandas as pd
import re
import os
import tkinter as tk
from tkinter import filedialog

def parse_log(file_path):
    if not os.path.exists(file_path):
        return None
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r'Epoch (\d+) \| Loss: ([\d.]+) \| LR: ([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                lr = float(match.group(3))
                data.append({'Epoch': epoch, 'Loss': loss, 'LR': lr})
    
    if not data:
        return None
        
    df = pd.DataFrame(data)
    
    # Sap xep theo Epoch va chi giu lai ban ghi CUOI CUNG cho moi Epoch (ban ghi moi nhat)
    df = df.sort_values(by=['Epoch']).drop_duplicates(subset=['Epoch'], keep='last').reset_index(drop=True)
    
    return df

def main():
    road_df = parse_log('train_colab_log.txt')
    lane_df = parse_log('train_lane_log.txt')
    
    if road_df is None and lane_df is None:
        print("Khong tim thay bat ky file log nao!")
        return

    # Tao thu muc excel neu chua co
    excel_dir = 'excel'
    os.makedirs(excel_dir, exist_ok=True)

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    print("Dang mo hop thoai chon file...")
    output_file = filedialog.asksaveasfilename(
        title="Chon noi luu file Excel tong hop",
        initialdir=os.path.abspath(excel_dir),
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
        initialfile="Training_Summary.xlsx"
    )

    if not output_file:
        print("Ban da huy chon file.")
        return

    try:
        # Chuan bi du lieu de ghi side-by-side
        final_df = pd.DataFrame()
        
        if road_df is not None:
            # Them cot trong de phan cach neu can
            final_df['Road_Epoch'] = road_df['Epoch']
            final_df['Road_Loss'] = road_df['Loss']
            final_df['Road_LR'] = road_df['LR']
            
            # Tinh Min Loss cho Road
            min_road_loss = road_df['Loss'].min()
            min_row_road = pd.DataFrame({'Road_Epoch': ['MIN LOSS:'], 'Road_Loss': [min_road_loss], 'Road_LR': ['']})
            final_df = pd.concat([final_df, min_row_road], ignore_index=True)

        if lane_df is not None:
            # Tao mot DF tam cho Lane
            temp_lane = pd.DataFrame({
                'Lane_Epoch': lane_df['Epoch'],
                'Lane_Loss': lane_df['Loss'],
                'Lane_LR': lane_df['LR']
            })
            # Tinh Min Loss cho Lane
            min_lane_loss = lane_df['Loss'].min()
            min_row_lane = pd.DataFrame({'Lane_Epoch': ['MIN LOSS:'], 'Lane_Loss': [min_lane_loss], 'Lane_LR': ['']})
            temp_lane = pd.concat([temp_lane, min_row_lane], ignore_index=True)
            
            # Ghep vao DF chinh (dam bao ghep dung hang)
            # Neu road_df dai hon hoac ngan hon, pandas se tu dien NaN
            for col in temp_lane.columns:
                final_df[col] = temp_lane[col]

        # Ghi ra Excel
        final_df.to_excel(output_file, sheet_name='Training_Summary', index=False)

        print(f"\n--- THANH CONG! ---")
        print(f"File da duoc sap xep va luu tai: {output_file}")
        if road_df is not None: print(f"Road Min Loss: {road_df['Loss'].min()}")
        if lane_df is not None: print(f"Lane Min Loss: {lane_df['Loss'].min()}")
        
        # TU DONG MO FILE EXCEL LEN
        os.startfile(output_file)
        
    except Exception as e:
        print(f"Loi khi luu file: {e}")

if __name__ == "__main__":
    main()
