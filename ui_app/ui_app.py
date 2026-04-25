import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AIResultViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Result Viewer - Lane & Road Segmentation")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2c3e50")

        # Duong dan thu muc
        self.result_dir = "result/result_train"
        self.original_dir = "LaneDataset/train2017" # Thu muc anh goc
        self.video_path = "result/result_video/result_video.mp4"
        self.excel_path = "excel/Training_Summary.xlsx"
        
        self.image_list = []
        self.current_index = 0

        # Load danh sach anh tu thu muc ket qua
        if os.path.exists(self.result_dir):
            self.image_list = [f for f in os.listdir(self.result_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.image_list.sort()

        # UI Components
        self.create_widgets()
        
        if self.image_list:
            self.show_image()
        else:
            self.label_info.config(text="Khong tim thay anh trong: " + self.result_dir)

    def create_widgets(self):
        # Header Frame
        self.header_frame = tk.Frame(self.root, bg="#2c3e50")
        self.header_frame.pack(fill=tk.X, pady=10)

        # Dropdown
        self.view_mode = tk.StringVar()
        self.dropdown = ttk.Combobox(self.header_frame, textvariable=self.view_mode, state="readonly", width=18, font=("Helvetica", 12))
        self.dropdown['values'] = ("Ảnh (Result)", "Video (Trained)", "Biểu đồ (Chart)")
        self.dropdown.current(0)
        self.dropdown.pack(side=tk.LEFT, padx=20)
        self.dropdown.bind("<<ComboboxSelected>>", self.on_dropdown_change)

        # Header Title
        self.header = tk.Label(self.header_frame, text="AI SEGMENTATION RESULTS (BEFORE vs AFTER)", font=("Helvetica", 20, "bold"), 
                              bg="#2c3e50", fg="#ecf0f1")
        self.header.pack(side=tk.LEFT, expand=True)

        # Content Area
        self.content_frame = tk.Frame(self.root, bg="#2c3e50")
        self.content_frame.pack(expand=True, fill=tk.BOTH)

        # 1. Image View (Side by Side)
        self.image_view_frame = tk.Frame(self.content_frame, bg="#2c3e50")
        self.image_view_frame.pack(expand=True, fill=tk.BOTH)

        # Khung ben trai: ANH GOC
        self.left_frame = tk.Frame(self.image_view_frame, bg="#34495e")
        self.left_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)
        tk.Label(self.left_frame, text="ORIGINAL (BEFORE)", bg="#34495e", fg="#bdc3c7", font=("Helvetica", 10, "bold")).pack()
        self.original_label = tk.Label(self.left_frame, bg="#34495e")
        self.original_label.pack(expand=True)

        # Khung ben phai: ANH KET QUA
        self.right_frame = tk.Frame(self.image_view_frame, bg="#34495e")
        self.right_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)
        tk.Label(self.right_frame, text="AI RESULT (AFTER)", bg="#34495e", fg="#1abc9c", font=("Helvetica", 10, "bold")).pack()
        self.result_label = tk.Label(self.right_frame, bg="#34495e")
        self.result_label.pack(expand=True)

        # 2. Chart View
        self.chart_canvas = None

        # Info Label
        self.label_info = tk.Label(self.root, text="", font=("Helvetica", 12), bg="#2c3e50", fg="#bdc3c7")
        self.label_info.pack(pady=5)

        # Controls Frame
        self.controls = tk.Frame(self.root, bg="#2c3e50")
        self.controls.pack(pady=20)

        self.btn_prev = ttk.Button(self.controls, text="<< TRÁI", command=self.prev_image)
        self.btn_prev.grid(row=0, column=0, padx=20)

        self.btn_next = ttk.Button(self.controls, text="PHẢI >>", command=self.next_image)
        self.btn_next.grid(row=0, column=1, padx=20)

        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())

    def on_dropdown_change(self, event):
        selected = self.view_mode.get()
        if selected == "Video (Trained)":
            if os.path.exists(self.video_path):
                os.startfile(os.path.abspath(self.video_path))
            else:
                self.label_info.config(text=f"Chua co video tai: {self.video_path}")
            self.dropdown.current(0)
            self.show_image_mode()
        elif selected == "Biểu đồ (Chart)":
            self.show_chart_mode()
        else:
            self.show_image_mode()

    def show_image_mode(self):
        if self.chart_canvas:
            self.chart_canvas.get_tk_widget().destroy()
            self.chart_canvas = None
        self.image_view_frame.pack(expand=True, fill=tk.BOTH)
        self.controls.pack(pady=20)
        self.show_image()

    def show_chart_mode(self):
        self.image_view_frame.pack_forget()
        self.controls.pack_forget()
        if self.chart_canvas:
            self.chart_canvas.get_tk_widget().destroy()

        if not os.path.exists(self.excel_path):
            self.label_info.config(text="Khong tim thay file Excel!")
            return

        try:
            df = pd.read_excel(self.excel_path)
            fig, ax = plt.subplots(figsize=(9, 5), dpi=100)
            fig.patch.set_facecolor('#2c3e50')
            ax.set_facecolor('#34495e')
            
            if 'Road_Epoch' in df.columns and 'Road_Loss' in df.columns:
                road_data = df[pd.to_numeric(df['Road_Epoch'], errors='coerce').notnull()]
                ax.plot(road_data['Road_Epoch'], road_data['Road_Loss'], label='Road Loss', color='#1abc9c', linewidth=2, marker='o', markersize=4)
            if 'Lane_Epoch' in df.columns and 'Lane_Loss' in df.columns:
                lane_data = df[pd.to_numeric(df['Lane_Epoch'], errors='coerce').notnull()]
                ax.plot(lane_data['Lane_Epoch'], lane_data['Lane_Loss'], label='Lane Loss', color='#e74c3c', linewidth=2, marker='s', markersize=4)

            ax.set_title("Training Loss Progression", color='white', fontsize=14, pad=20)
            ax.set_xlabel("Epoch", color='white')
            ax.set_ylabel("Loss", color='white')
            ax.tick_params(colors='white')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend()

            self.chart_canvas = FigureCanvasTkAgg(fig, master=self.content_frame)
            self.chart_canvas.draw()
            self.chart_canvas.get_tk_widget().pack(pady=10, fill=tk.BOTH, expand=True)
        except Exception as e:
            self.label_info.config(text=f"Loi ve bieu do: {e}")

    def show_image(self):
        if not self.image_list: return
        
        img_name = self.image_list[self.current_index]
        result_path = os.path.join(self.result_dir, img_name)
        original_path = os.path.join(self.original_dir, img_name)

        # Resize display config
        display_width = 580
        display_height = 450

        # 1. Show Original (If exists)
        if os.path.exists(original_path):
            orig_img = Image.open(original_path)
            orig_img.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
            self.photo_orig = ImageTk.PhotoImage(orig_img)
            self.original_label.config(image=self.photo_orig, text="")
        else:
            self.original_label.config(image="", text="ORIGINAL NOT FOUND")

        # 2. Show Result
        res_img = Image.open(result_path)
        res_img.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
        self.photo_res = ImageTk.PhotoImage(res_img)
        self.result_label.config(image=self.photo_res)

        self.label_info.config(text=f"File: {img_name} ({self.current_index + 1}/{len(self.image_list)})")

    def next_image(self):
        if self.image_list:
            self.current_index = (self.current_index + 1) % len(self.image_list)
            self.show_image()

    def prev_image(self):
        if self.image_list:
            self.current_index = (self.current_index - 1) % len(self.image_list)
            self.show_image()

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.configure("TButton", font=("Helvetica", 12), padding=10)
    app = AIResultViewer(root)
    root.mainloop()
