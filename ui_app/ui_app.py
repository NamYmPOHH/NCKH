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
        self.root.geometry("1100x800")
        self.root.configure(bg="#2c3e50")

        # Duong dan thu muc anh ket qua moi
        self.image_dir = "result/result_train"
        self.video_path = "result/result_video/result_video.mp4"
        self.excel_path = "excel/Training_Summary.xlsx"
        
        self.image_list = []
        self.current_index = 0

        # Load danh sach anh
        if os.path.exists(self.image_dir):
            self.image_list = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.image_list.sort()

        # UI Components
        self.create_widgets()
        
        if self.image_list:
            self.show_image()
        else:
            self.label_info.config(text="Khong tim thay anh trong: " + self.image_dir)

    def create_widgets(self):
        # Header Frame de chua title va dropdown
        self.header_frame = tk.Frame(self.root, bg="#2c3e50")
        self.header_frame.pack(fill=tk.X, pady=10)

        # Dropdown ở góc trên bên trái
        self.view_mode = tk.StringVar()
        self.dropdown = ttk.Combobox(self.header_frame, textvariable=self.view_mode, state="readonly", width=18, font=("Helvetica", 12))
        self.dropdown['values'] = ("Ảnh (Result)", "Video (Trained)", "Biểu đồ (Chart)")
        self.dropdown.current(0)
        self.dropdown.pack(side=tk.LEFT, padx=20)
        self.dropdown.bind("<<ComboboxSelected>>", self.on_dropdown_change)

        # Header Title
        self.header = tk.Label(self.header_frame, text="AI SEGMENTATION RESULTS", font=("Helvetica", 20, "bold"), 
                              bg="#2c3e50", fg="#ecf0f1")
        self.header.pack(side=tk.LEFT, expand=True)

        # Main Content Area (de hoán đổi giữa ảnh và biểu đồ)
        self.content_frame = tk.Frame(self.root, bg="#2c3e50")
        self.content_frame.pack(expand=True, fill=tk.BOTH)

        # 1. Image View (Canvas)
        self.canvas = tk.Canvas(self.content_frame, width=800, height=450, bg="#34495e", highlightthickness=0)
        self.canvas.pack(pady=10)
        self.image_label = tk.Label(self.canvas, bg="#34495e")
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")

        # 2. Chart View (Sẽ được tạo khi chọn mode Chart)
        self.chart_canvas = None

        # Info Label
        self.label_info = tk.Label(self.root, text="", font=("Helvetica", 12), bg="#2c3e50", fg="#bdc3c7")
        self.label_info.pack(pady=5)

        # Controls Frame (An di khi xem bieu do)
        self.controls = tk.Frame(self.root, bg="#2c3e50")
        self.controls.pack(pady=20)

        self.btn_prev = ttk.Button(self.controls, text="<< TRÁI", command=self.prev_image)
        self.btn_prev.grid(row=0, column=0, padx=20)

        self.btn_next = ttk.Button(self.controls, text="PHẢI >>", command=self.next_image)
        self.btn_next.grid(row=0, column=1, padx=20)

        # Keyboard binding
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())

    def on_dropdown_change(self, event):
        selected = self.view_mode.get()
        
        if selected == "Video (Trained)":
            if os.path.exists(self.video_path):
                try:
                    os.startfile(os.path.abspath(self.video_path))
                except Exception as e:
                    print(f"Loi khi mo video: {e}")
            else:
                self.label_info.config(text=f"Chua co video tai: {self.video_path}")
            self.dropdown.current(0) # Quay lai mode Anh
            self.show_image_mode()
            
        elif selected == "Biểu đồ (Chart)":
            self.show_chart_mode()
            
        else: # Anh (Result)
            self.show_image_mode()

    def show_image_mode(self):
        if self.chart_canvas:
            self.chart_canvas.get_tk_widget().destroy()
            self.chart_canvas = None
        self.canvas.pack(pady=10)
        self.controls.pack(pady=20)
        self.show_image()

    def show_chart_mode(self):
        # An phan xem anh
        self.canvas.pack_forget()
        self.controls.pack_forget()
        
        if self.chart_canvas:
            self.chart_canvas.get_tk_widget().destroy()

        if not os.path.exists(self.excel_path):
            self.label_info.config(text="Khong tim thay file Excel de ve bieu do!")
            return

        try:
            df = pd.read_excel(self.excel_path)
            
            # Tao figure matplotlib
            fig, ax = plt.subplots(figsize=(9, 5), dpi=100)
            fig.patch.set_facecolor('#2c3e50')
            ax.set_facecolor('#34495e')
            
            # Lay du lieu Road va Lane (xu ly truong hop NaN do Epoch khac nhau)
            # Gia su cot la Road_Epoch, Road_Loss, Lane_Epoch, Lane_Loss
            if 'Road_Epoch' in df.columns and 'Road_Loss' in df.columns:
                # Loc bo dong MIN LOSS cuoi cung truoc khi ve
                road_data = df[pd.to_numeric(df['Road_Epoch'], errors='coerce').notnull()]
                ax.plot(road_data['Road_Epoch'], road_data['Road_Loss'], label='Road Loss', color='#1abc9c', linewidth=2, marker='o', markersize=4)
                
            if 'Lane_Epoch' in df.columns and 'Lane_Loss' in df.columns:
                lane_data = df[pd.to_numeric(df['Lane_Epoch'], errors='coerce').notnull()]
                ax.plot(lane_data['Lane_Epoch'], lane_data['Lane_Loss'], label='Lane Loss', color='#e74c3c', linewidth=2, marker='s', markersize=4)

            ax.set_title("Training Loss Progression", color='white', fontsize=14, pad=20)
            ax.set_xlabel("Epoch", color='white')
            ax.set_ylabel("Loss (Lower is Better)", color='white')
            ax.tick_params(colors='white')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend()

            # Nhung vao Tkinter
            self.chart_canvas = FigureCanvasTkAgg(fig, master=self.content_frame)
            self.chart_canvas.draw()
            self.chart_canvas.get_tk_widget().pack(pady=10, fill=tk.BOTH, expand=True)
            
            self.label_info.config(text="Bieu do hien thi Loss giam dan qua cac Epoch.")
            
        except Exception as e:
            self.label_info.config(text=f"Loi khi ve bieu do: {e}")

    def show_image(self):
        if not self.image_list: return
        img_path = os.path.join(self.image_dir, self.image_list[self.current_index])
        img = Image.open(img_path)
        display_width = 800
        display_height = 450
        img.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.photo)
        self.label_info.config(text=f"Anh {self.current_index + 1}/{len(self.image_list)}: {self.image_list[self.current_index]}")

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
