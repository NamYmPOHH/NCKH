import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import subprocess

class AIResultViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Result Viewer - Lane & Road Segmentation")
        self.root.geometry("1000x700")
        self.root.configure(bg="#2c3e50")

        # Duong dan thu muc anh ket qua moi
        self.image_dir = "result/result_train"
        self.video_path = "result/result_video/result_video.mp4"
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
        self.dropdown = ttk.Combobox(self.header_frame, textvariable=self.view_mode, state="readonly", width=15, font=("Helvetica", 12))
        self.dropdown['values'] = ("Ảnh (Result)", "Video (Trained)")
        self.dropdown.current(0)
        self.dropdown.pack(side=tk.LEFT, padx=20)
        self.dropdown.bind("<<ComboboxSelected>>", self.on_dropdown_change)

        # Header Title
        self.header = tk.Label(self.header_frame, text="AI SEGMENTATION RESULTS", font=("Helvetica", 20, "bold"), 
                              bg="#2c3e50", fg="#ecf0f1")
        self.header.pack(side=tk.LEFT, padx=150)

        # Image Container
        self.canvas = tk.Canvas(self.root, width=800, height=450, bg="#34495e", highlightthickness=0)
        self.canvas.pack(pady=10)

        # Image Label (inside canvas effectively)
        self.image_label = tk.Label(self.canvas, bg="#34495e")
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")

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

        # Keyboard binding
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())

    def on_dropdown_change(self, event):
        selected = self.view_mode.get()
        if selected == "Video (Trained)":
            if os.path.exists(self.video_path):
                # Mo video bang trinh phat mac dinh cua Windows
                try:
                    os.startfile(os.path.abspath(self.video_path))
                except Exception as e:
                    print(f"Lỗi khi mở video: {e}")
            else:
                self.label_info.config(text=f"Chua co video tai: {self.video_path}. Hay chay predict_video.py truoc.")
            
            # Reset lai dropdown ve Ảnh sau khi an mo Video
            self.dropdown.current(0)

    def show_image(self):
        img_path = os.path.join(self.image_dir, self.image_list[self.current_index])
        
        # Mo va resize anh de vua khung hinh
        img = Image.open(img_path)
        
        # Resize giữ tỉ lệ
        display_width = 800
        display_height = 450
        img.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
        
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.photo)
        
        # Cap nhat thong tin
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
    # Style cho ttk buttons
    style = ttk.Style()
    style.configure("TButton", font=("Helvetica", 12), padding=10)
    
    app = AIResultViewer(root)
    root.mainloop()
