import tkinter as tk
from tkinter import messagebox
from PIL import ImageGrab, Image, ImageOps
import numpy as np
import tensorflow as tf

import sys
import os

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

class DrawingApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        root.title("손글씨 숫자 예측")
        root.resizable(width=False, height=False)

        # Canvas 생성 및 마우스 이벤트 연결
        self.canvas = tk.Canvas(root, bg="white", width=280, height=280)
        self.canvas.grid(row=0, column=0, columnspan=2, sticky='nsew')
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)

        # 예측하기 버튼 생성
        predict_button = tk.Button(root, text="예측하기", command=self.predict_number)
        predict_button.grid(row=1, column=0, sticky='nsew')

        # 초기화
        self.last_x = self.last_y = None
        
        # 캔버스 초기화 버튼 생성
        clear_button = tk.Button(root, text="초기화", command=self.clear_canvas)
        clear_button.grid(row=1, column=1, sticky='nsew')

        # 모델 불러오기
        self.model = tf.keras.models.load_model(self.model)

    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_oval((self.last_x, self.last_y, event.x, event.y), fill="black", width=20)
        self.last_x, self.last_y = event.x, event.y

    def is_canvas_empty(self):
        return len(self.canvas.find_all()) == 0

    def predict_number(self):
        if self.is_canvas_empty():
            messagebox.showinfo("경고", "캔버스가 비어 있습니다.")
            return

        # Canvas 영역의 좌표를 얻음
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        # 캔버스 영역 캡처
        img = ImageGrab.grab().crop((x, y, x1, y1))
        img = img.convert("L")
        img = ImageOps.invert(img)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

        # 이미지를 모델에 입력할 수 있는 형식으로 변환
        img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0

        # 숫자 예측
        prediction = self.model.predict(img_array, verbose=False)
        predicted_number = np.argmax(prediction)

        # 결과 출력
        messagebox.showinfo("예측 결과", f"예측된 숫자는 {predicted_number}입니다.")
        
        self.clear_canvas()
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.last_x = self.last_y = None

# 애플리케이션 실행
if __name__ == "__main__":
    root = tk.Tk()
    model = resource_path("my_model.keras")
    app = DrawingApp(root, model)
    root.mainloop()
    
