import cv2
import numpy as np
import torch
from ultralytics import FastSAM
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

class FastSAMSegmentation:
    def __init__(self):
        # FastSAM 모델 초기화
        self.device = 'cpu'  # GPU 메모리 문제로 CPU 모드로 변경
        print(f"Using device: {self.device}")
        
        # FastSAM 모델 로드 (자동으로 가중치 다운로드)
        try:
            self.model = FastSAM('FastSAM-s.pt')
            print("FastSAM 모델이 성공적으로 로드되었습니다!")
        except Exception as e:
            print(f"모델 로드 중 오류: {e}")
        
        # 이미지와 클릭 포인트 저장 변수
        self.image = None
        self.click_points = []
        self.image_path = None
        self.original_size = None
        self.scale_factor = 1.0
        self.current_result = None
        
        # GUI 초기화
        self.root = tk.Tk()
        self.root.title("FastSAM 세그멘테이션")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # 상태 라벨
        self.status_label = tk.Label(self.root, text="이미지를 로드해주세요.", 
                                   font=("Arial", 12, "bold"), bg='#f0f0f0')
        self.status_label.pack(pady=5)
        
        # 버튼 생성
        button_frame = tk.Frame(self.root, bg='#f0f0f0')
        button_frame.pack(pady=5)
        
        # 버튼 스타일 설정
        button_style = {
            'font': ("Arial", 10, "bold"),
            'width': 15,
            'height': 2,
            'borderwidth': 0,
            'relief': "flat"
        }
        
        self.load_button = tk.Button(button_frame, text="이미지 로드", command=self.load_image,
                                   bg="#2E7D32", fg="white", activebackground="#1B5E20",
                                   **button_style)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        self.segment_button = tk.Button(button_frame, text="전체 세그멘테이션", 
                                      command=self.run_full_segmentation,
                                      bg="#1565C0", fg="white", activebackground="#0D47A1",
                                      **button_style)
        self.segment_button.pack(side=tk.LEFT, padx=5)
        
        self.point_segment_button = tk.Button(button_frame, text="포인트 세그멘테이션",
                                            command=self.run_point_segmentation,
                                            bg="#6A1B9A", fg="white", activebackground="#4A148C",
                                            **button_style)
        self.point_segment_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = tk.Button(button_frame, text="포인트 초기화",
                                    command=self.clear_points,
                                    bg="#E65100", fg="white", activebackground="#BF360C",
                                    **button_style)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # 캔버스 프레임
        canvas_frame = tk.Frame(self.root, bg='#f0f0f0')
        canvas_frame.pack(pady=10)
        
        # 원본 이미지 캔버스
        self.original_canvas = tk.Canvas(canvas_frame, width=500, height=500, bg="white",
                                       highlightthickness=1, highlightbackground="#BDBDBD")
        self.original_canvas.pack(side=tk.LEFT, padx=10)
        self.original_canvas.bind("<Button-1>", self.on_canvas_click)
        
        # 결과 이미지 캔버스
        self.result_canvas = tk.Canvas(canvas_frame, width=500, height=500, bg="white",
                                     highlightthickness=1, highlightbackground="#BDBDBD")
        self.result_canvas.pack(side=tk.LEFT, padx=10)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_path:
            self.image_path = file_path
            # 이미지 로드 및 크기 제한
            self.image = cv2.imread(file_path)
            if self.image is None:
                messagebox.showerror("오류", "이미지를 로드할 수 없습니다.")
                return
                
            height, width = self.image.shape[:2]
            max_size = 1000  
            if height > max_size or width > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                self.image = cv2.resize(self.image, (new_width, new_height))
                print(f"이미지 크기 조정: {width}x{height} -> {new_width}x{new_height}")
            
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.original_size = (self.image.shape[1], self.image.shape[0])
            self.display_image()
            self.status_label.config(text=f"이미지 로드 완료: {os.path.basename(file_path)}")
            # 결과 캔버스 초기화
            self.result_canvas.delete("all")
            self.current_result = None
            
    def display_image(self):
        if self.image is not None:
            # 이미지 크기 조정
            height, width = self.image.shape[:2]
            canvas_size = 500  # 캔버스 크기
            
            # 스케일 계산
            scale = min(canvas_size/width, canvas_size/height)
            self.scale_factor = scale
            
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            resized_image = cv2.resize(self.image, (new_width, new_height))
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(resized_image))
            
            # 원본 이미지 캔버스에 표시
            self.original_canvas.delete("all")
            self.original_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            
            # 클릭 포인트 표시
            for point in self.click_points:
                x, y = point
                self.original_canvas.create_oval(x-5, y-5, x+5, y+5, fill='red', outline='white', width=2)
                self.original_canvas.create_text(x, y-15, text=f"({x},{y})", fill='red', font=("Arial", 8))
    
    def display_result(self, result_image, title):
        if result_image is not None:
            # 결과 이미지 크기 조정
            height, width = result_image.shape[:2]
            canvas_size = 500
            
            scale = min(canvas_size/width, canvas_size/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            resized_result = cv2.resize(result_image, (new_width, new_height))
            result_photo = ImageTk.PhotoImage(image=Image.fromarray(resized_result))
            
            # 결과 캔버스에 표시
            self.result_canvas.delete("all")
            self.result_canvas.create_image(0, 0, anchor=tk.NW, image=result_photo)
            self.result_canvas.create_text(250, 20, text=title, font=("Arial", 12, "bold"))
            self.current_result = result_photo  # 참조 유지
    
    def on_canvas_click(self, event):
        if self.image is not None:
            self.click_points.append((event.x, event.y))
            self.display_image()
            self.status_label.config(text=f"클릭 포인트: {len(self.click_points)}개 추가됨")
    
    def run_full_segmentation(self):
        if self.image is None:
            messagebox.showwarning("경고", "먼저 이미지를 로드해주세요!")
            return
        
        try:
            self.status_label.config(text="전체 세그멘테이션 진행 중...")
            self.root.update()
            
            results = self.model(self.image_path, device=self.device, retina_masks=True, 
                               imgsz=640, conf=0.4, iou=0.9)
            
            if results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                
                # 결과 이미지 생성
                result_image = self.image.copy()
                colored_mask = np.zeros_like(result_image)
                
                # HSV 색상 생성 함수
                def get_color(idx, total):
                    hue = idx / total  # 0~1 사이의 값
                    saturation = 0.8  # 채도
                    value = 0.9  # 명도
                    
                    # HSV를 RGB로 변환
                    h = int(hue * 179)  # OpenCV HSV는 H: 0-179, S: 0-255, V: 0-255
                    s = int(saturation * 255)
                    v = int(value * 255)
                    
                    # HSV 색상을 BGR로 변환
                    hsv = np.uint8([[[h, s, v]]])
                    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
                    return bgr
                
                # 각 마스크에 다른 색상 적용
                for i, mask in enumerate(masks):
                    # 마스크를 원본 이미지 크기로 리사이즈
                    resized_mask = cv2.resize(mask.astype(np.float32), 
                                            (self.image.shape[1], self.image.shape[0]))
                    
                    # 객체별 색상 생성
                    color = get_color(i, len(masks))
                    
                    # 마스크에 색상 적용
                    mask_colored = np.zeros_like(result_image)
                    mask_colored[resized_mask > 0.5] = color
                    
                    # 알파 블렌딩으로 합성
                    alpha = 0.5
                    colored_mask = cv2.addWeighted(colored_mask, 1, mask_colored, alpha, 0)
                
                # 원본 이미지와 컬러 마스크 합성
                result_image = cv2.addWeighted(result_image, 0.7, colored_mask, 0.3, 0)
                
                # 결과 표시
                self.display_result(result_image, f"Full Segmentation ({len(masks)} objects)")
                self.status_label.config(text="전체 세그멘테이션 완료!")
            else:
                self.display_result(self.image, "No Segmentation Results")
                self.status_label.config(text="세그멘테이션 결과 없음")
            
        except Exception as e:
            messagebox.showerror("오류", f"세그멘테이션 중 오류가 발생했습니다: {str(e)}")
            self.status_label.config(text="세그멘테이션 실패")
    
    def run_point_segmentation(self):
        if self.image is None:
            messagebox.showwarning("경고", "먼저 이미지를 로드해주세요!")
            return
            
        if len(self.click_points) == 0:
            messagebox.showwarning("경고", "세그멘테이션할 포인트를 클릭해주세요!")
            return
        
        try:
            self.status_label.config(text="포인트 기반 세그멘테이션 진행 중...")
            self.root.update()
            
            results = self.model(self.image_path, device=self.device, retina_masks=True, 
                               imgsz=640, conf=0.4, iou=0.9)
            
            if results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                
                # 클릭 포인트와 가장 가까운 마스크 찾기
                best_masks = []
                for canvas_x, canvas_y in self.click_points:
                    original_x = int(canvas_x / self.scale_factor)
                    original_y = int(canvas_y / self.scale_factor)
                    
                    # 마스크 크기에 맞게 좌표 조정
                    mask_h, mask_w = masks[0].shape
                    img_h, img_w = self.image.shape[:2]
                    
                    mask_x = int(original_x * mask_w / img_w)
                    mask_y = int(original_y * mask_h / img_h)
                    
                    for i, mask in enumerate(masks):
                        if mask_y < mask.shape[0] and mask_x < mask.shape[1]:
                            if mask[mask_y, mask_x] > 0.5:
                                # 마스크를 원본 이미지 크기로 리사이즈
                                resized_mask = cv2.resize(mask.astype(np.float32), 
                                                        (self.image.shape[1], self.image.shape[0]))
                                best_masks.append(resized_mask)
                                break
                
                if best_masks:
                    # 결과 이미지 생성
                    result_image = self.image.copy()
                    combined_mask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.float32)
                    
                    # 선택된 마스크 합성
                    for mask in best_masks:
                        combined_mask = np.maximum(combined_mask, mask)
                    
                    # 마스크를 컬러로 변환
                    mask_colored = np.zeros_like(result_image)
                    mask_colored[..., 0] = (combined_mask > 0) * 255  # 빨간색 채널
                    
                    # 원본 이미지와 마스크 합성
                    alpha = 0.5
                    result_image = cv2.addWeighted(result_image, 1, mask_colored, alpha, 0)
                    
                    # 클릭 포인트 표시
                    for canvas_x, canvas_y in self.click_points:
                        original_x = int(canvas_x / self.scale_factor)
                        original_y = int(canvas_y / self.scale_factor)
                        cv2.circle(result_image, (original_x, original_y), 5, (0, 0, 255), -1)
                    
                    self.display_result(result_image, f"Point-based Segmentation ({len(best_masks)} objects)")
                else:
                    self.display_result(self.image, "No Objects Found at Click Points")
                
                self.status_label.config(text="포인트 기반 세그멘테이션 완료!")
            else:
                self.display_result(self.image, "No Segmentation Results")
                self.status_label.config(text="세그멘테이션 결과 없음")
            
        except Exception as e:
            messagebox.showerror("오류", f"세그멘테이션 중 오류가 발생했습니다: {str(e)}")
            self.status_label.config(text="세그멘테이션 실패")
            print(f"상세 오류: {e}")
    
    def clear_points(self):
        self.click_points = []
        self.display_image()
        self.status_label.config(text="클릭 포인트가 초기화되었습니다.")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = FastSAMSegmentation()
    app.run() 