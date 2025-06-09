import sys
import torch
import requests
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                            QFileDialog, QMessageBox, QComboBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image

class InferenceThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, model, processor, image_path, prompt, task_type="query"):
        super().__init__()
        self.model = model
        self.processor = processor
        self.image_path = image_path
        self.prompt = prompt
        self.task_type = task_type
        
    def run(self):
        try:
            # 이미지 로드
            image = Image.open(self.image_path).convert('RGB')
            
            # 이미지 크기 조정 (비율 유지하면서 1000 이하로)
            max_size = 1000
            width, height = image.size
            
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 작업 타입에 따른 처리
            if self.task_type == "caption":
                # 이미지 캡셔닝 - 더 상세한 설명을 위한 질문
                inputs = self.processor(image, "Describe this image in detail with all visible objects, colors, and activities", return_tensors="pt")
                
                # 추론 실행 - 더 긴 텍스트를 위한 파라미터 조정
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_length=100,
                        num_beams=5,
                        min_length=20,
                        length_penalty=1.0
                    )
                
                # 결과 디코딩
                result = self.processor.decode(out[0], skip_special_tokens=True)
                
            else:  # VQA
                # VQA 처리
                inputs = self.processor(image, self.prompt, return_tensors="pt")
                
                # 추론 실행
                with torch.no_grad():
                    out = self.model.generate(**inputs)
                
                # 결과 디코딩
                result = self.processor.decode(out[0], skip_special_tokens=True)
            
            # 메모리 정리
            torch.cuda.empty_cache()
            
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class LightweightVLApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BLIP-VQA - Vision Language Model")
        self.setGeometry(100, 100, 1400, 900)
        
        # UI 초기화 먼저
        self.init_ui()
        
        # 모델 초기화
        self.init_model()
        
    def init_model(self):
        try:
            self.status_label.setText("BLIP-VQA 모델 로딩 중...")
            QApplication.processEvents()
            
            # BLIP-VQA 모델 로드
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
            
            self.status_label.setText("모델 로드 완료 (BLIP-VQA-base)")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"모델 로드 실패: {str(e)}")
            self.status_label.setText("모델 로드 실패")
            
    def init_ui(self):
        # 메인 위젯과 레이아웃
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout()
        
        # 왼쪽 패널 (이미지 표시)
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # 이미지 표시 레이블
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(500, 400)
        self.image_label.setStyleSheet("border: 2px solid #ccc; background-color: #f5f5f5;")
        self.image_label.setText("Load an image to get started")
        left_layout.addWidget(self.image_label)
        
        # 이미지 로드 버튼
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        self.load_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 10px; font-size: 14px; }")
        left_layout.addWidget(self.load_button)
        
        # 상태 표시 레이블
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        left_layout.addWidget(self.status_label)
        
        left_panel.setLayout(left_layout)
        layout.addWidget(left_panel)
        
        # 오른쪽 패널 (컨트롤 및 결과)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        # 작업 타입 선택
        task_label = QLabel("Task Type:")
        task_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        right_layout.addWidget(task_label)
        
        self.task_combo = QComboBox()
        self.task_combo.addItems([
            "Caption - Generate image description",
            "Query - Ask questions about image (VQA)"
        ])
        self.task_combo.currentTextChanged.connect(self.on_task_changed)
        self.task_combo.setStyleSheet("font-size: 12px; padding: 5px;")
        right_layout.addWidget(self.task_combo)
        
        # 빠른 질문 선택
        quick_label = QLabel("Quick Questions:")
        quick_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        right_layout.addWidget(quick_label)
        
        self.quick_combo = QComboBox()
        self.update_quick_questions()
        self.quick_combo.currentTextChanged.connect(self.on_quick_question_changed)
        self.quick_combo.setStyleSheet("font-size: 12px; padding: 5px;")
        right_layout.addWidget(self.quick_combo)
        
        # 프롬프트 입력
        prompt_label = QLabel("Input (for Query/Detect/Point tasks):")
        prompt_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        right_layout.addWidget(prompt_label)
        
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Enter your question or object to find...")
        self.prompt_input.setMaximumHeight(80)
        self.prompt_input.setText("What do you see in this image?")
        right_layout.addWidget(self.prompt_input)
        
        # 실행 버튼
        self.run_button = QPushButton("Run Analysis")
        self.run_button.clicked.connect(self.run_inference)
        self.run_button.setStyleSheet("QPushButton { background-color: #2196F3; color: white; padding: 12px; font-size: 14px; font-weight: bold; }")
        right_layout.addWidget(self.run_button)
        
        # 결과 표시
        result_label = QLabel("Results:")
        result_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        right_layout.addWidget(result_label)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("background-color: #f9f9f9; border: 1px solid #ddd; font-size: 13px;")
        right_layout.addWidget(self.result_text)
        
        right_panel.setLayout(right_layout)
        layout.addWidget(right_panel)
        
        main_widget.setLayout(layout)
        
        # 초기 이미지 설정
        self.current_image = None
        
    def update_quick_questions(self):
        task = self.task_combo.currentText().split(" - ")[0].lower()
        
        if task == "query":
            items = [
                "What do you see in this image?",
                "Describe this image in detail",
                "What are the main objects in this image?",
                "What colors are prominent?",
                "Are there any people in this image?",
                "What is the setting or location?",
                "What is happening in this scene?",
                "How many objects are there?",
                "Custom (type below)"
            ]
        else:  # caption
            items = ["Generate Caption (no input needed)"]
            
        self.quick_combo.clear()
        self.quick_combo.addItems(items)
        
    def on_task_changed(self):
        self.update_quick_questions()
        task = self.task_combo.currentText().split(" - ")[0].lower()
        
        if task == "caption":
            self.prompt_input.setText("")
            self.prompt_input.setEnabled(False)
            self.prompt_input.setPlaceholderText("캡셔닝 모드에서는 텍스트 입력이 필요하지 않습니다")
        else:
            self.prompt_input.setEnabled(True)
            self.prompt_input.setPlaceholderText("질문을 입력하세요...")
            self.prompt_input.setText("What do you see in this image?")
        
    def on_quick_question_changed(self, text):
        if text != "Custom (type below)" and text != "Generate Caption (no input needed)":
            self.prompt_input.setText(text)
        elif text == "Custom (type below)":
            self.prompt_input.clear()
            self.prompt_input.setFocus()
        
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        
        if file_name:
            # 이미지 로드 및 표시
            pixmap = QPixmap(file_name)
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            
            # 현재 이미지 저장
            self.current_image = file_name
            self.status_label.setText("Image loaded successfully")
    
    def run_inference(self):
        if not hasattr(self, 'model') or self.model is None:
            QMessageBox.warning(self, "Warning", "Model not loaded yet.")
            return
            
        if not self.current_image:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return
            
        # 작업 타입 확인
        task = self.task_combo.currentText().split(" - ")[0].lower()
        prompt = self.prompt_input.toPlainText().strip()
        
        if task != "caption" and not prompt:
            QMessageBox.warning(self, "Warning", "Please enter a question.")
            return
            
        # UI 상태 업데이트
        self.run_button.setEnabled(False)
        self.status_label.setText(f"Running {task}...")
        self.result_text.setText("Processing... Please wait...")
        QApplication.processEvents()
        
        # 백그라운드 스레드에서 추론 실행
        self.inference_thread = InferenceThread(
            self.model, self.processor, self.current_image, prompt, task
        )
        self.inference_thread.finished.connect(self.on_inference_finished)
        self.inference_thread.error.connect(self.on_inference_error)
        self.inference_thread.start()
    
    def on_inference_finished(self, result):
        self.result_text.setText(result)
        self.status_label.setText("Analysis completed")
        self.run_button.setEnabled(True)
    
    def on_inference_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"Analysis failed: {error_msg}")
        self.result_text.setText("An error occurred during analysis.")
        self.status_label.setText("Error occurred")
        self.run_button.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LightweightVLApp()
    window.show()
    sys.exit(app.exec_()) 