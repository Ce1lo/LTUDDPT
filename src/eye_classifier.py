# classifier.py - Bộ phân loại mắt đóng/mở bằng mô hình AI học sâu CNN
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

class EyeClassifier:
    """Tải mô hình Keras và dự đoán trạng thái nhắm/mở mắt."""
    
    def __init__(self):
        src_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(src_dir)
        
        model_path = os.path.join(root_dir, "models", "keras_model.h5")
        labels_path = os.path.join(root_dir, "models", "labels.txt")

        self.model = load_model(model_path, compile=False)

        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = [line.strip() for line in f.readlines()]

    def predict(self, eye_crop):
        """Dự đoán trạng thái nhắm hay mở mắt từ ảnh vùng mắt được cắt."""
        resized = cv2.resize(eye_crop, (224, 224))
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Teachable Machine/Keras model thuong nhan anh 224x224 va gia tri pixel [-1, 1].
        # Neu bo buoc normalize nay, do tin cay cua model se sai lech ro.
        normalized = (rgb_img.astype(np.float32) / 127.5) - 1.0
        batch = np.expand_dims(normalized, axis=0)

        predictions = self.model.predict(batch, verbose=0)
        max_idx = np.argmax(predictions)

        return self.labels[max_idx], predictions[0][max_idx]
