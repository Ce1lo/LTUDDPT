# calibration.py - Tự động hiệu chuẩn ngưỡng EAR động
class EARCalibrator:
    """Tự động hiệu chuẩn ngưỡng EAR của tài xế trong vài giây đầu."""
    
    def __init__(self, duration=5.0, default_threshold=0.20, ratio=0.60):
        self.duration = duration
        self.default_threshold = default_threshold
        self.ratio = ratio
        
        self.start_time = None
        self.ear_history = []
        self.calibrated_threshold = None

    def update(self, ear, current_time):
        """Lưu trữ giá trị EAR mở mắt bình thường và tính ngưỡng đóng mắt khi hết thời gian."""
        if self.calibrated_threshold is not None:
            return

        if self.start_time is None:
            self.start_time = current_time

        elapsed = current_time - self.start_time
        
        if elapsed < self.duration:
            self.ear_history.append(ear)
        else:
            if self.ear_history:
                avg_ear = sum(self.ear_history) / len(self.ear_history)
                # nguong EAR khong co dinh tuyet doi.
                # He thong lay EAR trung binh luc mat mo va nhan ratio de ca nhan hoa nguong.
                # Ngưỡng động nằm trong khoảng an toàn [0.12, 0.28]
                self.calibrated_threshold = max(0.12, min(avg_ear * self.ratio, 0.28))
            else:
                self.calibrated_threshold = self.default_threshold

    def is_complete(self):
        """Kiểm tra xem hiệu chuẩn đã hoàn thành chưa."""
        return self.calibrated_threshold is not None

    def get_threshold(self):
        """Lấy ngưỡng EAR sau khi hiệu chuẩn (hoặc ngưỡng mặc định nếu chưa xong)."""
        if self.is_complete():
            return self.calibrated_threshold
        return self.default_threshold

    def get_progress(self, current_time):
        """Lấy phần trăm tiến trình hiệu chuẩn (0.0 - 100.0)."""
        if self.is_complete():
            return 100.0
        if self.start_time is None:
            return 0.0
        
        elapsed = current_time - self.start_time
        return min((elapsed / self.duration) * 100.0, 100.0)

    def reset(self):
        """Đặt lại bộ cân bằng để có thể đo EAR theshold."""
        self.start_time = None
        self.ear_history = []
        self.calibrated_threshold = None
