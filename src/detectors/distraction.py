# distraction.py - Phat hien mat tap trung khi khuon mat roi khoi khung hinh.


class DistractionDetector:
    """Theo doi trang thai co/khong co khuon mat de phat canh bao mat tap trung."""

    def __init__(self, threshold=2.0):
        self.threshold = threshold
        self.lost_time = None
        self.is_distracted = False

    def update(self, face_detected, current_time):
        """Cap nhat trang thai nhan dien khuon mat va thoi gian bi mat dau."""
        if face_detected:
            self.lost_time = None
            self.is_distracted = False
        else:
            if self.lost_time is None:
                self.lost_time = current_time

            elapsed = current_time - self.lost_time
            # Neu mat khuon mat qua nguong, he thong xem nhu nguoi lai roi khoi vung quan sat.
            # Day chua phai gaze tracking; khi bao cao nen goi la "mat dau khuon mat".
            if elapsed >= self.threshold:
                self.is_distracted = True

    def has_alert(self):
        """Kiem tra co dang canh bao mat tap trung hay khong."""
        return self.is_distracted

    def get_lost_duration(self, current_time):
        """Lay thoi gian mat dau khuon mat hien tai."""
        if self.lost_time is None:
            return 0.0
        return current_time - self.lost_time
