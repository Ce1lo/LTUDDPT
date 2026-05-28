# closure.py - Theo dõi thời gian nhắm mắt liên tục (ngủ gật sâu)
class ContinuousClosureDetector:
    """Theo dõi thời gian nhắm mắt liên tục và phát cảnh báo nếu vượt ngưỡng."""
    
    def __init__(self, threshold=2.5):
        self.threshold = threshold
        self.closed_start_time = None
        self.closed_duration = 0.0

    def update(self, is_closed, current_time):
        """Cập nhật bộ đếm thời gian nhắm mắt liên tục."""
        if is_closed:
            if self.closed_start_time is None:
                self.closed_start_time = current_time
            self.closed_duration = current_time - self.closed_start_time
        else:
            # Chi can mat mo lai thi bo dem nham mat lien tuc ve 0.
            # Khac PERCLOS: module nay chi bat truong hop ngu gat tuc thoi.
            self.closed_start_time = None
            self.closed_duration = 0.0

    def get_closed_duration(self):
        """Lấy thời gian nhắm mắt liên tục hiện tại (giây)."""
        return self.closed_duration

    def has_alert(self):
        """Kiểm tra xem thời gian nhắm mắt liên tục có vượt ngưỡng nguy hiểm."""
        return self.closed_duration >= self.threshold

    def reset(self):
        """Đặt lại bộ đếm thời gian nhắm mắt liên tục."""
        self.closed_start_time = None
        self.closed_duration = 0.0
