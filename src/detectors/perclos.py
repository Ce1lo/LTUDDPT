# perclos.py - Tinh PERCLOS tren cua so thoi gian truot.
from collections import deque


class PerclosDetector:
    """Theo doi ty le thoi gian mat o trang thai nham trong mot khoang gan nhat."""

    def __init__(self, window_seconds=60.0, threshold=0.4, min_observation_seconds=10.0):
        self.window_seconds = window_seconds
        self.threshold = threshold
        self.min_observation_seconds = min_observation_seconds
        self.samples = deque()

    def update(self, is_closed, current_time):
        """Luu trang thai mat hien tai va xoa cac mau da nam ngoai cua so thoi gian."""
        self.samples.append((current_time, is_closed))
        self._clear_old_samples(current_time)

    def get_perclos(self):
        """Tra ve ty le thoi gian nham mat trong cua so thoi gian hien tai."""
        if len(self.samples) < 2:
            return 0.0

        closed_time = 0.0
        total_time = 0.0

        for index in range(1, len(self.samples)):
            previous_time, previous_closed = self.samples[index - 1]
            current_time, _ = self.samples[index]
            elapsed = current_time - previous_time

            if elapsed <= 0:
                continue

            total_time += elapsed
            if previous_closed:
                closed_time += elapsed

        if total_time == 0:
            return 0.0

        # Diem GV hay hoi: PERCLOS = tong thoi gian mat nham / tong thoi gian quan sat.
        # Chi so nay on dinh hon viec chi bat mot lan nham mat dai, vi no do met moi tich luy.
        return closed_time / total_time

    def has_alert(self):
        """Kiem tra PERCLOS co vuot nguong canh bao buon ngu hay khong."""
        if self.get_observation_duration() < self.min_observation_seconds:
            return False
        return self.get_perclos() >= self.threshold

    def get_observation_duration(self):
        """Lay thoi gian da quan sat trong cua so hien tai."""
        if len(self.samples) < 2:
            return 0.0
        return max(0.0, self.samples[-1][0] - self.samples[0][0])

    def reset(self):
        """Xoa toan bo lich su mau."""
        self.samples.clear()

    def _clear_old_samples(self, current_time):
        while self.samples and current_time - self.samples[0][0] > self.window_seconds:
            self.samples.popleft()
