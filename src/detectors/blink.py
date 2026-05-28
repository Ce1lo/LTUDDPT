# blink.py - Blink state filtering and blink-rate tracking.
from collections import deque


class BlinkDetector:
    def __init__(
        self,
        window_seconds=10,
        blink_threshold=10,
        debounce_frames=3,
        min_blink_duration=0.08,
        max_blink_duration=0.45,
        refractory_seconds=0.10,
    ):
        self.window_seconds = window_seconds
        self.blink_threshold = blink_threshold
        self.debounce_frames = debounce_frames
        self.min_blink_duration = min_blink_duration
        self.max_blink_duration = max_blink_duration
        self.refractory_seconds = refractory_seconds

        self.is_closed_confirmed = False
        self.is_closed_candidate = False
        self.candidate_frames = 0

        self.closed_start_time = None
        self.last_blink_time = None

        self.total_blinks = 0
        self.blink_timestamps = deque()
        self.has_rate_alert = False

    def update(self, is_closed_instant, current_time):
        self._clear_old_history(current_time)

        if is_closed_instant == self.is_closed_confirmed:
            self.candidate_frames = 0
            self.is_closed_candidate = self.is_closed_confirmed
        else:
            if is_closed_instant != self.is_closed_candidate:
                self.is_closed_candidate = is_closed_instant
                self.candidate_frames = 1
            else:
                self.candidate_frames += 1

            if self.candidate_frames >= self.debounce_frames:
                previous_state = self.is_closed_confirmed
                new_state = self.is_closed_candidate

                self.is_closed_confirmed = new_state
                self.candidate_frames = 0

                if (not previous_state) and new_state:
                    self.closed_start_time = current_time

                if previous_state and (not new_state):
                    blink_duration = None
                    if self.closed_start_time is not None:
                        blink_duration = current_time - self.closed_start_time

                    # Loc blink theo thoi luong hop le:
                    # qua ngan => rung, qua dai => closure dai han (khong phai blink thuong).
                    is_duration_valid = (
                        blink_duration is not None
                        and self.min_blink_duration <= blink_duration <= self.max_blink_duration
                    )
                    # Refractory de tranh dem 2 lan cho cung mot blink do dao dong tin hieu.
                    is_outside_refractory = (
                        self.last_blink_time is None
                        or (current_time - self.last_blink_time) > self.refractory_seconds
                    )

                    if is_duration_valid and is_outside_refractory:
                        self.total_blinks += 1
                        self.blink_timestamps.append(current_time)
                        self.last_blink_time = current_time

                    self.closed_start_time = None

        self.has_rate_alert = len(self.blink_timestamps) >= self.blink_threshold

    def get_total_blinks(self):
        return self.total_blinks

    def get_blink_rate(self, current_time):
        self._clear_old_history(current_time)
        return len(self.blink_timestamps)

    def has_alert(self):
        return self.has_rate_alert

    def is_closed(self):
        return self.is_closed_confirmed

    def _clear_old_history(self, current_time):
        while self.blink_timestamps and (current_time - self.blink_timestamps[0] > self.window_seconds):
            self.blink_timestamps.popleft()

    def reset_temp_state(self):
        self.candidate_frames = 0
        self.is_closed_candidate = self.is_closed_confirmed
        self.closed_start_time = None

    def reset(self):
        self.is_closed_confirmed = False
        self.is_closed_candidate = False
        self.candidate_frames = 0
        self.closed_start_time = None
        self.last_blink_time = None
        self.total_blinks = 0
        self.blink_timestamps.clear()
        self.has_rate_alert = False
