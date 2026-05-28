# alarm.py - Quản lý phát âm thanh cảnh báo còi
import os
import pygame

pygame.mixer.init()

src_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(src_dir)
alarm_path = os.path.join(root_dir, "assets", "haruharu.mp3")

alarm_sound = pygame.mixer.Sound(alarm_path)
is_playing = False

def play_alarm():
    """Bắt đầu phát còi báo động lặp vô hạn."""
    global is_playing
    if not is_playing:
        alarm_sound.play(-1)
        is_playing = True

def stop_alarm():
    """Tắt còi báo động."""
    global is_playing
    if is_playing:
        alarm_sound.stop()
        is_playing = False
