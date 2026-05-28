import time
import sys
import cv2
import mediapipe as mp

from config import (
    DEFAULT_EAR_THRESHOLD,
    DROWSY_CLOSED_DURATION,
    PERCLOS_WINDOW,
    PERCLOS_THRESHOLD,
    PERCLOS_MIN_OBSERVATION_TIME,
    CNN_CLOSED_CONFIDENCE_THRESHOLD,
    EAR_SMOOTHING_ALPHA,
    BLINK_RATE_WINDOW,
    BLINK_RATE_THRESHOLD,
    CALIBRATION_DURATION,
    CALIBRATION_RATIO,
    FACE_LOSS_THRESHOLD,
    BLINK_DEBOUNCE_FRAMES,
    BLINK_MIN_DURATION,
    BLINK_MAX_DURATION,
    BLINK_REFRACTORY,
    BLINK_EAR_MARGIN,
)

from alarm_controller import play_alarm, stop_alarm
from eye_classifier import EyeClassifier
from eye_metrics import calculate_ear

from detectors.ear_calibration import EARCalibrator
from detectors.distraction import DistractionDetector
from detectors.blink import BlinkDetector
from detectors.eye_closure import ContinuousClosureDetector
from detectors.perclos import PerclosDetector

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
# 6 diem moc nay duoc chon theo cong thuc EAR: 2 diem khoe mat va 4 diem mi tren/duoi.
LEFT_EYE_WINDOW = "Left Eye"
RIGHT_EYE_WINDOW = "Right Eye"
EYE_PREVIEW_MIN_SCALE = 1.0
EYE_PREVIEW_MAX_SCALE = 5.0
EYE_PREVIEW_SCALE_STEP = 0.5


def get_eye_coordinates(frame, landmarks, eye_indices):
    """Trích xuất tọa độ pixel của mắt từ Landmarks trên khuôn mặt."""
    h, w, _ = frame.shape
    coords = []
    for idx in eye_indices:
        pt = landmarks[idx]
        coords.append((int(pt.x * w), int(pt.y * h)))
    return coords


def draw_eye_landmarks(frame, eye_coords):
    """Vẽ điểm mốc mắt lên khung hình."""
    for x, y in eye_coords:
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

def destroy_eye_preview_windows():
    for window_name in (LEFT_EYE_WINDOW, RIGHT_EYE_WINDOW):
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass


def show_eye_preview(window_name, eye_crop, scale):
    if eye_crop is None or eye_crop.size == 0:
        return

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if scale != 1.0:
        eye_crop = cv2.resize(
            eye_crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR,
        )
    cv2.imshow(window_name, eye_crop)


def crop_eye_region(frame, landmarks, eye_indices, padding=25):
    """Cắt vùng ảnh mắt từ khung hình để chuẩn bị đưa vào mô hình AI phân loại."""
    h, w, _ = frame.shape
    xs = [int(landmarks[idx].x * w) for idx in eye_indices]
    ys = [int(landmarks[idx].y * h) for idx in eye_indices]

    x_min = max(min(xs) - padding, 0)
    x_max = min(max(xs) + padding, w)
    y_min = max(min(ys) - padding, 0)
    y_max = min(max(ys) + padding, h)

    return frame[y_min:y_max, x_min:x_max]


def predict_eye_state(classifier, frame, landmarks):
    """Sử dụng mô hình AI CNN dự đoán trạng thái nhắm/mở mắt của cả hai bên."""
    left_eye_crop = crop_eye_region(frame, landmarks, LEFT_EYE_INDICES)
    right_eye_crop = crop_eye_region(frame, landmarks, RIGHT_EYE_INDICES)

    if left_eye_crop.size == 0 or right_eye_crop.size == 0:
        return "Unknown", 0.0, left_eye_crop, right_eye_crop

    label_l, conf_l = classifier.predict(left_eye_crop)
    label_r, conf_r = classifier.predict(right_eye_crop)

    # labels.txt cua Teachable Machine co the la "closed_eyes", "open_eyes".
    # Chuan hoa chu thuong de tranh sai do khac biet hoa/thuong.
    label_l_normalized = label_l.lower()
    label_r_normalized = label_r.lower()

    is_closed_l = "closed" in label_l_normalized
    is_closed_r = "closed" in label_r_normalized
    is_open_l = "open" in label_l_normalized
    is_open_r = "open" in label_r_normalized

    # Cach pho bien la doi trang thai dong/mo dong bo hai mat:
    # closed khi ca hai mat closed, open khi ca hai mat open.
    # Chi ket luan Closed/Open khi 2 mat dong bo. Lech 2 ben => Unknown de giam false positive.
    if is_closed_l and is_closed_r:
        combined_label = "Closed"
        combined_conf = min(conf_l, conf_r)
    elif is_open_l and is_open_r:
        combined_label = "Open"
        combined_conf = min(conf_l, conf_r)
    else:
        combined_label = "Unknown"
        combined_conf = max(conf_l, conf_r)

    return combined_label, combined_conf, left_eye_crop, right_eye_crop


def main():
    # Đảm bảo mã hóa đầu ra hỗ trợ tiếng Việt trên console Windows
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    # Khởi tạo mô hình AI và camera
    classifier = EyeClassifier()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("LỖI: Không thể mở được Webcam!")
        return

    # Khởi tạo các detectors phục vụ bài toán buồn ngủ & mất tập trung
    calibrator = EARCalibrator(
        duration=CALIBRATION_DURATION,
        default_threshold=DEFAULT_EAR_THRESHOLD,
        ratio=CALIBRATION_RATIO
    )
    distraction_detector = DistractionDetector(
        threshold=FACE_LOSS_THRESHOLD
    )
    blink_detector = BlinkDetector(
        window_seconds=BLINK_RATE_WINDOW,
        blink_threshold=BLINK_RATE_THRESHOLD,
        debounce_frames=BLINK_DEBOUNCE_FRAMES,
        min_blink_duration=BLINK_MIN_DURATION,
        max_blink_duration=BLINK_MAX_DURATION,
        refractory_seconds=BLINK_REFRACTORY,
    )
    closure_detector = ContinuousClosureDetector(
        threshold=DROWSY_CLOSED_DURATION
    )
    perclos_detector = PerclosDetector(
        window_seconds=PERCLOS_WINDOW,
        threshold=PERCLOS_THRESHOLD,
        min_observation_seconds=PERCLOS_MIN_OBSERVATION_TIME
    )

    left_eye_crop = None
    right_eye_crop = None
    show_eye_windows = True
    eye_preview_scale = 2.0
    smoothed_ear = None

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while True:
            current_time = time.time()

            success, frame = cap.read()
            if not success:
                print("LỖI: Không đọc được dữ liệu từ camera!")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            status = "NO FACE"
            color = (0, 255, 255)
            alert_msg = ""

            raw_avg_ear = 0.0
            ear_for_decision = smoothed_ear if smoothed_ear is not None else 0.0
            label = "Unknown"
            confidence = 0.0
            blink_rate = 0
            total_blinks = blink_detector.get_total_blinks()
            closed_duration = 0.0
            perclos = perclos_detector.get_perclos()
            ear_threshold = calibrator.get_threshold()

            face_detected = results.multi_face_landmarks is not None
            distraction_detector.update(face_detected, current_time)

            # Xử lý chính khi có nhận diện khuôn mặt
            if face_detected:
                landmarks = results.multi_face_landmarks[0].landmark

                left_eye_pts = get_eye_coordinates(frame, landmarks, LEFT_EYE_INDICES)
                right_eye_pts = get_eye_coordinates(frame, landmarks, RIGHT_EYE_INDICES)

                left_ear = calculate_ear(left_eye_pts)
                right_ear = calculate_ear(right_eye_pts)
                raw_avg_ear = (left_ear + right_ear) / 2.0
                # EAR smooth (EMA) dung cho quyet dinh dai han de giam rung landmark.
                if smoothed_ear is None:
                    smoothed_ear = raw_avg_ear
                else:
                    smoothed_ear = (
                        EAR_SMOOTHING_ALPHA * raw_avg_ear
                        + (1.0 - EAR_SMOOTHING_ALPHA) * smoothed_ear
                    )
                ear_for_decision = smoothed_ear

                ear_threshold = calibrator.get_threshold()

                label, confidence, left_eye_crop, right_eye_crop = predict_eye_state(
                    classifier, frame, landmarks
                )

                # Xác định trạng thái nhắm mắt tức thời từ cả EAR hình học và AI CNN
                # Blink rate dung EAR raw de bat kip chop mat nhanh.
                blink_ear_signal = min(left_ear, right_ear)
                is_ear_closed_for_blink = blink_ear_signal < (ear_threshold + BLINK_EAR_MARGIN)
                is_cnn_closed = (
                    label == "Closed"
                    and confidence >= CNN_CLOSED_CONFIDENCE_THRESHOLD
                )

                # Ket hop song song hai tin hieu:
                # - EAR bat truong hop hinh hoc ro rang.
                # - CNN bat truong hop mat gan dong nhung EAR chua giam du, thuong gap voi mat hip/goc camera.
                is_closed_instant_for_blink = is_ear_closed_for_blink or is_cnn_closed

                was_calibrating = not calibrator.is_complete()
                if was_calibrating:
                    # Trong luc calibration chi lay mau EAR, khong tinh PERCLOS/blink/closure.
                    # Neu khong freeze, viec nguoi dung chinh mat/camera luc calibrate co the gay alarm sai.
                    calibrator.update(ear_for_decision, current_time)

                is_calibrating = not calibrator.is_complete()

                if is_calibrating:
                    # Freeze toan bo detector trong giai doan calibration.
                    blink_detector.reset()
                    closure_detector.reset()
                    perclos_detector.reset()
                    is_closed = False
                    alert_long_closed = False
                    alert_blink_rate = False
                    alert_perclos = False
                else:
                    if was_calibrating:
                        # Vua calibrate xong thi xoa sach bo dem truoc khi bat dau giam sat that.
                        blink_detector.reset()
                        closure_detector.reset()
                        perclos_detector.reset()

                    # Debounce yeu cau trang thai dong/mo lap lai nhieu frame moi chap nhan,
                    # tranh dem nham do landmark rung hoac model nhay lien tuc.
                    blink_detector.update(is_closed_instant_for_blink, current_time)
                    is_closed = blink_detector.is_closed()

                    closure_detector.update(is_closed, current_time)
                    perclos_detector.update(is_closed, current_time)

                    alert_long_closed = closure_detector.has_alert()
                    alert_blink_rate = blink_detector.has_alert()
                    alert_perclos = perclos_detector.has_alert()

                # Xác định cảnh báo hệ thống
                # Thu tu uu tien canh bao:
                # 1. Dang calibration thi khong bao dong.
                # 2. Mat khuon mat qua nguong thi xem la mat tap trung.
                # 3. Buon ngu neu nham mat lau, chop mat qua nhieu, hoac PERCLOS cao.
                if is_calibrating:
                    status = "CALIBRATING"
                    color = (255, 165, 0)
                    alert_msg = f"CALIBRATING EAR: {calibrator.get_progress(current_time):.0f}%"
                    stop_alarm()
                elif distraction_detector.has_alert():
                    status = "DISTRACTED"
                    color = (0, 0, 255)
                    alert_msg = f"WARNING: DISTRACTED!"
                    play_alarm()
                elif alert_long_closed or alert_blink_rate or alert_perclos:
                    status = "DROWSY"
                    color = (0, 0, 255)

                    if alert_long_closed:
                        alert_msg = "WARNING: LONG EYE CLOSURE! (DROWSY)"
                    elif alert_blink_rate:
                        alert_msg = "WARNING: HIGH BLINK RATE!"
                    else:
                        alert_msg = "WARNING: HIGH PERCLOS!"

                    play_alarm()
                elif is_closed:
                    status = "CLOSED"
                    color = (0, 165, 255)
                    alert_msg = ""
                    stop_alarm()
                else:
                    status = "OPEN"
                    color = (0, 255, 0)
                    alert_msg = ""
                    stop_alarm()

                draw_eye_landmarks(frame, left_eye_pts)
                draw_eye_landmarks(frame, right_eye_pts)

                total_blinks = blink_detector.get_total_blinks()
                blink_rate = blink_detector.get_blink_rate(current_time)
                closed_duration = closure_detector.get_closed_duration()
                perclos = perclos_detector.get_perclos()

                print(
                    f"EAR(raw={raw_avg_ear:.2f}, smooth={ear_for_decision:.2f}, thresh={ear_threshold:.2f}) "
                    f"| BlinkEAR(min={blink_ear_signal:.2f}, margin={BLINK_EAR_MARGIN:.3f}) "
                    f"| AI={label} ({confidence:.2f}) | Closed={closed_duration:.1f}s "
                    f"| PERCLOS={perclos * 100:.1f}% | Blinks={blink_rate}/{BLINK_RATE_WINDOW}s "
                    f"| Status={status}"
                )

            # Xử lý khi mất khuôn mặt
            else:
                # Mat khuon mat: khong tiep tuc tich luy closure/PERCLOS tren du lieu cu.
                closure_detector.reset()
                blink_detector.reset_temp_state()
                left_eye_crop = None
                right_eye_crop = None
                smoothed_ear = None
                ear_for_decision = 0.0
                
                if distraction_detector.has_alert():
                    perclos_detector.reset()
                    status = "DISTRACTED"
                    color = (0, 0, 255)
                    alert_msg = "CRITICAL: FACE NOT DETECTED!"
                    play_alarm()
                else:
                    status = "NO FACE"
                    color = (0, 255, 255)
                    alert_msg = ""
                    stop_alarm()

                total_blinks = blink_detector.get_total_blinks()
                blink_rate = blink_detector.get_blink_rate(current_time)
                closed_duration = 0.0
                perclos = perclos_detector.get_perclos()

            # Hiển thị các thông tin lên khung hình camera
            ai_label_color = (0, 255, 255) if label == "Unknown" else color

            cv2.putText(frame, f"EAR: {ear_for_decision:.2f} (Thresh: {ear_threshold:.2f})", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(frame, f"STATUS: {status}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(frame, f"CLOSED TIME: {closed_duration:.1f}s", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(frame, f"PERCLOS: {perclos * 100:.1f}% (Thresh: {PERCLOS_THRESHOLD * 100:.0f}%)", (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(frame, f"MODEL AI: {label} ({confidence:.2f})", (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, ai_label_color, 2)
            cv2.putText(frame, f"TOTAL BLINKS: {total_blinks}", (30, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(frame, f"BLINK RATE: {blink_rate}/{BLINK_RATE_WINDOW}s (Thresh: {BLINK_RATE_THRESHOLD})", (30, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(frame, "C: Recalibrate | E: Eye Windows | +/-: Eye Size | Q: Quit", (30, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            if alert_msg:
                cv2.putText(frame, alert_msg, (30, 370),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

            cv2.imshow("Driver Drowsiness Alert System", frame)

            if show_eye_windows:
                show_eye_preview(LEFT_EYE_WINDOW, left_eye_crop, eye_preview_scale)
                show_eye_preview(RIGHT_EYE_WINDOW, right_eye_crop, eye_preview_scale)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("c"), ord("C")):
                # Cho phep can chinh lai khi camera/tu the/anh sang thay doi trong luc demo.
                calibrator.reset()
                closure_detector.reset()
                blink_detector.reset()
                perclos_detector.reset()
                smoothed_ear = None
                stop_alarm()
                print("Recalibrating EAR threshold...")
            elif key in (ord("e"), ord("E")):
                show_eye_windows = not show_eye_windows
                if not show_eye_windows:
                    destroy_eye_preview_windows()
            elif key in (ord("+"), ord("=")):
                eye_preview_scale = min(
                    eye_preview_scale + EYE_PREVIEW_SCALE_STEP,
                    EYE_PREVIEW_MAX_SCALE,
                )
            elif key in (ord("-"), ord("_")):
                eye_preview_scale = max(
                    eye_preview_scale - EYE_PREVIEW_SCALE_STEP,
                    EYE_PREVIEW_MIN_SCALE,
                )

    stop_alarm()
    destroy_eye_preview_windows()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
