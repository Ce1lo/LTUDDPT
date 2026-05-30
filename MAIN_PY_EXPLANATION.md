# Giai Thich Code `src/main.py`

Duoi day la luong cua `src/main.py` tu tren xuong duoi. File nay la "bo dieu phoi": lay hinh webcam, dung MediaPipe tim mat/mat, tinh EAR, goi AI CNN, goi cac detector, roi quyet dinh bao dong.

## Import Va Hang So

```python
import time
import sys
import cv2
import mediapipe as mp
```

- `time`: lay thoi gian hien tai de tinh thoi luong nham mat, mat mat, PERCLOS.
- `sys`: chinh encoding console.
- `cv2`: OpenCV, doc webcam, ve text, hien thi anh.
- `mediapipe`: nhan dien landmark khuon mat.

Tu `config.py` import cac nguong:

- `DEFAULT_EAR_THRESHOLD = 0.2`: nguong EAR mac dinh, thap hon thi mat co the dang nham.
- `DROWSY_CLOSED_DURATION = 2.0`: nham mat lien tuc 2 giay thi canh bao.
- `PERCLOS_WINDOW = 60.0`: tinh PERCLOS trong 60 giay gan nhat.
- `PERCLOS_THRESHOLD = 0.4`: nham mat >= 40% thoi gian thi canh bao.
- `CNN_CLOSED_CONFIDENCE_THRESHOLD = 0.88`: AI phai tu tin >= 88% moi tin la mat dong.
- `EAR_SMOOTHING_ALPHA = 0.25`: he so lam muot EAR.
- Cac bien `BLINK_*`: dung de dem chop mat va canh bao chop mat nhieu.
- `CALIBRATION_DURATION = 5.0`: 5 giay dau de tu hieu chuan EAR.
- `FACE_LOSS_THRESHOLD = 2.0`: mat mat 2 giay thi coi la mat tap trung.

Import cac module chinh:

```python
from alarm_controller import play_alarm, stop_alarm
from eye_classifier import EyeClassifier
from eye_metrics import calculate_ear
```

- `play_alarm()`: bat nhac/coi canh bao.
- `stop_alarm()`: tat canh bao.
- `EyeClassifier`: load model Keras de phan loai mat mo/nham.
- `calculate_ear()`: tinh EAR tu 6 diem mat.

Import cac detector:

- `EARCalibrator`: tu tinh nguong EAR ca nhan hoa.
- `DistractionDetector`: phat hien mat khuon mat.
- `BlinkDetector`: loc trang thai nham/mo va dem chop mat.
- `ContinuousClosureDetector`: do thoi gian nham mat lien tuc.
- `PerclosDetector`: tinh phan tram thoi gian nham mat.

```python
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
```

Day la 6 landmark MediaPipe cho moi mat. 6 diem nay dung cho cong thuc EAR: 2 khoe mat ngang, 4 diem mi tren/duoi.

## `get_eye_coordinates(frame, landmarks, eye_indices)`

Ham nay doi landmark dang ti le MediaPipe sang toa do pixel.

Tham so:

- `frame`: anh hien tai tu webcam, dang ma tran OpenCV.
- `landmarks`: danh sach diem landmark cua khuon mat.
- `eye_indices`: danh sach index cua mat trai hoac phai.

Bien:

```python
h, w, _ = frame.shape
coords = []
```

- `h`: chieu cao anh.
- `w`: chieu rong anh.
- `_`: so kenh mau, khong dung.
- `coords`: danh sach toa do pixel.

Vong lap:

```python
pt = landmarks[idx]
coords.append((int(pt.x * w), int(pt.y * h)))
```

MediaPipe tra `pt.x`, `pt.y` trong khoang 0-1. Nhan voi `w`, `h` de ra pixel that.

## `draw_eye_landmarks(frame, eye_coords)`

Ve cac cham xanh len mat.

- `frame`: anh webcam.
- `eye_coords`: danh sach toa do mat da doi sang pixel.

Goi:

```python
cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
```

Nghia la ve hinh tron ban kinh 2, mau xanh la, to kin.

## `destroy_eye_preview_windows()`

Dong 2 cua so preview mat:

```python
LEFT_EYE_WINDOW = "Left Eye"
RIGHT_EYE_WINDOW = "Right Eye"
```

Dung `try/except cv2.error` de neu cua so chua ton tai thi khong bi crash.

## `show_eye_preview(window_name, eye_crop, scale)`

Hien thi anh crop cua mat.

Tham so:

- `window_name`: ten cua so, `"Left Eye"` hoac `"Right Eye"`.
- `eye_crop`: anh vung mat da cat.
- `scale`: he so phong to.

Neu `eye_crop` rong thi return. Neu `scale != 1.0`, ham goi `cv2.resize()` de phong anh roi `cv2.imshow()`.

## `crop_eye_region(frame, landmarks, eye_indices, padding=25)`

Cat vung mat tu frame.

Tham so:

- `frame`: anh webcam.
- `landmarks`: landmark mat.
- `eye_indices`: diem mat trai/phai.
- `padding=25`: noi vung crop them 25 pixel quanh mat.

Bien:

```python
xs = [...]
ys = [...]
```

Lay toan bo toa do x/y cua 6 diem mat.

```python
x_min, x_max, y_min, y_max
```

Tinh khung chu nhat bao quanh mat, co padding, dong thoi dung `max(..., 0)` va `min(..., w/h)` de khong vuot khoi anh.

Return:

```python
frame[y_min:y_max, x_min:x_max]
```

Day la anh con vung mat de dua vao AI.

## `predict_eye_state(classifier, frame, landmarks)`

Ham nay goi model AI de doan mat mo/nham.

Tham so:

- `classifier`: object `EyeClassifier`.
- `frame`: anh webcam.
- `landmarks`: landmark mat.

Dau tien cat hai mat:

```python
left_eye_crop = crop_eye_region(...)
right_eye_crop = crop_eye_region(...)
```

Neu crop rong:

```python
return "Unknown", 0.0, left_eye_crop, right_eye_crop
```

Sau do goi:

```python
label_l, conf_l = classifier.predict(left_eye_crop)
label_r, conf_r = classifier.predict(right_eye_crop)
```

`EyeClassifier.predict()` trong `src/eye_classifier.py` lam nhu sau:

- resize anh mat ve `224x224`.
- doi BGR sang RGB.
- normalize pixel ve khoang `[-1, 1]`.
- them batch dimension bang `np.expand_dims`.
- goi `self.model.predict(batch)`.
- lay class co xac suat cao nhat bang `np.argmax`.
- return `label`, `confidence`.

Quay lai `predict_eye_state()`:

```python
is_closed_l = "closed" in label_l_normalized
is_open_l = "open" in label_l_normalized
```

Ham khong so sanh cung label, ma kiem tra chuoi co chua `"closed"` hoac `"open"`.

Logic ket hop:

- Neu ca 2 mat deu closed -> `combined_label = "Closed"`.
- Neu ca 2 mat deu open -> `combined_label = "Open"`.
- Neu lech nhau -> `"Unknown"` de giam bao dong sai.

Confidence:

- Khi hai mat dong bo, lay `min(conf_l, conf_r)` vi mat yeu hon quyet dinh do tin cay chung.
- Khi unknown, lay `max(conf_l, conf_r)`.

## `main()`

Day la ham chay toan bo chuong trinh.

Dau tien:

```python
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
```

Cho console Windows in tieng Viet tot hon.

Khoi tao:

```python
classifier = EyeClassifier()
cap = cv2.VideoCapture(0)
```

- `EyeClassifier()` load model `models/keras_model.h5` va labels `models/labels.txt`.
- `cv2.VideoCapture(0)` mo webcam mac dinh.

Neu khong mo duoc webcam:

```python
if not cap.isOpened():
    print(...)
    return
```

## Khoi Tao Detector

```python
calibrator = EARCalibrator(...)
```

`EARCalibrator` nhan:

- `duration`: thoi gian hieu chuan.
- `default_threshold`: nguong mac dinh.
- `ratio`: lay EAR trung binh luc mat mo nhan voi ratio de ra nguong ca nhan.

Trong `ear_calibration.py`, no luu:

- `start_time`: luc bat dau calibration.
- `ear_history`: danh sach EAR trong luc calibration.
- `calibrated_threshold`: nguong sau khi tinh xong.

```python
distraction_detector = DistractionDetector(threshold=FACE_LOSS_THRESHOLD)
```

Theo doi mat mat qua 2 giay.

```python
blink_detector = BlinkDetector(...)
```

Các tham so quan trong:

- `window_seconds`: cua so tinh blink rate.
- `blink_threshold`: so blink toi da truoc khi canh bao.
- `debounce_frames`: can bao nhieu frame giong nhau moi chap nhan doi trang thai.
- `min_blink_duration`, `max_blink_duration`: loc blink qua ngan hoac qua dai.
- `refractory_seconds`: chong dem trung mot blink.

```python
closure_detector = ContinuousClosureDetector(threshold=DROWSY_CLOSED_DURATION)
```

Do nham mat lien tuc.

```python
perclos_detector = PerclosDetector(...)
```

Luu mau `(time, is_closed)` roi tinh:

```text
PERCLOS = tong thoi gian mat nham / tong thoi gian quan sat
```

## Bien Trang Thai Ban Dau

```python
left_eye_crop = None
right_eye_crop = None
show_eye_windows = True
eye_preview_scale = 2.0
smoothed_ear = None
```

- `left_eye_crop`, `right_eye_crop`: anh crop mat de preview.
- `show_eye_windows`: co hien cua so mat hay khong.
- `eye_preview_scale`: phong to preview mat 2 lan.
- `smoothed_ear`: EAR da lam muot bang EMA.

## MediaPipe FaceMesh

```python
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:
```

Tham so:

- `max_num_faces=1`: chi tim 1 khuon mat.
- `refine_landmarks=True`: landmark chi tiet hon, tot cho mat.
- `min_detection_confidence=0.5`: nguong phat hien mat.
- `min_tracking_confidence=0.5`: nguong tracking.

## Vong Lap Webcam

```python
while True:
    current_time = time.time()
    success, frame = cap.read()
```

- `current_time`: timestamp hien tai.
- `success`: doc webcam thanh cong khong.
- `frame`: anh hien tai.

Neu khong doc duoc thi break.

```python
frame = cv2.flip(frame, 1)
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb_frame)
```

- Flip de hinh giong guong.
- MediaPipe can RGB, OpenCV mac dinh la BGR.
- `results` chua landmark neu thay mat.

## Reset Bien Moi Frame

```python
status = "NO FACE"
color = (0, 255, 255)
alert_msg = ""
```

- `status`: trang thai hien thi.
- `color`: mau text.
- `alert_msg`: dong canh bao.

Cac bien do:

```python
raw_avg_ear = 0.0
ear_for_decision = smoothed_ear if smoothed_ear is not None else 0.0
label = "Unknown"
confidence = 0.0
blink_rate = 0
total_blinks = blink_detector.get_total_blinks()
closed_duration = 0.0
perclos = perclos_detector.get_perclos()
ear_threshold = calibrator.get_threshold()
```

`face_detected`:

```python
face_detected = results.multi_face_landmarks is not None
distraction_detector.update(face_detected, current_time)
```

Neu co mat thi detector reset mat mat. Neu khong co mat thi bat dau tinh thoi gian mat mat.

## Khi Co Mat

```python
landmarks = results.multi_face_landmarks[0].landmark
```

Lay landmark cua mat dau tien.

Tinh toa do mat:

```python
left_eye_pts = get_eye_coordinates(...)
right_eye_pts = get_eye_coordinates(...)
```

Tinh EAR:

```python
left_ear = calculate_ear(left_eye_pts)
right_ear = calculate_ear(right_eye_pts)
raw_avg_ear = (left_ear + right_ear) / 2.0
```

`calculate_ear()` trong `src/eye_metrics.py`:

- nhan 6 diem mat.
- tinh 2 khoang cach doc mi mat.
- tinh 1 khoang cach ngang giua 2 khoe mat.
- tra ve:

```text
EAR = (vertical_1 + vertical_2) / (2 * horizontal)
```

Mat nham thi khoang cach doc giam, EAR giam.

Lam muot EAR:

```python
smoothed_ear = EAR_SMOOTHING_ALPHA * raw_avg_ear + (1 - EAR_SMOOTHING_ALPHA) * smoothed_ear
```

Neu frame dau tien thi `smoothed_ear = raw_avg_ear`.

Goi AI:

```python
label, confidence, left_eye_crop, right_eye_crop = predict_eye_state(classifier, frame, landmarks)
```

Tinh tin hieu nham mat cho blink:

```python
blink_ear_signal = min(left_ear, right_ear)
is_ear_closed_for_blink = blink_ear_signal < (ear_threshold + BLINK_EAR_MARGIN)
```

Dung `min(left_ear, right_ear)` vi chi can mot mat co tin hieu thap ro la co kha nang dang chop/nham.

AI closed:

```python
is_cnn_closed = label == "Closed" and confidence >= CNN_CLOSED_CONFIDENCE_THRESHOLD
```

Ket hop EAR va CNN:

```python
is_closed_instant_for_blink = is_ear_closed_for_blink or is_cnn_closed
```

Tuc la chi can mot trong hai nguon tin noi "closed" thi coi frame hien tai la closed.

## Calibration

```python
was_calibrating = not calibrator.is_complete()
```

Neu chua calibration xong:

```python
calibrator.update(ear_for_decision, current_time)
```

`EARCalibrator.update()`:

- neu lan dau thi set `start_time`.
- trong 5 giay dau, append EAR vao `ear_history`.
- het 5 giay, lay trung binh EAR roi nhan `ratio`.
- gioi han threshold trong `[0.12, 0.28]`.

Sau do:

```python
is_calibrating = not calibrator.is_complete()
```

Neu van dang calibration:

- reset blink/closure/perclos.
- khong cho bao dong.
- `is_closed = False`.

Neu vua calibration xong:

```python
if was_calibrating:
    blink_detector.reset()
    closure_detector.reset()
    perclos_detector.reset()
```

Xoa du lieu nhieu trong giai doan calibration.

## Cap Nhat Detector Khi Khong Calibration

```python
blink_detector.update(is_closed_instant_for_blink, current_time)
is_closed = blink_detector.is_closed()
```

`BlinkDetector.update()`:

- debounce trang thai dong/mo qua nhieu frame.
- khi chuyen tu mo -> dong: luu `closed_start_time`.
- khi chuyen tu dong -> mo: tinh `blink_duration`.
- neu duration hop le va khong bi refractory thi tang `total_blinks`.
- luu timestamp blink vao `blink_timestamps`.
- neu so blink trong cua so >= threshold thi `has_rate_alert = True`.

Tiep:

```python
closure_detector.update(is_closed, current_time)
perclos_detector.update(is_closed, current_time)
```

`ContinuousClosureDetector.update()`:

- neu `is_closed=True`, bat dau/tang `closed_duration`.
- neu mo mat lai, reset duration ve 0.

`PerclosDetector.update()`:

- append `(current_time, is_closed)`.
- xoa mau cu ngoai cua so 60 giay.

Lay canh bao:

```python
alert_long_closed = closure_detector.has_alert()
alert_blink_rate = blink_detector.has_alert()
alert_perclos = perclos_detector.has_alert()
```

## Quyet Dinh Status

Uu tien:

1. Calibration.
2. Mat mat.
3. Buon ngu.
4. Mat dang nham.
5. Mat mo.

```python
if is_calibrating:
    status = "CALIBRATING"
    stop_alarm()
```

Neu dang calibration thi khong bao dong.

```python
elif distraction_detector.has_alert():
    status = "DISTRACTED"
    play_alarm()
```

Neu mat mat qua nguong thi bao dong.

```python
elif alert_long_closed or alert_blink_rate or alert_perclos:
    status = "DROWSY"
    play_alarm()
```

Neu mot trong ba dau hieu buon ngu xay ra:

- nham mat lau,
- chop mat qua nhieu,
- PERCLOS cao,

thi bat alarm.

```python
elif is_closed:
    status = "CLOSED"
    stop_alarm()
```

Mat nham nhung chua nguy hiem thi chi hien CLOSED.

```python
else:
    status = "OPEN"
    stop_alarm()
```

Mat mo binh thuong.

Sau do ve landmark mat, cap nhat so lieu moi nhat:

```python
total_blinks = blink_detector.get_total_blinks()
blink_rate = blink_detector.get_blink_rate(current_time)
closed_duration = closure_detector.get_closed_duration()
perclos = perclos_detector.get_perclos()
```

## Khi Khong Co Mat

Neu khong detect duoc mat:

```python
closure_detector.reset()
blink_detector.reset_temp_state()
left_eye_crop = None
right_eye_crop = None
smoothed_ear = None
ear_for_decision = 0.0
```

Reset cac trang thai phu de khong dung du lieu cu.

Neu mat mat du lau:

```python
status = "DISTRACTED"
alert_msg = "CRITICAL: FACE NOT DETECTED!"
play_alarm()
```

Neu moi mat mat chua lau:

```python
status = "NO FACE"
stop_alarm()
```

## Hien Thi Len Man Hinh

Các dong `cv2.putText()` ve thong tin:

- EAR hien tai va threshold.
- STATUS.
- CLOSED TIME.
- PERCLOS.
- MODEL AI.
- TOTAL BLINKS.
- BLINK RATE.
- phim dieu khien.

Neu co `alert_msg`, ve canh bao do.

```python
cv2.imshow("Driver Drowsiness Alert System", frame)
```

Hien cua so chinh.

Neu bat preview mat:

```python
show_eye_preview(LEFT_EYE_WINDOW, left_eye_crop, eye_preview_scale)
show_eye_preview(RIGHT_EYE_WINDOW, right_eye_crop, eye_preview_scale)
```

## Xu Ly Phim

```python
key = cv2.waitKey(1) & 0xFF
```

Cho phim 1ms.

- `q`: thoat.
- `c`: reset calibration, detector, alarm.
- `e`: bat/tat cua so mat.
- `+` hoac `=`: tang kich thuoc preview mat.
- `-` hoac `_`: giam kich thuoc preview mat.

## Ket Thuc

Sau khi thoat vong lap:

```python
stop_alarm()
destroy_eye_preview_windows()
cap.release()
cv2.destroyAllWindows()
```

- Tat alarm.
- Dong cua so preview mat.
- Nha webcam.
- Dong toan bo cua so OpenCV.

Cuoi file:

```python
if __name__ == "__main__":
    main()
```

Neu chay truc tiep `python src/main.py`, chuong trinh goi `main()`. Neu file nay bi import tu file khac, `main()` se khong tu chay.
