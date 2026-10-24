import time, cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def measure(label, fn, n=60):
    t = time.time()
    for _ in range(n):
        fn()
    print(f"{label}: {n / (time.time() - t):.1f} FPS")

for _ in range(10):
    cap.read()

# 1. camera alone
measure("camera only", lambda: cap.read())

# 2. camera + hand landmarker
lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
    num_hands=2, running_mode=vision.RunningMode.VIDEO))
ts = [0]
def with_hands():
    ret, f = cap.read()
    ts[0] += 33
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    lm.detect_for_video(img, ts[0])
measure("camera + hands", with_hands)

# 3. camera + hands + segmenter
seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
    base_options=python.BaseOptions(model_asset_path='selfie_segmenter.tflite'),
    running_mode=vision.RunningMode.VIDEO, output_category_mask=True))
def with_both():
    ret, f = cap.read()
    ts[0] += 33
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    lm.detect_for_video(img, ts[0])
    seg.segment_for_video(img, ts[0])
measure("camera + hands + segmenter", with_both)
cap.release()
