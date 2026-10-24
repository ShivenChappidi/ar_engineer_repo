import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='selfie_segmenter.tflite')
options = vision.ImageSegmenterOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    output_category_mask=True
)
segmenter = vision.ImageSegmenter.create_from_options(options)

cap = cv2.VideoCapture(0)
# Give camera a moment and grab a few frames to skip initial garbage frames
for _ in range(10):
    ret, frame = cap.read()

if not ret:
    print("Failed to read frame")
    exit()

frame = cv2.flip(frame, 1)
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

result = segmenter.segment(mp_image)
category_mask = result.category_mask.numpy_view()

print(f"mask dtype: {category_mask.dtype}, shape: {category_mask.shape}")
print(f"unique values: {np.unique(category_mask)}")
print(f"count of each value:")
values, counts = np.unique(category_mask, return_counts=True)
for v, c in zip(values, counts):
    print(f"  value {v}: {c} pixels")

cv2.imwrite('test_original.jpg', frame)
cv2.imwrite('test_mask.jpg', category_mask)

print("Saved test_original.jpg and test_mask.jpg")
cap.release()
