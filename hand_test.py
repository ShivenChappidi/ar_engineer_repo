import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Set up the hand landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)
landmarker = vision.HandLandmarker.create_from_options(options)

# Define cube corners in 3D (centered at origin, size ~0.08 in normalized-ish units)
cube_size = 0.06
cube_points = np.array([
    [-cube_size, -cube_size, -cube_size],
    [ cube_size, -cube_size, -cube_size],
    [ cube_size,  cube_size, -cube_size],
    [-cube_size,  cube_size, -cube_size],
    [-cube_size, -cube_size,  cube_size],
    [ cube_size, -cube_size,  cube_size],
    [ cube_size,  cube_size,  cube_size],
    [-cube_size,  cube_size,  cube_size],
], dtype=np.float32)

# Edges connecting the 8 corners into a cube wireframe
edges = [
    (0,1),(1,2),(2,3),(3,0),  # bottom face
    (4,5),(5,6),(6,7),(7,4),  # top face
    (0,4),(1,5),(2,6),(3,7)   # vertical edges
]

cap = cv2.VideoCapture(0)
frame_timestamp_ms = 0

# Simple smoothing state
smoothed_center = None
smoothing_factor = 0.3  # lower = smoother/slower, higher = snappier

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_timestamp_ms += 33

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]  # just the first hand for now

        # Palm center approx = average of wrist(0), index_mcp(5), pinky_mcp(17)
        palm_ids = [0, 5, 17]
        px = np.mean([hand_landmarks[i].x for i in palm_ids])
        py = np.mean([hand_landmarks[i].y for i in palm_ids])
        pz = np.mean([hand_landmarks[i].z for i in palm_ids])

        raw_center = np.array([px, py, pz])

        # Smooth the center position to reduce jitter
        if smoothed_center is None:
            smoothed_center = raw_center
        else:
            smoothed_center = smoothing_factor * raw_center + (1 - smoothing_factor) * smoothed_center

        # Convert normalized (0-1) palm x,y to pixel coords, keep z as relative depth
        center_px = smoothed_center[0] * w
        center_py = smoothed_center[1] * h
        center_pz = smoothed_center[2] * w  # scale z roughly similarly to x for depth feel

        # Position cube points around this center (simple orthographic-ish projection)
        translated = cube_points * w * 0.5 + np.array([center_px, center_py, center_pz])

        # Project to 2D by just dropping z with a slight perspective scale
        projected = []
        for pt in translated:
            depth_scale = 1.0 / (1.0 + pt[2] / (w * 2))  # cheap perspective feel
            x2d = int(center_px + (pt[0] - center_px) * depth_scale)
            y2d = int(center_py + (pt[1] - center_py) * depth_scale)
            projected.append((x2d, y2d))

        # Draw wireframe edges
        for edge in edges:
            pt1 = projected[edge[0]]
            pt2 = projected[edge[1]]
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Draw palm center dot for reference
        cv2.circle(frame, (int(center_px), int(center_py)), 5, (0, 0, 255), -1)

    cv2.imshow('Hand-Anchored Cube', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
