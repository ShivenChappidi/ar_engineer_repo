import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def capture_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frames_rgb = []
    processing_frame = None
    try:
        for _ in range(10):
            ok, _ = cap.read()
            if not ok:
                raise RuntimeError('Could not read warm-up frame from camera')

        print('Hold your hand in view. Capturing frames in 3 seconds...')
        time.sleep(3)

        for _ in range(60):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Could not capture all 60 frames from camera')
            if processing_frame is None:
                processing_frame = frame.copy()
            mirrored_frame = cv2.flip(frame, 1)
            frames_rgb.append(cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    return frames_rgb, processing_frame


def main():
    frames_rgb, processing_frame = capture_frames()
    small_frames_rgb = [cv2.resize(frame, (640, 360)) for frame in frames_rgb]

    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO,
    )
    segmenter_options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path='selfie_segmenter.tflite'),
        running_mode=vision.RunningMode.VIDEO,
        output_category_mask=True,
    )

    def average_inference_ms(rgb_frames, inference_fn):
        started = time.perf_counter()
        for frame_index, frame in enumerate(rgb_frames):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp_ms = frame_index * 33
            inference_fn(mp_image, timestamp_ms)
        elapsed = time.perf_counter() - started
        return elapsed * 1000.0 / len(rgb_frames)

    def print_result(label, average_ms):
        fps = 1000.0 / average_ms
        print(f'{label}: {average_ms:.2f} ms/frame ({fps:.2f} FPS equivalent)')

    with vision.HandLandmarker.create_from_options(hand_options) as hands:
        average_ms = average_inference_ms(
            frames_rgb,
            lambda image, timestamp: hands.detect_for_video(image, timestamp),
        )
    print_result('hands at 1280x720', average_ms)

    with vision.ImageSegmenter.create_from_options(segmenter_options) as segmenter:
        average_ms = average_inference_ms(
            frames_rgb,
            lambda image, timestamp: segmenter.segment_for_video(image, timestamp),
        )
    print_result('segmenter at 1280x720', average_ms)

    with vision.HandLandmarker.create_from_options(hand_options) as hands, \
            vision.ImageSegmenter.create_from_options(segmenter_options) as segmenter:
        def run_both(image, timestamp):
            hands.detect_for_video(image, timestamp)
            segmenter.segment_for_video(image, timestamp)

        average_ms = average_inference_ms(frames_rgb, run_both)
    print_result('hands + segmenter together at 1280x720', average_ms)

    with vision.HandLandmarker.create_from_options(hand_options) as hands:
        average_ms = average_inference_ms(
            small_frames_rgb,
            lambda image, timestamp: hands.detect_for_video(image, timestamp),
        )
    print_result('hands at 640x360', average_ms)

    with vision.ImageSegmenter.create_from_options(segmenter_options) as segmenter:
        average_ms = average_inference_ms(
            small_frames_rgb,
            lambda image, timestamp: segmenter.segment_for_video(image, timestamp),
        )
    print_result('segmenter at 640x360', average_ms)

    started = time.perf_counter()
    for _ in range(60):
        cv2.cvtColor(cv2.flip(processing_frame, 1), cv2.COLOR_BGR2RGB)
    average_ms = (time.perf_counter() - started) * 1000.0 / 60
    print_result('cv2.cvtColor + cv2.flip at 1280x720', average_ms)


if __name__ == '__main__':
    main()
