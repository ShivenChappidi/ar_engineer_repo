# AR Hand Object

A Python webcam prototype that tracks hand gestures with MediaPipe and renders a
3D object anchored to a pinching hand. It uses person segmentation to produce an
occlusion mask, so the hand can appear in front of the rendered object.

## Requirements

- Python with `pip`
- A webcam (the scripts use camera device `0`)
- An OpenGL 3.3-capable graphics environment

The MediaPipe model assets and sample OBJ model are included in this repository.
Keep them in the project root: the scripts load these files by their relative
names.

## Setup

From the project root, create and activate a virtual environment, then install
the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run the AR demo

```bash
python combined_test.py
```

- Pinch your thumb and index finger to grab the object.
- While holding the first pinch, pinch with your other hand and move it to scale
  the object.
- Press `O` to toggle person/hand occlusion.
- Press `M` to show or hide the debug mask.
- Close the window to exit.

The application prints render and tracking frame rates in the terminal. Camera
access and OpenGL support are required.

## Other scripts

These scripts are small manual experiments and also need a webcam unless noted:

| Script | Purpose |
| --- | --- |
| `test_camera.py` | Check camera capture and preview |
| `hand_test.py` | Draw a hand-anchored wireframe cube |
| `segment_test.py` | Inspect segmentation output and save sample captures |
| `render_test.py` | Render a rotating 3D cube without camera tracking |
| `fps_test.py` | Measure camera and MediaPipe throughput live |
| `fps_test2.py` | Benchmark inference using captured webcam frames |

Generated captures and benchmark output are excluded by `.gitignore`.
