import cv2
import numpy as np
import moderngl_window as mglw
import moderngl
import glm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading
import time

class ARWindow(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "AR Hand Object"
    window_size = (1280, 720)
    aspect_ratio = None
    PERSON_VALUE = 0
    OCCLUDE_LANDMARKS = list(range(5, 21))
    DILATE_PX = 12
    MASK_W, MASK_H = 640, 360

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ---- Camera setup ----
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # ---- Hand tracking setup ----
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=vision.RunningMode.VIDEO
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        seg_base_options = python.BaseOptions(model_asset_path='selfie_segmenter.tflite')
        seg_options = vision.ImageSegmenterOptions(
            base_options=seg_base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_category_mask=True
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(seg_options)
        self.start_time = time.time()
        self.smoothing_factor = 0.6
        self.pinch_threshold = 0.07
        self.grip_lost_threshold = 8

        def make_hand_state():
            return {
                'smoothed_center': None,
                'is_gripping': False,
                'frozen_model_matrix': None,
                'grip_lost_frames': 0,
            }

        self.primary_hand = None
        self.grip_lost_frames = 0
        self.frozen_model_matrix = None
        self.smoothed_center = None

        self.scale_multiplier = 1.0
        self.two_hand_active = False
        self.two_hand_start_dist = None
        self.two_hand_start_multiplier = None
        self.touch_threshold = 0.20  # normalized distance; tune by feel
        self.smoothed_hand_size = None

        # ---- Background quad (camera passthrough) ----
        bg_vertex = '''
        #version 330
        in vec2 in_position;
        in vec2 in_uv;
        out vec2 uv;
        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);
            uv = in_uv;
        }
        '''
        bg_fragment = '''
        #version 330
        uniform sampler2D tex;
        in vec2 uv;
        out vec4 fragColor;
        void main() {
            fragColor = texture(tex, uv);
        }
        '''
        self.bg_prog = self.ctx.program(vertex_shader=bg_vertex, fragment_shader=bg_fragment)
        bg_verts = np.array([
            -1,-1, 0,1,
             1,-1, 1,1,
             1, 1, 1,0,
            -1,-1, 0,1,
             1, 1, 1,0,
            -1, 1, 0,0,
        ], dtype='f4')
        self.bg_vbo = self.ctx.buffer(bg_verts.tobytes())
        self.bg_vao = self.ctx.vertex_array(self.bg_prog, [(self.bg_vbo, '2f 2f', 'in_position', 'in_uv')])
        occ_fragment = '''
        #version 330
        uniform sampler2D cam;
        uniform sampler2D mask;
        uniform float debug;
        in vec2 uv;
        out vec4 fragColor;
        void main() {
            float m = smoothstep(0.3, 0.7, texture(mask, uv).r);
            if (debug > 0.5) {
                fragColor = vec4(1.0, 0.0, 0.0, 0.5 * m);
            } else {
                fragColor = vec4(texture(cam, uv).rgb, m);
            }
        }
        '''
        self.occ_prog = self.ctx.program(vertex_shader=bg_vertex, fragment_shader=occ_fragment)
        self.occ_vao = self.ctx.vertex_array(self.occ_prog, [(self.bg_vbo, '2f 2f', 'in_position', 'in_uv')])
        self.mask_tex = self.ctx.texture((self.MASK_W, self.MASK_H), 1)
        self.mask_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.mask_tex.write(np.zeros((self.MASK_H, self.MASK_W), dtype=np.uint8).tobytes())
        self.tex = self.ctx.texture((1280, 720), 3)
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # ---- Cylinder setup ----
        # ---- Model shaders (with basic lighting) ----
        model_vertex = '''
        #version 330
        uniform mat4 mvp;
        uniform mat4 model;
        in vec3 in_position;
        in vec3 in_normal;
        out vec3 frag_normal;
        out vec3 frag_pos;
        void main() {
            gl_Position = mvp * vec4(in_position, 1.0);
            frag_normal = mat3(model) * in_normal;
            frag_pos = vec3(model * vec4(in_position, 1.0));
        }
        '''
        model_fragment = '''
        #version 330
        in vec3 frag_normal;
        in vec3 frag_pos;
        out vec4 fragColor;
        void main() {
            vec3 normal = normalize(frag_normal);
            // Map normal direction directly to color - each face direction gets a distinct unambiguous color
            vec3 color = normal * 0.5 + 0.5;
            fragColor = vec4(color, 1.0);
        }
        '''
        self.model_prog = self.ctx.program(vertex_shader=model_vertex, fragment_shader=model_fragment)
        self._load_model('test_model.obj')

        self.frame_lock = threading.Lock()
        self.result_lock = threading.Lock()
        self.pending_rgb = None
        self.latest_hands = None
        self.latest_mask = None
        self.run_segmenter = True
        self.occlusion_enabled = True
        self.debug_mask = False
        self.mask_version = 0
        self.uploaded_mask_version = -1
        self.tracking_count = 0
        self.render_count = 0
        self.fps_timer = time.time()
        self.stop_event = threading.Event()
        ARWindow.instance = self
        self.worker = threading.Thread(target=self._inference_worker, daemon=True)
        self.worker.start()

    def _inference_worker(self):
        last_ts = -1
        while not self.stop_event.is_set():
            with self.frame_lock:
                rgb = self.pending_rgb
                self.pending_rgb = None
            if rgb is None:
                time.sleep(0.002)
                continue
            ts = int((time.time() - self.start_time) * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hands = self.landmarker.detect_for_video(mp_image, ts)
            mask = None
            if self.run_segmenter:
                small = cv2.resize(rgb, (640, 360))
                seg_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small)
                seg_result = self.segmenter.segment_for_video(seg_image, ts)
                mask = self._build_occlusion_mask(hands, seg_result.category_mask.numpy_view())
            with self.result_lock:
                self.latest_hands = hands
                if mask is not None:
                    self.latest_mask = mask
                    self.mask_version += 1
                self.tracking_count += 1

    def _build_occlusion_mask(self, hands, person_mask):
        if person_mask.ndim == 3:
            person_mask = person_mask[:, :, 0]
        person = (person_mask == self.PERSON_VALUE).astype(np.uint8) * 255
        region = np.zeros((self.MASK_H, self.MASK_W), dtype=np.uint8)
        if hands is not None and hands.hand_landmarks:
            for landmarks in hands.hand_landmarks:
                pts = np.array([[landmarks[i].x * self.MASK_W, landmarks[i].y * self.MASK_H]
                                for i in self.OCCLUDE_LANDMARKS], dtype=np.float32)
                hull = cv2.convexHull(pts).astype(np.int32)
                cv2.fillConvexPoly(region, hull, 255)
            k = self.DILATE_PX * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            region = cv2.dilate(region, kernel)
        return cv2.bitwise_and(person, region)

    def _load_model(self, path):
        import trimesh
        mesh = trimesh.load(path)
        vertices = mesh.vertices.astype('f4')
        normals = mesh.vertex_normals.astype('f4')
        faces = mesh.faces.astype('i4')

        # Interleave position + normal per vertex, indexed by face
        vertex_data = np.hstack([vertices, normals]).astype('f4')
        index_data = faces.flatten().astype('i4')

        self.model_vbo = self.ctx.buffer(vertex_data.tobytes())
        self.model_ibo = self.ctx.buffer(index_data.tobytes())
        self.model_vao = self.ctx.vertex_array(
            self.model_prog,
            [(self.model_vbo, '3f 3f', 'in_position', 'in_normal')],
            self.model_ibo
        )

    def on_render(self, time_val, frame_time):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with self.frame_lock:
            self.pending_rgb = rgb
        with self.result_lock:
            result = self.latest_hands
        mask_to_upload = None
        with self.result_lock:
            if self.mask_version != self.uploaded_mask_version and self.latest_mask is not None:
                mask_to_upload = self.latest_mask
                self.uploaded_mask_version = self.mask_version
        if mask_to_upload is not None:
            self.mask_tex.write(np.ascontiguousarray(mask_to_upload).tobytes())

        # Update camera background texture
        self.tex.write(rgb.tobytes())

        self.ctx.clear(0.0, 0.0, 0.0)

        # Draw background (no depth test)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.tex.use(location=0)
        self.bg_prog['tex'] = 0
        self.bg_vao.render(moderngl.TRIANGLES)

        # Draw cube anchored to hand (with depth test)
        # ---- Process each detected hand independently ----
        # ---- Gather raw per-hand data ----
        hands_data = {}
        if result is not None and result.hand_landmarks and result.handedness:
            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                label = handedness[0].category_name
                thumb_tip = np.array([hand_landmarks[4].x, hand_landmarks[4].y, hand_landmarks[4].z])
                index_tip = np.array([hand_landmarks[8].x, hand_landmarks[8].y, hand_landmarks[8].z])
                pinch_dist = np.linalg.norm(thumb_tip - index_tip)
                is_pinching = pinch_dist < self.pinch_threshold

                palm_ids = [0, 5, 17]
                px = np.mean([hand_landmarks[i].x for i in palm_ids])
                py = np.mean([hand_landmarks[i].y for i in palm_ids])
                pz = np.mean([hand_landmarks[i].z for i in palm_ids])
                center = np.array([px, py, pz])

                hands_data[label] = {'landmarks': hand_landmarks, 'center': center, 'pinching': is_pinching}

        # ---- Assign primary hand (whichever pinches first) ----
        if self.primary_hand is None:
            for label, d in hands_data.items():
                if d['pinching']:
                    self.primary_hand = label
                    break

        if self.primary_hand is not None and self.primary_hand in hands_data and hands_data[self.primary_hand]['pinching']:
            self.grip_lost_frames = 0
            primary = hands_data[self.primary_hand]
            hand_landmarks = primary['landmarks']

            raw_center = primary['center']
            if self.smoothed_center is None:
                self.smoothed_center = raw_center
            else:
                self.smoothed_center = self.smoothing_factor * raw_center + (1 - self.smoothing_factor) * self.smoothed_center

            sc = self.smoothed_center
            ndc_x = (sc[0] * 2.0) - 1.0
            ndc_y = 1.0 - (sc[1] * 2.0)
            depth_z = -2.0 + sc[2] * 2.0

            fh, fw = frame.shape[:2]

            def px_dist(a, b):
                return np.hypot((hand_landmarks[a].x - hand_landmarks[b].x) * fw,
                                (hand_landmarks[a].y - hand_landmarks[b].y) * fh)

            # Apparent hand size in pixels: wrist->middle knuckle and index->pinky knuckle
            hand_size = (px_dist(0, 9) + px_dist(5, 17)) / 2.0
            if self.smoothed_hand_size is None:
                self.smoothed_hand_size = hand_size
            else:
                self.smoothed_hand_size = 0.3 * hand_size + 0.7 * self.smoothed_hand_size

            size_far, size_near = 60.0, 200.0   # placeholder calibration values
            t = np.clip((self.smoothed_hand_size - size_far) / (size_near - size_far), 0.0, 1.0)
            base_scale = 0.3 + t * (0.9 - 0.3)
            # print(f"hand_size_px: {self.smoothed_hand_size:.1f}, t: {t:.2f}, scale: {base_scale:.2f}")

            # ---- Check for second hand touching the object (activates scale mode) ----
            other_label = 'Right' if self.primary_hand == 'Left' else 'Left'
            if other_label in hands_data and hands_data[other_label]['pinching']:
                other_center = hands_data[other_label]['center']
                two_hand_dist = np.linalg.norm(raw_center - other_center)

                if not self.two_hand_active:
                    # Only check "touching" at the MOMENT the second hand starts pinching
                    dist_to_object = np.linalg.norm(other_center - sc)
                    if dist_to_object < self.touch_threshold:
                        self.two_hand_active = True
                        self.two_hand_start_dist = two_hand_dist
                        self.two_hand_start_multiplier = self.scale_multiplier
                else:
                    # Already active - keep scaling regardless of how far hand has moved since
                    ratio = two_hand_dist / (self.two_hand_start_dist + 1e-6)
                    self.scale_multiplier = np.clip(self.two_hand_start_multiplier * ratio, 0.3, 3.0)
            else:
                self.two_hand_active = False  # only exits when second hand actually stops pinching

            final_scale = base_scale * self.scale_multiplier

            wrist = np.array([hand_landmarks[0].x, -hand_landmarks[0].y, -hand_landmarks[0].z])
            index_base = np.array([hand_landmarks[5].x, -hand_landmarks[5].y, -hand_landmarks[5].z])
            pinky_base = np.array([hand_landmarks[17].x, -hand_landmarks[17].y, -hand_landmarks[17].z])

            up_vec = index_base - wrist
            up_vec = up_vec / (np.linalg.norm(up_vec) + 1e-6)
            right_vec = wrist - pinky_base
            right_vec = right_vec / (np.linalg.norm(right_vec) + 1e-6)
            forward_vec = np.cross(up_vec, right_vec)
            forward_vec = forward_vec / (np.linalg.norm(forward_vec) + 1e-6)
            right_vec = np.cross(up_vec, forward_vec)
            right_vec = right_vec / (np.linalg.norm(right_vec) + 1e-6)

            rotation_mat = glm.mat4(
                glm.vec4(float(right_vec[0]), float(right_vec[1]), float(right_vec[2]), 0.0),
                glm.vec4(float(up_vec[0]), float(up_vec[1]), float(up_vec[2]), 0.0),
                glm.vec4(float(forward_vec[0]), float(forward_vec[1]), float(forward_vec[2]), 0.0),
                glm.vec4(0.0, 0.0, 0.0, 1.0)
            )

            model = glm.translate(glm.mat4(1.0), glm.vec3(ndc_x * 1.5, ndc_y * 1.0, depth_z))
            model = model * rotation_mat
            model = glm.scale(model, glm.vec3(final_scale, final_scale, final_scale))

            self.frozen_model_matrix = model
        else:
            self.grip_lost_frames += 1
            if self.grip_lost_frames > self.grip_lost_threshold:
                self.primary_hand = None
                self.two_hand_active = False

        # ---- Render ----
        self.ctx.enable(moderngl.DEPTH_TEST)
        view = glm.mat4(1.0)
        proj = glm.perspective(glm.radians(45.0), self.wnd.aspect_ratio, 0.1, 100.0)

        if self.frozen_model_matrix is not None:
            mvp = proj * view * self.frozen_model_matrix
            self.model_prog['mvp'].write(mvp.to_bytes())
            self.model_prog['model'].write(self.frozen_model_matrix.to_bytes())
            self.model_vao.render(moderngl.TRIANGLES)

        if self.occlusion_enabled or self.debug_mask:
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            self.tex.use(location=0)
            self.mask_tex.use(location=1)
            self.occ_prog['cam'] = 0
            self.occ_prog['mask'] = 1
            self.occ_prog['debug'] = 1.0 if self.debug_mask else 0.0
            self.occ_vao.render(moderngl.TRIANGLES)
            self.ctx.disable(moderngl.BLEND)

        self.render_count += 1
        now = time.time()
        if now - self.fps_timer >= 1.0:
            with self.result_lock:
                tc = self.tracking_count
                self.tracking_count = 0
            elapsed = now - self.fps_timer
            print(f"render FPS: {self.render_count / elapsed:.1f}, tracking FPS: {tc / elapsed:.1f}")
            self.render_count = 0
            self.fps_timer = now

    def on_key_event(self, key, action, modifiers):
        print(f"key event: key={key}, action={action}")
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.O:
                self.occlusion_enabled = not self.occlusion_enabled
                print(f"occlusion_enabled = {self.occlusion_enabled}")
            elif key == self.wnd.keys.M:
                self.debug_mask = not self.debug_mask
                print(f"debug_mask = {self.debug_mask}")

if __name__ == '__main__':
    try:
        mglw.run_window_config(ARWindow)
    finally:
        inst = getattr(ARWindow, 'instance', None)
        if inst is not None:
            inst.stop_event.set()
            inst.worker.join(timeout=3.0)
            tasks = [inst.landmarker]
            segmenter = getattr(inst, 'segmenter', None)
            if segmenter is not None:
                tasks.append(segmenter)
            for task in tasks:
                try:
                    task.close()
                except Exception:
                    pass
