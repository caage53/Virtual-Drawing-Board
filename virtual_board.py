import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque

#OPTIMIZATION SETTINGS
PROCESS_WIDTH = 640

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
   max_num_hands=2,
   model_complexity=0,
   min_detection_confidence=0.8,
   min_tracking_confidence=0.8
)

#CAMERA SETUP
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

window_name = "Virtual Board"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#Drawing State
canvas = None
drawn_shapes = []
draw_color = (0, 255, 255)
undo_stack = []
redo_stack = []

#SMOOTHING STATE
hand_smooth_state = {}
#For smoother writing
SMOOTH_LEN = 6

#STRICT PINCH SETTINGS
PINCH_THRESHOLD = 0.11
UNPINCH_THRESHOLD = 0.20

#Shape/Mode State
shape_mode = False
shape_delay = 1.0
last_shape_time = 0
pending_shape = None
shape_detect_start = 0
moving_shape_idx = None
last_move_pos = None
resizing_shape_idx = None
initial_resize_dist = 0
initial_shape_size = 0

# FPS
prev_frame_time = 0

# --- HAND UTILITY FUNCTIONS ---
# Defines helper functions to check hand size, pinch status, and open palms.
def get_hand_center(lm, w, h):
   cx = np.mean([p.x for p in lm]) * w
   cy = np.mean([p.y for p in lm]) * h
   return int(cx), int(cy)

def get_hand_scale(lm, w, h):
   return math.hypot(lm[0].x * w - lm[9].x * w, lm[0].y * h - lm[9].y * h)

def detect_pinch(lm, w, h):
   index_tip = (lm[8].x * w, lm[8].y * h)
   thumb_tip = (lm[4].x * w, lm[4].y * h)
   hand_scale = get_hand_scale(lm, w, h)
   dist = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
   return dist < hand_scale * PINCH_THRESHOLD, dist > hand_scale * UNPINCH_THRESHOLD

def euclid(a, b):
   return math.hypot(a[0] - b[0], a[1] - b[1])

def get_hand_size(lm, w, h):
   xs = [p.x for p in lm]
   ys = [p.y for p in lm]
   return ((max(xs) - min(xs)) * w + (max(ys) - min(ys)) * h) / 2.0

def is_open_palm(lm, w, h):
   wrist = (lm[0].x * w, lm[0].y * h)
   fingers_extended = 0
   if euclid((lm[4].x * w, lm[4].y * h), wrist) > euclid((lm[2].x * w, lm[2].y * h), wrist): fingers_extended += 1
   if euclid((lm[8].x * w, lm[8].y * h), wrist) > euclid((lm[6].x * w, lm[6].y * h), wrist): fingers_extended += 1
   if euclid((lm[12].x * w, lm[12].y * h), wrist) > euclid((lm[10].x * w, lm[10].y * h), wrist): fingers_extended += 1
   if euclid((lm[16].x * w, lm[16].y * h), wrist) > euclid((lm[14].x * w, lm[14].y * h), wrist): fingers_extended += 1
   if euclid((lm[20].x * w, lm[20].y * h), wrist) > euclid((lm[18].x * w, lm[18].y * h), wrist): fingers_extended += 1
   return fingers_extended >= 4

# --- SHAPE DETECTION & DRAWING ---
# Defines functions for detecting shape gestures and drawing them.
def hit_test_shape(shape, pos, threshold=40):
   cx, cy = shape['center']
   px, py = pos
   if shape['type'] == 'circle':
       return abs(math.hypot(cx - px, cy - py) - shape['size']) < threshold
   elif shape['type'] == 'rect':
       w, h = shape['size']
       x1, y1, x2, y2 = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2
       return ((abs(px - x1) < threshold or abs(px - x2) < threshold) and y1 - threshold < py < y2 + threshold) or \
           ((abs(py - y1) < threshold or abs(py - y2) < threshold) and x1 - threshold < px < x2 + threshold)
   elif shape['type'] == 'tri':
       h_tri = int(shape['size'] * math.sqrt(3) / 2)
       cnt = np.array([(cx, cy - h_tri // 2), (cx - shape['size'] // 2, cy + h_tri // 2),
                       (cx + shape['size'] // 2, cy + h_tri // 2)], np.int32)
       return abs(cv2.pointPolygonTest(cnt, (px, py), True)) < threshold
   return False

def is_circle_o_shape(lm, w, h):
   idx = np.array([lm[8].x * w, lm[8].y * h])
   thm = np.array([lm[4].x * w, lm[4].y * h])
   wri = np.array([lm[0].x * w, lm[0].y * h])

   hs = get_hand_size(lm, w, h)

   if np.linalg.norm(idx - thm) > hs * 0.35: return False

   middle_ext = np.linalg.norm(np.array([lm[12].x * w, lm[12].y * h]) - wri) > hs * 0.9
   ring_ext = np.linalg.norm(np.array([lm[16].x * w, lm[16].y * h]) - wri) > hs * 0.9
   pinky_ext = np.linalg.norm(np.array([lm[20].x * w, lm[20].y * h]) - wri) > hs * 0.9

   return middle_ext and ring_ext and pinky_ext

def interpolate_line(canvas, p1, p2, color, thickness=2):
   cv2.line(canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness, cv2.LINE_AA)

def save_state():
   undo_stack.append((canvas.copy(), [s.copy() for s in drawn_shapes]))
   if len(undo_stack) > 20: undo_stack.pop(0)

def add_rect(p1, p2, c): drawn_shapes.append({'type': 'rect', 'center': ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2),
                                             'size': (abs(p1[0] - p2[0]), abs(p1[1] - p2[1])), 'color': c})
def add_tri(cen, s, c): drawn_shapes.append({'type': 'tri', 'center': cen, 'size': s, 'color': c})

def add_circ(cen, r, c): drawn_shapes.append({'type': 'circle', 'center': cen, 'size': r, 'color': c})

def render_shapes(img, shapes):
   for s in shapes:
       cx, cy = s['center']
       if s['type'] == 'circle':
           cv2.circle(img, (cx, cy), int(s['size']), s['color'], 3, cv2.LINE_AA)
       elif s['type'] == 'rect':
           cv2.rectangle(img, (cx - s['size'][0] // 2, cy - s['size'][1] // 2),
                         (cx + s['size'][0] // 2, cy + s['size'][1] // 2), s['color'], 3, cv2.LINE_AA)
       elif s['type'] == 'tri':
           h = int(s['size'] * math.sqrt(3) / 2)
           pts = np.array([(cx, cy - h // 2), (cx - s['size'] // 2, cy + h // 2), (cx + s['size'] // 2, cy + h // 2)],
                          np.int32)
           cv2.polylines(img, [pts], True, s['color'], 3, cv2.LINE_AA)

# MAIN LOOP
# The main application loop that runs every frame.
while True:
   ok, frame = cap.read()
   inverted_frame = cv2.flip(frame, 1)
   if not ok: break
   h, w, _ = frame.shape
   if canvas is None: canvas = np.zeros_like(frame)

   ar = w / h
   frame_small = cv2.resize(frame, (PROCESS_WIDTH, int(PROCESS_WIDTH / ar)))
   result = hands_detector.process(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))

   hand_data = []

   # Hand Detection Loop
   # Processes the camera frame, finds all hands, and calculates their positions.
   if result.multi_hand_landmarks:
       for i, hlm in enumerate(result.multi_hand_landmarks):
           lbl = result.multi_handedness[i].classification[0].label
           lm = hlm.landmark
           if lbl not in hand_smooth_state: hand_smooth_state[lbl] = {'pinching': False, 'prev': None,
                                                                      'smooth': deque(maxlen=SMOOTH_LEN)}
           st = hand_smooth_state[lbl]
           raw_x, raw_y = lm[8].x * w, lm[8].y * h

           # Append to deque BEFORE calculating mean to use the new point
           st['smooth'].append((raw_x, raw_y))


           # Calculate smoothed position based on the updated deque
           sx = int(np.mean([p[0] for p in st['smooth']]))
           sy = int(np.mean([p[1] for p in st['smooth']]))
           smooth_pos = (sx, sy)

           is_p, is_u = detect_pinch(lm, w, h)
           is_open = is_open_palm(lm, w, h)

           if is_p and not st['pinching']:
               st['pinching'], st['prev'] = True, smooth_pos
               if not shape_mode: save_state()
               redo_stack.clear()
           elif is_u and st['pinching']:
               st['pinching'], st['prev'] = False, None


           hand_data.append({'center': get_hand_center(lm, w, h), 'index_tip': smooth_pos,
                             'thumb_tip': (int(lm[4].x * w), int(lm[4].y * h)), 'lm': lm, 'open': is_open,
                             'pinching': st['pinching']})
           cv2.circle(frame, smooth_pos, 5, (0, 255, 0) if st['pinching'] else (0, 0, 255), -1, cv2.LINE_AA)
           if not shape_mode and st['pinching'] and st['prev']:
               interpolate_line(canvas, st['prev'], smooth_pos, draw_color)
               st['prev'] = smooth_pos

   is_moving = False
   is_resizing = False
   palm_is_open = any(h['open'] for h in hand_data)

   # RESIZE
   # Checks if one *pinching* hand is pulling a shape's edge to resize it.
   if shape_mode and palm_is_open and hand_data:
       mover = next((h for h in hand_data if h['open']), None)
       if mover:
           ix, iy = mover['index_tip']
           tx, ty = mover['thumb_tip']
           if moving_shape_idx is None:
               for i, s in enumerate(reversed(drawn_shapes)):
                   if hit_test_shape(s, (ix, iy)) and hit_test_shape(s, (tx, ty)):
                       moving_shape_idx, last_move_pos = len(drawn_shapes) - 1 - i, ((ix + tx) // 2, (iy + ty) // 2)
                       break
           else:
               is_moving = True
               cur_pos = ((ix + tx) // 2, (iy + ty) // 2)
               dx, dy = cur_pos[0] - last_move_pos[0], cur_pos[1] - last_move_pos[1]
               drawn_shapes[moving_shape_idx]['center'] = (drawn_shapes[moving_shape_idx]['center'][0] + dx,
                                                           drawn_shapes[moving_shape_idx]['center'][1] + dy)
               last_move_pos = cur_pos
   else:
       moving_shape_idx = None

   # SHAPE CREATION
   # Checks if specific hand gestures are held long enough to create a new shape.
   if shape_mode and not is_moving and not palm_is_open and hand_data:
       resizer = next((h for h in hand_data if h['pinching']), None)
       if resizer:
           is_resizing = True
           curr_pos = resizer['index_tip']
           if resizing_shape_idx is None:
               for i, s in enumerate(reversed(drawn_shapes)):
                   if hit_test_shape(s, curr_pos):
                       resizing_shape_idx, initial_resize_dist, initial_shape_size = len(drawn_shapes) - 1 - i, euclid(
                           curr_pos, drawn_shapes[len(drawn_shapes) - 1 - i]['center']), \
                           drawn_shapes[len(drawn_shapes) - 1 - i]['size']
                       save_state()
                       break
           elif resizing_shape_idx < len(drawn_shapes):
               scale = euclid(curr_pos, drawn_shapes[resizing_shape_idx]['center']) / max(20, initial_resize_dist)
               s = drawn_shapes[resizing_shape_idx]
               if s['type'] in ['circle', 'tri']:
                   s['size'] = max(15, int(initial_shape_size * scale))
               elif s['type'] == 'rect':
                   s['size'] = (max(15, int(initial_shape_size[0] * scale)),
                                max(15, int(initial_shape_size[1] * scale)))
       else:
           resizing_shape_idx = None
   else:
       resizing_shape_idx = None

   if shape_mode and not is_moving and not is_resizing:
       cur_shape = None
       if len(hand_data) == 2:
           if not hand_data[0]['open'] and not hand_data[1]['open']:
               h1, h2 = hand_data
               idx_d, thm_d = euclid(h1['index_tip'], h2['index_tip']), euclid(h1['thumb_tip'], h2['thumb_tip'])
               avg_scale = (get_hand_scale(h1['lm'], w, h) + get_hand_scale(h2['lm'], w, h)) / 2
               if idx_d < avg_scale * 0.6 and thm_d < avg_scale * 0.6:
                   cur_shape = "triangle"
               elif (0.8 < thm_d / (idx_d + 1e-5) < 1.2) and (idx_d < avg_scale * 3.5) and (idx_d > avg_scale * 1.0):
                   cur_shape = "rectangle"
       elif len(hand_data) == 1 and not hand_data[0]['open'] and is_circle_o_shape(hand_data[0]['lm'], w, h):
           cur_shape = "circle"

       if cur_shape:
           if pending_shape != cur_shape:
               pending_shape, shape_detect_start = cur_shape, time.time()
           elif time.time() - shape_detect_start > shape_delay:
               save_state()
               if cur_shape == "triangle":
                   h1, h2 = hand_data
                   top, base = ((h1['index_tip'][0] + h2['index_tip'][0]) // 2,
                                (h1['index_tip'][1] + h2['index_tip'][1]) // 2), (
                       (h1['thumb_tip'][0] + h2['thumb_tip'][0]) // 2, (h1['thumb_tip'][1] + h2['thumb_tip'][1]) // 2)
                   add_tri(((top[0] + base[0]) // 2, (top[1] + base[1]) // 2), int(euclid(top, base) * 1.5),
                           draw_color)
               elif cur_shape == "rectangle":
                   h1, h2 = hand_data
                   xs, ys = [h1['thumb_tip'][0], h1['index_tip'][0], h2['thumb_tip'][0], h2['index_tip'][0]], [
                       h1['thumb_tip'][1], h1['index_tip'][1], h2['thumb_tip'][1], h2['index_tip'][1]]
                   add_rect((min(xs), min(ys)), (max(xs), max(ys)), draw_color)
               elif cur_shape == "circle":
                   add_circ(hand_data[0]['center'],
                            int(euclid(hand_data[0]['index_tip'], hand_data[0]['thumb_tip']) * 2.0), draw_color)
               pending_shape, last_shape_time = None, time.time()
       else:
           pending_shape = None

   # Compositing/Rendering
   # Merges the black drawing canvas, the live camera feed, and the vector shapes into one final image.
   gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
   _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
   out = cv2.add(cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask)), cv2.bitwise_and(canvas, canvas, mask=mask))
   render_shapes(out, drawn_shapes)

   # UI & FPS
   # Calculates and draws the FPS counter and current mode (e.g., "MOVING") on the screen.
   new_frame_time = time.time()
   fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
   prev_frame_time = new_frame_time
   cv2.putText(out, f"FPS: {int(fps)}", (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
   status = "MOVING" if is_moving else ("RESIZING" if is_resizing else (
       "DETECTING: " + pending_shape if pending_shape else "ON")) if shape_mode else "OFF"
   cv2.putText(out, f"Shape Mode: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
               (0, 255, 0) if shape_mode else (0, 0, 255), 2)
   cv2.putText(out, "M:Shapes U:Undo R:Redo C:Clear Q:Quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

   # Key Handling
   # Checks for keyboard presses like 'm' (mode), 'u' (undo), 'c' (clear), and 'q' (quit).
   cv2.imshow(window_name, out)
   key = cv2.waitKey(1) & 0xFF
   if key == ord('q'):
       break
   elif key == ord('c'):
       canvas, drawn_shapes, undo_stack, redo_stack = np.zeros_like(frame), [], [], [];
       resizing_shape_idx = None
   elif key == ord('u') and undo_stack:
       redo_stack.append((canvas.copy(), [s.copy() for s in drawn_shapes]));
       canvas, drawn_shapes = undo_stack.pop()
   elif key == ord('r') and redo_stack:
       undo_stack.append((canvas.copy(), [s.copy() for s in drawn_shapes]));
       canvas, drawn_shapes = redo_stack.pop()
   elif key == ord('m'):
       shape_mode, pending_shape, resizing_shape_idx = not shape_mode, None, None

cap.release()
cv2.destroyAllWindows()
