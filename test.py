#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import platform
import csv
import copy
import itertools
from collections import Counter, deque

import numpy as np
import cv2 as cv
import mediapipe as mp

# Hardware imports - make optional for cross-platform compatibility
HAS_GPIO = False
HAS_PICAMERA = False

try:
    from gpiozero import AngularServo, PWMOutputDevice, OutputDevice
    HAS_GPIO = True
except ImportError:
    print("⚠️  GPIO not available (not on Raspberry Pi) - Hardware control disabled")
    # Create dummy classes for local testing
    class AngularServo:
        def __init__(self, *args, **kwargs):
            self.angle = None
            self.min_angle = 0
            self.max_angle = 180
    class PWMOutputDevice:
        def __init__(self, *args, **kwargs):
            pass
        def on(self): pass
        def off(self): pass
        @property
        def value(self): return None
        @value.setter
        def value(self, v): pass
    class OutputDevice:
        def __init__(self, *args, **kwargs):
            pass
        def on(self): pass
        def off(self): pass

try:
    from picamera2 import Picamera2
    HAS_PICAMERA = True
except ImportError:
    print("⚠️  Picamera2 not available - Using regular webcam")

from tensorflow.keras.models import load_model  # still used by image-upload mode
from PIL import Image

try:
    from pynput import keyboard  # optional
except ImportError:
    pass  # pynput is optional

from utils import CvFpsCalc
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.point_history_classifier.point_history_classifier import PointHistoryClassifier

# --------------------------------------------------------------------
# Hardware setup (works on both Pi and local systems)
# --------------------------------------------------------------------
MODEL_PATH = "fyp.h5"  # still used for image-upload mode
if os.path.exists(MODEL_PATH):
    try:
        upload_model = load_model(MODEL_PATH)
        print("✅ Image-upload model loaded.")
    except Exception as e:
        print(f"⚠️ Could not load fyp.h5: {e}")
        upload_model = None
else:
    upload_model = None
    print("⚠️ fyp.h5 not found; image-upload mode will be disabled.")

class_names = ['back', 'front', 'left', 'right', 'stop']

def preprocess(img_path):
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_upload_gesture(img_path):
    if upload_model is None:
        raise RuntimeError("Upload model not available.")
    data = preprocess(img_path)
    preds = upload_model.predict(data)
    class_id = int(np.argmax(preds[0]))
    label = class_names[class_id]
    conf = float(np.max(preds[0]))
    print(f"🤖 Upload prediction: {label.upper()} ({conf:.2f})")
    return label

# Hardware setup (only if GPIO available)
if HAS_GPIO:
    servo = AngularServo(
        19,
        min_angle=0,
        max_angle=180,
        min_pulse_width=0.5/1000,
        max_pulse_width=2.5/1000,
        frame_width=20/1000
    )
    ENA = PWMOutputDevice(13)
    IN1 = OutputDevice(17)
    IN2 = OutputDevice(18)
    CENTER_ANGLE = 90
    servo.angle = CENTER_ANGLE
else:
    # Dummy hardware objects for local testing
    servo = AngularServo(19)
    ENA = PWMOutputDevice(13)
    IN1 = OutputDevice(17)
    IN2 = OutputDevice(18)
    CENTER_ANGLE = 90

def move_servo(target_angle, step=10, delay=0.05):
    if not HAS_GPIO or servo is None:
        print(f"🔧 [SIMULATED] Servo moving to {target_angle}°")
        return
    current = servo.angle if servo.angle is not None else CENTER_ANGLE
    seq = range(int(current), int(target_angle) + (1 if target_angle > current else -1), step if target_angle > current else -step)
    for angle in seq:
        servo.angle = max(servo.min_angle, min(servo.max_angle, angle))
        time.sleep(delay)

def rotate_left():
    target = CENTER_ANGLE - 100
    print("⬅️ Rotating left")
    move_servo(target)
    time.sleep(2)
    move_servo(CENTER_ANGLE)

def rotate_right():
    target = CENTER_ANGLE + 100
    print("➡️ Rotating right")
    move_servo(target)
    time.sleep(2)
    move_servo(CENTER_ANGLE)

def forward(speed=0.8):
    if HAS_GPIO and IN1 is not None:
        IN1.on(); IN2.off(); ENA.value = speed
    print("🚗 Forward")

def backward(speed=0.8):
    if HAS_GPIO and IN1 is not None:
        IN1.off(); IN2.on(); ENA.value = speed
    print("↩️ Backward")

def stop():
    if HAS_GPIO and IN1 is not None:
        IN1.off(); IN2.off(); ENA.off()
    print("🛑 Stopped")


def open_camera_with_fallbacks(preferred_index=0, width=640, height=480):
    """Try common camera backends/indexes so webcam startup is reliable on Windows."""
    indexes = [preferred_index] + [idx for idx in [0, 1, 2] if idx != preferred_index]
    system_name = platform.system().lower()

    backend_options = [None]
    if system_name == 'windows':
        backend_options = [cv.CAP_DSHOW, cv.CAP_MSMF, None]

    for cam_index in indexes:
        for backend in backend_options:
            if backend is None:
                cap = cv.VideoCapture(cam_index)
                backend_name = 'default'
            else:
                cap = cv.VideoCapture(cam_index, backend)
                backend_name = 'CAP_DSHOW' if backend == cv.CAP_DSHOW else 'CAP_MSMF'

            if cap.isOpened():
                cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
                print(f"✅ Camera opened on index {cam_index} using {backend_name} backend")
                return cap

            cap.release()

    return None

# --------------------------------------------------------------------
# MediaPipe/app.py-based AI camera mode
# --------------------------------------------------------------------

def load_labels(path):
    with open(path, encoding='utf-8-sig') as f:
        return [row[0].strip() for row in csv.reader(f) if row]

keypoint_labels = load_labels('model/keypoint_classifier/keypoint_classifier_label.csv')
point_history_labels = load_labels('model/point_history_classifier/point_history_classifier_label.csv')

STOP_IDX_KEY = next((i for i, lab in enumerate(keypoint_labels) if lab.lower() == 'stop'), len(keypoint_labels)-1)
STOP_IDX_POINT = next((i for i, lab in enumerate(point_history_labels) if lab.lower() == 'stop'), len(point_history_labels)-1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()
cvFpsCalc = CvFpsCalc(buffer_len=10)

history_length = 16
point_history = deque(maxlen=history_length)
finger_gesture_history = deque(maxlen=history_length)

def detect_palm_side(hand_landmarks):
    """
    Detect if palm is facing camera (front) or back of hand is facing camera.
    Uses z-coordinates: palm facing camera has positive z for fingertips.
    Returns: True if palm facing camera, False if back of hand
    """
    # Get z-coordinates of key points
    # Wrist (0), Middle finger MCP (9), Index finger tip (8)
    wrist_z = hand_landmarks.landmark[0].z
    middle_mcp_z = hand_landmarks.landmark[9].z
    index_tip_z = hand_landmarks.landmark[8].z
    
    # If fingertips are closer to camera (more positive z) than wrist, palm is facing camera
    # For palm facing camera, fingertips should have higher z values
    palm_facing = index_tip_z > wrist_z
    
    return palm_facing

def count_fingers(landmark_points, is_right_hand=True, hand_landmarks=None):
    """
    Count the number of extended fingers based on MediaPipe landmarks.
    Args:
        landmark_points: List of 21 landmark points [x, y]
        is_right_hand: True if right hand, False if left hand
        hand_landmarks: MediaPipe hand landmarks object (for z-coordinates)
    Returns: (finger_count, is_open_hand, is_closed_hand, is_two_fingers, 
             all_fingers_point_right, all_fingers_point_left)
    """
    if len(landmark_points) < 21:
        return 0, False, False, False, False, False
    
    finger_count = 0
    fingers_up = []
    finger_tips = []
    
    # Detect palm side for better finger detection
    palm_facing = True
    if hand_landmarks is not None:
        palm_facing = detect_palm_side(hand_landmarks)
    
    # Thumb detection (depends on hand orientation and palm side)
    if palm_facing:
        # Palm facing camera: check horizontal position
        if is_right_hand:
            thumb_up = landmark_points[4][0] > landmark_points[3][0]
        else:
            thumb_up = landmark_points[4][0] < landmark_points[3][0]
    else:
        # Back of hand: reverse the logic
        if is_right_hand:
            thumb_up = landmark_points[4][0] < landmark_points[3][0]
        else:
            thumb_up = landmark_points[4][0] > landmark_points[3][0]
    
    if thumb_up:
        finger_count += 1
        fingers_up.append(True)
        finger_tips.append(landmark_points[4])
    else:
        fingers_up.append(False)
    
    # Index finger (check if tip is above PIP joint)
    if landmark_points[8][1] < landmark_points[6][1]:
        finger_count += 1
        fingers_up.append(True)
        finger_tips.append(landmark_points[8])
    else:
        fingers_up.append(False)
    
    # Middle finger (check if tip is above PIP joint)
    if landmark_points[12][1] < landmark_points[10][1]:
        finger_count += 1
        fingers_up.append(True)
        finger_tips.append(landmark_points[12])
    else:
        fingers_up.append(False)
    
    # Ring finger (check if tip is above PIP joint)
    if landmark_points[16][1] < landmark_points[14][1]:
        finger_count += 1
        fingers_up.append(True)
        finger_tips.append(landmark_points[16])
    else:
        fingers_up.append(False)
    
    # Little finger (check if tip is above PIP joint)
    if landmark_points[20][1] < landmark_points[18][1]:
        finger_count += 1
        fingers_up.append(True)
        finger_tips.append(landmark_points[20])
    else:
        fingers_up.append(False)
    
    # Determine gesture types
    is_open_hand = finger_count == 5
    is_closed_hand = finger_count == 0  # All fingers down = punch/closed fist
    is_two_fingers = finger_count == 2
    
    # Check if all extended fingers are pointing in the same direction (left/right)
    all_fingers_point_right = False
    all_fingers_point_left = False
    
    if finger_count >= 3 and len(finger_tips) >= 3:  # At least 3 fingers extended
        # Get wrist position as reference
        wrist_x = landmark_points[0][0]
        
        # Check if all finger tips are to the right of wrist
        all_right = all(tip[0] > wrist_x + 30 for tip in finger_tips)  # 30px threshold
        # Check if all finger tips are to the left of wrist
        all_left = all(tip[0] < wrist_x - 30 for tip in finger_tips)  # 30px threshold
        
        if all_right:
            all_fingers_point_right = True
        elif all_left:
            all_fingers_point_left = True
    
    return finger_count, is_open_hand, is_closed_hand, is_two_fingers, all_fingers_point_right, all_fingers_point_left

def interpret_gesture(label, finger_count=None, is_open_hand=False, is_closed_hand=False, 
                      is_two_fingers=False, all_fingers_point_right=False, all_fingers_point_left=False):
    """Map app labels to robot actions with improved gesture detection."""
    # Priority 1: Check if all fingers pointing in a direction
    if all_fingers_point_right:
        # All fingers pointing right → turn right
        rotate_right(); stop()
        return
    elif all_fingers_point_left:
        # All fingers pointing left → turn left
        rotate_left(); stop()
        return
    
    # Priority 2: Check finger-based gestures
    if is_two_fingers:
        # Two fingers (peace sign) = stop
        stop(); move_servo(CENTER_ANGLE)
        return
    
    if is_open_hand:
        # Open hand (fingers open/release) = stop
        stop(); move_servo(CENTER_ANGLE)
        return
    
    if is_closed_hand:
        # Closed hand (punch) = move forward
        forward(); move_servo(CENTER_ANGLE)
        return
    
    # Priority 3: Check label-based gestures (fallback)
    if label.lower() == 'forward':
        forward(); move_servo(CENTER_ANGLE)
    elif label.lower() == 'back' or label.lower() == 'backward':
        backward(); move_servo(CENTER_ANGLE)
    elif label.lower() == 'left':
        rotate_left(); stop()
    elif label.lower() == 'right':
        rotate_right(); stop()
    else:  # stop or unknown
        stop(); move_servo(CENTER_ANGLE)

def ai_camera_mode():
    print("\n🤖 Starting MediaPipe AI Camera Mode (Ctrl+C to exit)")
    print(f"⚠️  Hardware control is {'ENABLED (Raspberry Pi)' if HAS_GPIO else 'SIMULATED (Local Testing)'}")
    cap = open_camera_with_fallbacks(preferred_index=0, width=640, height=480)
    if cap is None:
        print("❌ Could not open any camera. Close other apps using webcam and try again.")
        return

    current_label = keypoint_labels[STOP_IDX_KEY] if STOP_IDX_KEY < len(keypoint_labels) else "Unknown"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Camera read failed")
                break

            frame = cv.flip(frame, 1)
            dbg = copy.deepcopy(frame)

            fps = cvFpsCalc.get()
            cv.putText(dbg, f"FPS:{fps:.2f}", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)

            image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True

            current_label = keypoint_labels[STOP_IDX_KEY] if STOP_IDX_KEY < len(keypoint_labels) else "Stop"
            
            # Initialize finger detection variables
            finger_count = 0
            is_open_hand = False
            is_closed_hand = False
            is_two_fingers = False
            all_fingers_point_right = False
            all_fingers_point_left = False
            hand_detected = False

            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                     results.multi_handedness):
                    # Get hand orientation
                    is_right_hand = handedness.classification[0].label == 'Right'
                    # Bounding rect + landmarks
                    image_width, image_height = dbg.shape[1], dbg.shape[0]
                    landmark_array = np.empty((0, 2), int)
                    landmark_points = []

                    for idx, lm in enumerate(hand_landmarks.landmark):
                        lx = min(int(lm.x * image_width), image_width - 1)
                        ly = min(int(lm.y * image_height), image_height - 1)
                        landmark_points.append([lx, ly])
                        landmark_array = np.append(landmark_array, np.array([[lx, ly]]), axis=0)

                    x, y, w, h = cv.boundingRect(landmark_array)
                    cv.rectangle(dbg, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    # Count fingers with palm detection
                    finger_count, is_open_hand, is_closed_hand, is_two_fingers, all_fingers_point_right, all_fingers_point_left = count_fingers(landmark_points, is_right_hand, hand_landmarks)

                    temp_landmarks = copy.deepcopy(landmark_points)
                    base_x, base_y = temp_landmarks[0][0], temp_landmarks[0][1]
                    for idx in range(len(temp_landmarks)):
                        temp_landmarks[idx][0] -= base_x
                        temp_landmarks[idx][1] -= base_y
                    temp_landmarks = list(itertools.chain.from_iterable(temp_landmarks))
                    max_value = max(list(map(abs, temp_landmarks))) or 1
                    temp_landmarks = [val / max_value for val in temp_landmarks]

                    pre_processed_point_history = []
                    point_history.append(landmark_points[8])
                    if len(point_history) < history_length:
                        point_history.append([0, 0])
                    for point in point_history:
                        pre_processed_point_history.extend([(point[0] - base_x) / image_width,
                                                            (point[1] - base_y) / image_height])

                    hand_sign_id = keypoint_classifier(temp_landmarks)
                    if hand_sign_id < len(keypoint_labels):
                        current_label = keypoint_labels[hand_sign_id]
                    else:
                        current_label = "Unknown"

                    finger_gesture_id = 0
                    if len(pre_processed_point_history) == history_length * 2:
                        finger_gesture_id = point_history_classifier(pre_processed_point_history)

                    finger_gesture_history.append(finger_gesture_id)

            else:
                # No hand detected - should stop
                hand_detected = False
                point_history.append([0, 0])
                finger_gesture_history.append(STOP_IDX_POINT)
                current_label = "No Hand"

            # Display gesture information
            gesture_text = f"Gesture: {current_label}"
            if hand_detected:
                gesture_text += f" | Fingers: {finger_count}"
                if all_fingers_point_right:
                    gesture_text += " (ALL RIGHT → TURN RIGHT)"
                elif all_fingers_point_left:
                    gesture_text += " (ALL LEFT → TURN LEFT)"
                elif is_two_fingers:
                    gesture_text += " (TWO - STOP)"
                elif is_open_hand:
                    gesture_text += " (OPEN - STOP)"
                elif is_closed_hand:
                    gesture_text += " (PUNCH - FORWARD)"
            
            cv.putText(dbg, gesture_text,
                       (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)
            cv.imshow("MediaPipe Gesture Drive", dbg)

            # Interpret gesture with finger detection
            interpret_gesture(current_label, finger_count, is_open_hand, is_closed_hand, 
                           is_two_fingers, all_fingers_point_right, all_fingers_point_left)

            if cv.waitKey(10) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n🛑 Interrupted AI camera mode")
    finally:
        stop()
        move_servo(CENTER_ANGLE)
        cap.release()
        cv.destroyAllWindows()

# --------------------------------------------------------------------
# Keep your other modes (manual drive, image upload) if desired
# --------------------------------------------------------------------

def manual_image_prediction():
    if upload_model is None:
        print("❌ Image upload model not available.")
        return
    print("\n🖼️ IMAGE UPLOAD MODE (type EXIT)")
    try:
        while True:
            path = input("Image path: ").strip()
            if path.lower() == "exit":
                break
            if not os.path.exists(path):
                print("❌ File not found"); continue
            gesture = predict_upload_gesture(path)
            interpret_gesture(gesture)
    finally:
        stop(); move_servo(CENTER_ANGLE)

def manual_drive():
    print("\n🎮 MANUAL DRIVE MODE (W/S/A/D/X, Q to quit)")
    print(f"⚠️  Hardware control is {'ENABLED' if HAS_GPIO else 'SIMULATED'}")
    while True:
        cmd = input("Command: ").strip().lower()
        if cmd == 'w': forward()
        elif cmd == 's': backward()
        elif cmd == 'a': rotate_left(); stop()
        elif cmd == 'd': rotate_right(); stop()
        elif cmd == 'x': stop()
        elif cmd == 'q': break
        else: print("❌ Invalid command")

def main_menu():
    print("\n" + "="*60)
    print("HAND GESTURE ROBOT CONTROL")
    print("="*60)
    print(f"Hardware Status: {'✅ ENABLED (Raspberry Pi)' if HAS_GPIO else '⚠️  SIMULATED (Local Testing)'}")
    print("="*60)
    print("\n📌 Select mode:")
    print("1) MediaPipe AI Camera Drive")
    print("2) Manual Drive")
    print("3) Manual Image → AI → Movement")
    mode = input("\nEnter mode (1/2/3): ").strip()
    if mode == "1": ai_camera_mode()
    elif mode == "2": manual_drive()
    elif mode == "3": manual_image_prediction()
    else: print("❌ Invalid mode")

if __name__ == "__main__":
    try:
        main_menu()
    finally:
        stop()
        move_servo(CENTER_ANGLE)