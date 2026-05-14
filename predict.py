#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Live Prediction Script for Hand Gesture Recognition
Uses TFLite model for real-time inference optimized for Raspberry Pi 5
"""

import cv2
import numpy as np
import tensorflow as tf
import json
import os

# Configuration
MODEL_TFLITE = 'gesture_model.tflite'  # Will be created after training with train_model.py
IMG_SIZE = 128

# ROI parameters (same as in collect_data.py)
ROI_SIZE = 300
ROI_OFFSET_X = 0
ROI_OFFSET_Y = -50

LABELS_JSON = 'gesture_labels.json'
DEFAULT_GESTURE_LABELS = ['RIGHT', 'LEFT', 'FORWARD', 'BACK', 'STOP']


class GesturePredictor:
    """Hand gesture predictor using TFLite"""
    
    def __init__(self, model_path):
        """Initialize TFLite interpreter"""
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Model loaded: {model_path}")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Input type: {self.input_details[0]['dtype']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
        print(f"Output type: {self.output_details[0]['dtype']}")
    
    def preprocess(self, image):
        """Preprocess image for model input"""
        # Resize to model input size
        resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_data = np.expand_dims(normalized, axis=0)
        
        return input_data
    
    def predict(self, image):
        """Predict gesture from image"""
        # Preprocess
        input_data = self.preprocess(image)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Apply softmax (if not already applied)
        probabilities = self.softmax(output_data[0])
        
        # Get predicted class
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        return predicted_class, confidence, probabilities
    
    @staticmethod
    def softmax(x):
        """Softmax function"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


def get_roi_coordinates(frame_width, frame_height):
    """Calculate ROI coordinates (centered)"""
    center_x = frame_width // 2 + ROI_OFFSET_X
    center_y = frame_height // 2 + ROI_OFFSET_Y
    
    x1 = max(0, center_x - ROI_SIZE // 2)
    y1 = max(0, center_y - ROI_SIZE // 2)
    x2 = min(frame_width, center_x + ROI_SIZE // 2)
    y2 = min(frame_height, center_y + ROI_SIZE // 2)
    
    return x1, y1, x2, y2


def draw_prediction(frame, gesture, confidence, x1, y1, x2, y2):
    """Draw prediction on frame"""
    # Draw ROI rectangle
    color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255) if confidence > 0.5 else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw prediction text
    text = f"{gesture} ({confidence:.2f})"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    
    # Background rectangle for text
    cv2.rectangle(frame, 
                  (x1, y1 - text_size[1] - 10),
                  (x1 + text_size[0] + 10, y1),
                  color, -1)
    
    # Text
    cv2.putText(frame, text, (x1 + 5, y1 - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame


def load_gesture_labels(num_classes):
    if os.path.exists(LABELS_JSON):
        try:
            with open(LABELS_JSON, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            labels = payload.get('labels', [])
            if isinstance(labels, list) and len(labels) == num_classes:
                return [str(label) for label in labels]
        except (json.JSONDecodeError, OSError):
            pass

    if len(DEFAULT_GESTURE_LABELS) == num_classes:
        return DEFAULT_GESTURE_LABELS

    return [f'class_{i}' for i in range(num_classes)]


def main():
    """Main prediction loop"""
    print("="*60)
    print("HAND GESTURE RECOGNITION - LIVE PREDICTION")
    print("="*60)
    
    # Check if model exists
    import os
    if not os.path.exists(MODEL_TFLITE):
        print(f"Error: Model file '{MODEL_TFLITE}' not found!")
        print("Please run train_model.py first to train and save the model.")
        return
    
    # Load model
    print("\nLoading model...")
    predictor = GesturePredictor(MODEL_TFLITE)
    
    num_classes = int(predictor.output_details[0]['shape'][-1])
    gesture_labels = load_gesture_labels(num_classes)
    print(f"Loaded {len(gesture_labels)} labels: {gesture_labels}")

    # Initialize webcam
    print("\nInitializing webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Get frame dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from webcam")
        cap.release()
        return
    
    frame_height, frame_width = frame.shape[:2]
    
    print("\nPress 'Q' to quit")
    print("="*60)
    
    # FPS calculation
    fps_counter = 0
    fps_start_time = cv2.getTickCount()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Get ROI coordinates
        x1, y1, x2, y2 = get_roi_coordinates(frame_width, frame_height)
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        
        if roi.size > 0:
            # Predict gesture
            predicted_class, confidence, probabilities = predictor.predict(roi)
            gesture = gesture_labels[predicted_class]
            
            # Draw prediction
            frame = draw_prediction(frame, gesture, confidence, x1, y1, x2, y2)
            
            # Draw all probabilities (optional, for debugging)
            y_offset = 30
            for i, (label, prob) in enumerate(zip(gesture_labels, probabilities)):
                color = (0, 255, 0) if i == predicted_class else (255, 255, 255)
                text = f"{label}: {prob:.2f}"
                cv2.putText(frame, text, (10, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            # No ROI available
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
            cv2.putText(frame, "Position hand in ROI", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        
        # Calculate and display FPS
        fps_counter += 1
        if fps_counter >= 30:
            fps_end_time = cv2.getTickCount()
            fps = 30.0 / ((fps_end_time - fps_start_time) / cv2.getTickFrequency())
            fps_counter = 0
            fps_start_time = fps_end_time
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame_width - 100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Hand Gesture Recognition', frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("\nPrediction stopped.")


if __name__ == '__main__':
    main()

