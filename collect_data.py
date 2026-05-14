#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset Collection Script for Hand Gesture Recognition
Collects images from webcam with ROI and stores them in labeled folders
"""

import cv2
import os
import numpy as np

# Gesture classes mapping
GESTURE_CLASSES = {
    'right_finger': 'RIGHT',
    'left_finger': 'LEFT',
    'open_hand': 'FORWARD',
    'close_hand': 'BACK',
    'two_finger': 'STOP'
}

# Dataset directory
DATASET_DIR = 'dataset'
MIN_IMAGES_PER_CLASS = 500

# ROI parameters (center of screen)
ROI_SIZE = 300  # Size of the ROI box
ROI_OFFSET_X = 0  # Offset from center X
ROI_OFFSET_Y = -50  # Offset from center Y


def create_directories():
    """Create dataset directories if they don't exist"""
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    
    for class_name in GESTURE_CLASSES.keys():
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            print(f"Created directory: {class_dir}")
    
    print("All directories ready!")


def get_roi_coordinates(frame_width, frame_height):
    """Calculate ROI coordinates (centered)"""
    center_x = frame_width // 2 + ROI_OFFSET_X
    center_y = frame_height // 2 + ROI_OFFSET_Y
    
    x1 = max(0, center_x - ROI_SIZE // 2)
    y1 = max(0, center_y - ROI_SIZE // 2)
    x2 = min(frame_width, center_x + ROI_SIZE // 2)
    y2 = min(frame_height, center_y + ROI_SIZE // 2)
    
    return x1, y1, x2, y2


def count_images_in_class(class_name):
    """Count existing images in a class folder"""
    class_dir = os.path.join(DATASET_DIR, class_name)
    if os.path.exists(class_dir):
        return len([f for f in os.listdir(class_dir) if f.endswith('.jpg')])
    return 0


def main():
    """Main function for data collection"""
    # Create directories
    create_directories()
    
    # Initialize webcam
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
    
    # Current class selection
    current_class_idx = 0
    class_names = list(GESTURE_CLASSES.keys())
    current_class = class_names[current_class_idx]
    
    print("\n" + "="*60)
    print("HAND GESTURE DATASET COLLECTOR")
    print("="*60)
    print("\nControls:")
    print("  SPACE - Capture image")
    print("  N - Next class")
    print("  P - Previous class")
    print("  Q - Quit")
    print("\n" + "="*60)
    
    image_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Get ROI coordinates
        x1, y1, x2, y2 = get_roi_coordinates(frame_width, frame_height)
        
        # Draw ROI rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw instructions
        cv2.putText(frame, f"Class: {current_class} ({GESTURE_CLASSES[current_class]})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        count = count_images_in_class(current_class)
        cv2.putText(frame, f"Images: {count}/{MIN_IMAGES_PER_CLASS}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, "SPACE: Capture | N: Next | P: Prev | Q: Quit", 
                   (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('Dataset Collection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space to capture
            # Extract ROI
            roi = frame[y1:y2, x1:x2]
            
            if roi.size > 0:
                # Resize to 128x128 (model input size)
                roi_resized = cv2.resize(roi, (128, 128))
                
                # Generate filename
                count = count_images_in_class(current_class)
                filename = f"{current_class}_{count:05d}.jpg"
                filepath = os.path.join(DATASET_DIR, current_class, filename)
                
                # Save image
                cv2.imwrite(filepath, roi_resized)
                image_count += 1
                print(f"Saved: {filepath} (Total: {image_count})")
                
                # Show confirmation
                cv2.putText(frame, "CAPTURED!", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Dataset Collection', frame)
                cv2.waitKey(100)  # Brief flash
        elif key == ord('n'):  # Next class
            current_class_idx = (current_class_idx + 1) % len(class_names)
            current_class = class_names[current_class_idx]
            print(f"Switched to class: {current_class}")
        elif key == ord('p'):  # Previous class
            current_class_idx = (current_class_idx - 1) % len(class_names)
            current_class = class_names[current_class_idx]
            print(f"Switched to class: {current_class}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "="*60)
    print("COLLECTION SUMMARY")
    print("="*60)
    for class_name in class_names:
        count = count_images_in_class(class_name)
        print(f"{class_name:15s}: {count:4d} images")
    print("="*60)
    print(f"Total images collected: {image_count}")
    print("\nCollection complete!")


if __name__ == '__main__':
    main()

