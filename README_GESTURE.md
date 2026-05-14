# Hand Gesture Recognition System

A lightweight hand gesture recognition system optimized for Raspberry Pi 5, using OpenCV and TensorFlow Lite.

## Features

- **Real-time gesture recognition** from webcam
- **Lightweight CNN model** optimized for edge devices
- **TensorFlow Lite** for fast inference
- **5 gesture classes**: RIGHT, LEFT, FORWARD, BACK, STOP
- **Easy dataset collection** with ROI visualization
- **Data augmentation** for better model generalization

## Gesture Classes

| Folder Name | Gesture Label | Description |
|------------|---------------|-------------|
| `right_finger` | RIGHT | Pointing right with finger |
| `left_finger` | LEFT | Pointing left with finger |
| `open_hand` | FORWARD | Open hand gesture |
| `close_hand` | BACK | Closed fist |
| `two_finger` | STOP | Two fingers (peace sign) |

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**For Raspberry Pi 5 (inference only):**
```bash
# Use TensorFlow Lite runtime instead of full TensorFlow for smaller size
pip install opencv-python numpy tensorflow-lite-runtime scikit-learn matplotlib seaborn
```

### 2. Verify Installation

```bash
python -c "import cv2; import tensorflow as tf; print('All dependencies installed!')"
```

## Usage

### Step 1: Collect Dataset

Run the data collection script to capture images from your webcam:

```bash
python collect_data.py
```

**Controls:**
- **SPACE** - Capture image
- **N** - Next class
- **P** - Previous class
- **Q** - Quit

**Instructions:**
1. Position your hand inside the green ROI box
2. Press SPACE to capture images
3. Collect at least 500 images per class (recommended)
4. Switch between classes using N/P keys
5. Press Q when done

The script will automatically create the following folder structure:
```
dataset/
├── right_finger/
├── left_finger/
├── open_hand/
├── close_hand/
└── two_finger/
```

### Step 2: Train Model

Train the CNN model on your collected dataset:

```bash
python train_model.py
```

**What it does:**
- Loads images from `dataset/` folder
- Applies data augmentation (rotation, zoom, flip, brightness)
- Splits data into 80% training / 20% validation
- Trains a lightweight CNN model
- Saves model as `model.h5` (Keras) and `gesture_model.tflite` (TFLite)
- Generates confusion matrix and training history plots
- Prints classification report with accuracy metrics

**Training Output:**
- `model.h5` - Keras model (for further training/fine-tuning)
- `gesture_model.tflite` - TensorFlow Lite model (for inference)
- `confusion_matrix.png` - Confusion matrix visualization
- `training_history.png` - Training/validation accuracy and loss curves

**Model Architecture:**
- Custom lightweight CNN with ~200K parameters
- Alternative: MobileNetV2 (commented in code, uncomment to use)
- Input: 128x128 RGB images
- Output: 5 gesture classes with softmax probabilities

### Step 3: Live Prediction

Run real-time gesture recognition:

```bash
python predict.py
```

**Features:**
- Real-time webcam feed
- ROI visualization (green box)
- Gesture prediction with confidence score
- FPS counter
- All class probabilities displayed
- Press **Q** to quit

## Project Structure

```
project/
├── collect_data.py          # Dataset collection script
├── train_model.py           # Model training script
├── predict.py               # Live prediction script
├── requirements.txt         # Python dependencies
├── model.h5                 # Trained Keras model (generated)
├── gesture_model.tflite     # TensorFlow Lite model (generated)
├── confusion_matrix.png     # Confusion matrix (generated)
├── training_history.png     # Training curves (generated)
└── dataset/                 # Dataset folder (created)
    ├── right_finger/
    ├── left_finger/
    ├── open_hand/
    ├── close_hand/
    └── two_finger/
```

## Model Optimization

### For Raspberry Pi 5

The model is optimized for edge devices:

1. **Lightweight Architecture**: Custom CNN with only ~200K parameters
2. **Quantization**: TFLite model uses int8 quantization (4x smaller)
3. **Small Input Size**: 128x128 images (fast preprocessing)
4. **Efficient Operations**: GlobalAveragePooling instead of Flatten+Dense

### Model Size Comparison

- **Keras Model (float32)**: ~800 KB
- **TFLite (float32)**: ~800 KB
- **TFLite (int8 quantized)**: ~200 KB ⚡

### Performance Tips

1. **Use TensorFlow Lite Runtime** instead of full TensorFlow for inference
2. **Reduce input size** to 96x96 if needed (modify `IMG_SIZE` in code)
3. **Use MobileNetV2** for better accuracy (slightly larger model)
4. **Adjust ROI size** based on your camera distance

## Troubleshooting

### Webcam not opening
- Check camera permissions
- Try different camera index: `cv2.VideoCapture(1)` instead of `0`
- On Linux: `sudo usermod -a -G video $USER` (logout/login)

### Low accuracy
- Collect more training data (aim for 500+ images per class)
- Ensure good lighting conditions
- Vary hand positions and angles during collection
- Check if classes are balanced

### Slow inference on Raspberry Pi
- Use TensorFlow Lite Runtime instead of full TensorFlow
- Reduce input image size (96x96 or 64x64)
- Close other applications to free up resources
- Use a USB 3.0 webcam for better performance

### Out of memory during training
- Reduce `BATCH_SIZE` in `train_model.py`
- Use smaller `IMG_SIZE` (96 instead of 128)
- Train on a subset of data first

## Customization

### Add More Gesture Classes

1. Add new folder in `dataset/` (e.g., `thumbs_up/`)
2. Update `CLASS_NAMES` and `GESTURE_LABELS` in `train_model.py` and `predict.py`
3. Update `NUM_CLASSES` in `train_model.py`
4. Collect images for new class
5. Retrain model

### Change Model Architecture

In `train_model.py`, you can:
- Use MobileNetV2: Uncomment `create_mobilenet_model()` function
- Modify CNN layers: Edit `create_lightweight_model()` function
- Adjust input size: Change `IMG_SIZE` (must match in all scripts)

### Adjust ROI Position

Modify `ROI_OFFSET_X` and `ROI_OFFSET_Y` in `collect_data.py` and `predict.py` to change ROI position.

## Performance Metrics

Expected performance on Raspberry Pi 5:
- **Inference Speed**: 15-30 FPS (depending on model and input size)
- **Model Size**: ~200 KB (quantized)
- **Memory Usage**: ~100-200 MB
- **Accuracy**: 85-95% (depends on dataset quality)

## License

This project is provided as-is for educational and research purposes.

## Credits

Optimized for Raspberry Pi 5 with focus on:
- Minimal dependencies
- Fast inference
- Small model size
- Easy deployment

