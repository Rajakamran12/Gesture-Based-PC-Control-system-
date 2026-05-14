# Hand Gesture Recognition AI

A comprehensive hand gesture recognition system with two approaches: **MediaPipe-based** (pre-trained) and **CNN-based** (custom training). Optimized for both local development and Raspberry Pi deployment.

## 🎯 Features

- **Dual Approach**: MediaPipe (pre-trained) and CNN (custom training)
- **Real-time Recognition**: Live webcam gesture detection
- **5 Gesture Classes**: RIGHT, LEFT, FORWARD, BACK, STOP
- **Raspberry Pi Ready**: Includes robot control integration
- **Easy to Use**: Simple scripts for data collection, training, and prediction

## 📋 Project Structure

```
Hand-Gesture-Recognition-AI/
├── app.py                      # MediaPipe-based gesture recognition
├── test.py                     # Raspberry Pi robot control (MediaPipe)
├── test_local.py               # Local testing version (Windows compatible)
├── collect_data.py             # Dataset collection tool
├── train_model.py              # CNN model training
├── predict.py                  # CNN-based live prediction
├── model/                      # Pre-trained models
│   ├── keypoint_classifier/
│   └── point_history_classifier/
├── utils/                      # Utility functions
└── dataset/                    # Training data (create with collect_data.py)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or 3.12 (for MediaPipe)
- Python 3.13 (for CNN approach)
- Webcam
- OpenCV, NumPy, TensorFlow

### Installation

```bash
# Clone the repository
git clone https://github.com/mubashir-ullah/Hand-Gesture-Recognition-AI.git
cd Hand-Gesture-Recognition-AI

# Install dependencies
pip install -r requirements.txt

# For MediaPipe (Python 3.11/3.12)
python -m venv venv_mediapipe
venv_mediapipe\Scripts\activate  # Windows
pip install opencv-python numpy mediapipe tensorflow
```

## 📖 Usage

### Approach 1: MediaPipe (Pre-trained)

**Local Testing:**
```bash
# Windows
run_test_local.bat
# Or
.\venv_mediapipe\Scripts\python.exe test_local.py
```

**Raspberry Pi:**
```bash
python test.py
```

### Approach 2: CNN (Custom Training)

**Step 1: Collect Dataset**
```bash
python collect_data.py
```
- Press SPACE to capture images
- Collect 100-200 images per class
- Use N/P to switch classes

**Step 2: Train Model**
```bash
python train_model.py
```

**Step 3: Test Prediction**
```bash
python predict.py
```

## 🎮 Gesture Classes

| Gesture | Label | Description |
|---------|-------|-------------|
| Point Right | RIGHT | Pointing right with finger |
| Point Left | LEFT | Pointing left with finger |
| Open Hand | FORWARD | Open hand gesture |
| Closed Fist | BACK | Closed fist |
| Two Fingers | STOP | Two fingers (peace sign) |

## 🔧 Configuration

### MediaPipe Approach
- Uses pre-trained TFLite models in `model/` folder
- Detects 21 hand landmarks
- Real-time processing

### CNN Approach
- Custom training on your dataset
- Lightweight model (~200K parameters)
- Optimized for edge devices

## 📚 Documentation

- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Complete testing instructions
- [MEDIAPIPE_SETUP.md](MEDIAPIPE_SETUP.md) - MediaPipe setup guide
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [CODE_REVIEW_FIXES.md](CODE_REVIEW_FIXES.md) - Code review and fixes

## 🤖 Raspberry Pi Integration

The `test.py` script includes robot control:
- Servo motor control (GPIO 19)
- DC motor control (GPIO 13, 17, 18)
- Gesture-based robot movement

## 📝 Requirements

See [requirements.txt](requirements.txt) for full dependency list.

## 📄 License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

## 👤 Author

**Mubashir Ullah**

## 🙏 Acknowledgments

- MediaPipe for hand detection
- TensorFlow for model training
- OpenCV for computer vision

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Repository**: [https://github.com/mubashir-ullah/Hand-Gesture-Recognition-AI](https://github.com/mubashir-ullah/Hand-Gesture-Recognition-AI)

