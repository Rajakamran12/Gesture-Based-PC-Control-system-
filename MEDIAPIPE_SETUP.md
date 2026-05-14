# MediaPipe Setup Guide

## ✅ **MediaPipe is Now Working!**

MediaPipe has been successfully installed in a Python 3.12 virtual environment.

---

## 🚀 **How to Run MediaPipe Scripts**

### **Option 1: Using Helper Scripts (Easiest)**

**For app.py:**
```bash
run_mediapipe.bat
```

**For test.py:**
```bash
run_mediapipe_test.bat
```

### **Option 2: Manual Activation**

1. **Activate the virtual environment:**
   ```bash
   .\venv_mediapipe\Scripts\activate
   ```

2. **Run the script:**
   ```bash
   python app.py
   # or
   python test.py
   ```

3. **Deactivate when done:**
   ```bash
   deactivate
   ```

### **Option 3: Direct Python Path**

```bash
.\venv_mediapipe\Scripts\python.exe app.py
```

---

## 📋 **What's Installed**

- **Python Version:** 3.12.10 (in virtual environment)
- **MediaPipe:** 0.10.14 ✅
- **OpenCV:** 4.12.0.88 ✅
- **TensorFlow:** 2.19.1 ✅
- **NumPy:** 2.1.3 ✅

---

## 🎯 **Available Scripts**

### **1. app.py - MediaPipe Gesture Recognition**
- Uses pre-trained TFLite models
- Detects hand landmarks (21 points)
- Classifies gestures: Forward, Backward, Right, Left, Stop
- Real-time webcam feed

**Run:**
```bash
.\venv_mediapipe\Scripts\python.exe app.py
```

**Controls:**
- `ESC` - Exit
- `0-9` - Select gesture number (for data collection)
- `N` - Normal mode
- `K` - Logging Key Point mode
- `H` - Logging Point History mode

### **2. test.py - Raspberry Pi Robot Control**
- Same gesture recognition as app.py
- Controls robot hardware (servo, motors)
- **Note:** Hardware control will only work on Raspberry Pi

**Run:**
```bash
.\venv_mediapipe\Scripts\python.exe test.py
```

**Modes:**
1. MediaPipe AI Camera Drive
2. Manual Drive
3. Manual Image → AI → Movement

---

## 🔄 **Switching Between Approaches**

### **CNN Approach (Python 3.13)**
```bash
# Use your default Python (3.13)
python collect_data.py
python train_model.py
python predict.py
```

### **MediaPipe Approach (Python 3.12)**
```bash
# Use the virtual environment
.\venv_mediapipe\Scripts\python.exe app.py
```

---

## 📁 **Virtual Environment Location**

- **Path:** `venv_mediapipe/`
- **Python:** 3.12.10
- **Activation:** `.\venv_mediapipe\Scripts\activate`

---

## 🛠️ **Troubleshooting**

### **If app.py doesn't start:**
1. Make sure virtual environment is activated
2. Check camera is available
3. Verify TFLite models exist in `model/` folder

### **If MediaPipe import fails:**
```bash
# Reinstall MediaPipe
.\venv_mediapipe\Scripts\python.exe -m pip install --upgrade mediapipe
```

### **To update packages:**
```bash
.\venv_mediapipe\Scripts\python.exe -m pip install --upgrade opencv-python numpy mediapipe tensorflow
```

---

## ✅ **Status Summary**

| Approach | Status | Python Version | Scripts |
|----------|--------|----------------|---------|
| **CNN-based** | ✅ Working | 3.13.1 | collect_data.py, train_model.py, predict.py |
| **MediaPipe-based** | ✅ Working | 3.12.10 (venv) | app.py, test.py |

**Both approaches are now fully functional!** 🎉

