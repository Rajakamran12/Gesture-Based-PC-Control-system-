# Testing Guide - Hand Gesture Recognition

## ✅ **CURRENT STATUS (Python 3.13)**

### **What's Ready to Test NOW:**

1. ✅ **CNN Approach (Full Workflow)**
   - `collect_data.py` - Data collection ✅ Ready
   - `train_model.py` - Model training ✅ Ready  
   - `predict.py` - Live prediction ✅ Ready (fixed path)

2. ✅ **All Required Packages Installed:**
   - OpenCV ✅
   - NumPy ✅
   - TensorFlow ✅
   - scikit-learn ✅
   - matplotlib ✅
   - seaborn ✅

3. ✅ **Camera Available** ✅

### **What DOESN'T Work (Python 3.13 Issue):**

❌ **MediaPipe** - Not available for Python 3.13
- `app.py` - Requires MediaPipe ❌
- `test.py` - Requires MediaPipe ❌

---

## 🚀 **HOW TO TEST (CNN Approach)**

### **Step 1: Collect Dataset**

```bash
python collect_data.py
```

**Controls:**
- `SPACE` - Capture image
- `N` - Next gesture class
- `P` - Previous gesture class  
- `Q` - Quit

**Instructions:**
1. Position hand inside green ROI box
2. Press SPACE to capture
3. Collect at least 100-200 images per class (minimum for testing)
4. 5 classes total: right_finger, left_finger, open_hand, close_hand, two_finger

**Expected Output:**
- Creates `dataset/` folder with subfolders for each class
- Saves images as `classname_00000.jpg`, `classname_00001.jpg`, etc.

---

### **Step 2: Train Model**

```bash
python train_model.py
```

**What it does:**
- Loads images from `dataset/` folder
- Applies data augmentation
- Trains CNN model for 15 epochs
- Saves `model.h5` and `gesture_model.tflite`
- Generates `confusion_matrix.png` and `training_history.png`

**Expected Output:**
- Training progress with accuracy/loss metrics
- Model saved as `gesture_model.tflite` (~200 KB)
- Training takes 5-15 minutes depending on dataset size

---

### **Step 3: Test Live Prediction**

```bash
python predict.py
```

**Features:**
- Real-time webcam feed
- Green ROI box (position hand inside)
- Gesture prediction with confidence score
- FPS counter
- All class probabilities displayed

**Controls:**
- `Q` - Quit

**Expected Output:**
- Shows predicted gesture (RIGHT, LEFT, FORWARD, BACK, STOP)
- Confidence score (0.0-1.0)
- Color-coded ROI (green = high confidence, orange = medium, red = low)

---

## 🔧 **OPTIONS FOR MEDIAPIPE (app.py & test.py)**

### **Option 1: Use Python 3.11 or 3.12 (Recommended)**

MediaPipe supports Python 3.11 and 3.12. You can:

1. **Install Python 3.11 or 3.12** alongside Python 3.13
2. **Create a virtual environment:**
   ```bash
   # Using Python 3.11 (if installed)
   py -3.11 -m venv venv_mediapipe
   venv_mediapipe\Scripts\activate
   pip install opencv-python numpy tensorflow mediapipe
   ```

3. **Then run:**
   ```bash
   python app.py
   ```

### **Option 2: Skip MediaPipe Scripts (For Now)**

Focus on testing the CNN approach first:
- `collect_data.py` ✅
- `train_model.py` ✅
- `predict.py` ✅

These don't require MediaPipe and work perfectly with Python 3.13.

---

## 📊 **TESTING CHECKLIST**

- [ ] Test camera access: `python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL'); cap.release()"`
- [ ] Run `collect_data.py` - Collect at least 50 images per class
- [ ] Run `train_model.py` - Train model successfully
- [ ] Check `gesture_model.tflite` exists
- [ ] Run `predict.py` - Test live prediction
- [ ] Verify gestures are recognized correctly

---

## 🐛 **TROUBLESHOOTING**

### **Camera not opening:**
```python
# Try different camera index in collect_data.py or predict.py:
cap = cv2.VideoCapture(1)  # Instead of 0
```

### **Low accuracy:**
- Collect more training data (500+ images per class recommended)
- Ensure good lighting
- Vary hand positions during collection

### **Model not found error:**
- Make sure you ran `train_model.py` first
- Check that `gesture_model.tflite` exists in project root

### **Import errors:**
```bash
# Reinstall packages:
python -m pip install --upgrade opencv-python numpy tensorflow scikit-learn matplotlib seaborn --user
```

---

## 📝 **QUICK TEST COMMANDS**

```bash
# 1. Verify packages
python -c "import cv2, numpy, tensorflow; print('✅ All packages OK')"

# 2. Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('✅ Camera OK' if cap.isOpened() else '❌ Camera Error'); cap.release()"

# 3. Start data collection
python collect_data.py

# 4. Train model (after collecting data)
python train_model.py

# 5. Test prediction (after training)
python predict.py
```

---

## 🎯 **NEXT STEPS FOR RASPBERRY PI**

Once tested locally:

1. **Transfer project to Raspberry Pi**
2. **Install dependencies on Pi:**
   ```bash
   pip install opencv-python numpy tensorflow-lite-runtime scikit-learn matplotlib seaborn
   ```
3. **For MediaPipe on Pi:**
   ```bash
   pip install mediapipe  # Should work on Pi with Python 3.9-3.11
   ```
4. **Run `test.py` on Pi** (includes hardware control)

---

## ✅ **SUMMARY**

**Ready to Test:**
- ✅ CNN workflow (collect → train → predict)
- ✅ All packages installed
- ✅ Camera working

**Needs Python 3.11/3.12:**
- ❌ MediaPipe-based scripts (app.py, test.py)

**Recommendation:**
Start with CNN approach testing. It's fully functional and will work on Raspberry Pi too!

