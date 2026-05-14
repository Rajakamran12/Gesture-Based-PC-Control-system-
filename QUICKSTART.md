# Quick Start Guide

## Frontend Launcher (Click-To-Run)

Use the new frontend if you want a simple Run App button, launch options, and logs in one window:

```bash
run_frontend_launcher.bat
```

Inside the frontend:
- Choose a run target (Gesture Dashboard is recommended)
- Click **Run App**
- View launch logs and environment status directly in the UI

## Web Dashboard (Vercel Ready)

Use the new web dashboard project when you want cloud hosting:

```bash
cd web-dashboard
npm install
npm run dev
```

For deployment details, see `web-dashboard/README.md`.

## Installation (One-time setup)

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage Workflow

### 1. Collect Dataset (5-10 minutes per class)
```bash
python collect_data.py
```
- Position hand in green ROI box
- Press SPACE to capture (aim for 500+ images per class)
- Use N/P to switch classes
- Press Q when done

### 2. Train Model (5-15 minutes)
```bash
python train_model.py
```
- Automatically loads dataset
- Trains model for 15 epochs
- Saves `gesture_model.tflite`
- Generates accuracy reports

### 3. Run Live Prediction
```bash
python predict.py
```
- Opens webcam
- Shows real-time gesture recognition
- Press Q to quit

## Expected Results

- **Model Size**: ~200 KB (quantized)
- **Accuracy**: 85-95% (with good dataset)
- **FPS**: 15-30 on Raspberry Pi 5
- **Training Time**: 5-15 minutes (depends on dataset size)

## Troubleshooting

**Webcam not working?**
- Try: `cv2.VideoCapture(1)` instead of `0`
- Check camera permissions

**Low accuracy?**
- Collect more images (500+ per class)
- Ensure good lighting
- Vary hand positions

**Slow on Raspberry Pi?**
- Use `tensorflow-lite-runtime` instead of full TensorFlow
- Reduce `IMG_SIZE` to 96 in all scripts

