# AI Driving Web Dashboard (Vercel Ready)

This is a web-based dashboard for your AI driving gesture app. It runs fully in the browser and supports:

- Dashboard UI with run/stop controls
- Permission modal before start
- Browser camera permission flow
- Live webcam preview + hand landmarks (MediaPipe Tasks Vision)
- Gesture status and runtime metrics

## Local Run

```bash
cd web-dashboard
npm install
npm run dev
```

Open `http://localhost:3000`.

## Production Build Check

```bash
cd web-dashboard
npm run build
npm start
```

## Deploy To Vercel

1. Push this repository to GitHub.
2. In Vercel, click **Add New Project** and import this repo.
3. Set **Root Directory** to `web-dashboard`.
4. Framework preset: **Next.js** (auto-detected).
5. Keep default build settings:
- Build Command: `npm run build`
- Output: `.next`
6. Deploy.

## Important Notes

- This web version uses browser camera APIs, so camera access must be allowed by the user.
- Browser deployment cannot control OS-wide mouse/keyboard directly like desktop Python `pyautogui`; actions are simulated in the UI.
- For full desktop control, keep using the desktop launcher (`frontend_launcher.py`).
