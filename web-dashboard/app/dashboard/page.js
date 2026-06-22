'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '../auth-context';
import ErrorBoundary from './ErrorBoundary';

const DEFAULT_GESTURE = "No hand";
const MODULE_BUILD = "modules v2-web";
const SMOOTH_PROFILE = "fast";
const ACTIVE_MODULES = [
  "Feature Engineering",
  "Gesture Classification",
  "Gesture Smoothing & Optimization",
];
const MODULE_DETAILS = {
  "Feature Engineering": "Normalizes hand landmarks and builds a compact feature vector for robust gesture recognition.",
  "Gesture Classification": "Runs rule-based + feature-assisted gesture labeling with confidence scoring.",
  "Gesture Smoothing & Optimization": "Applies temporal consensus and anti-jitter logic before triggering actions.",
};
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const LANDMARK_NAMES = [
  "Wrist",
  "Thumb CMC",
  "Thumb MCP",
  "Thumb IP",
  "Thumb Tip",
  "Index MCP",
  "Index PIP",
  "Index DIP",
  "Index Tip",
  "Middle MCP",
  "Middle PIP",
  "Middle DIP",
  "Middle Tip",
  "Ring MCP",
  "Ring PIP",
  "Ring DIP",
  "Ring Tip",
  "Pinky MCP",
  "Pinky PIP",
  "Pinky DIP",
  "Pinky Tip",
];

const TIP_INDICES = [4, 8, 12, 16, 20];

function detectPinchZoom(landmarks, previousPinchDistance = null) {
  if (!landmarks || landmarks.length < 9) return { gesture: null, distance: null };
  
  const thumb = landmarks[4];
  const index = landmarks[8];
  const distance = distanceXY(thumb, index);
  
  if (previousPinchDistance === null) {
    return { gesture: null, distance };
  }
  
  if (distance < previousPinchDistance - 0.03) {
    return { gesture: "Zoom In", distance };
  }
  if (distance > previousPinchDistance + 0.03) {
    return { gesture: "Zoom Out", distance };
  }
  
  return { gesture: null, distance };
}

function detectSwipe(landmarks, previousLandmarks = null) {
  if (!landmarks || landmarks.length < 9 || !previousLandmarks) {
    return null;
  }
  
  const wrist = landmarks[0];
  const prevWrist = previousLandmarks[0];
  const xDelta = wrist.x - prevWrist.x;
  
  if (Math.abs(xDelta) > 0.04) {
    return xDelta < 0 ? "Swipe Left" : "Swipe Right";
  }
  
  return null;
}

function classifyGesture(landmarks, previousPinchDistance = null, previousLandmarks = null) {
  if (!landmarks || landmarks.length !== 21) {
    return { gesture: DEFAULT_GESTURE, pinchDistance: null };
  }

  const wrist = landmarks[0];
  const index = landmarks[8];
  const middle = landmarks[12];
  const ring = landmarks[16];
  const little = landmarks[20];
  const thumb = landmarks[4];

  const pinch = detectPinchZoom(landmarks, previousPinchDistance);
  if (pinch.gesture) {
    return { gesture: pinch.gesture, pinchDistance: pinch.distance };
  }
  
  const swipe = detectSwipe(landmarks, previousLandmarks);
  if (swipe) {
    return { gesture: swipe, pinchDistance: pinch.distance };
  }

  const fingersUp = [index, middle, ring, little].map((tip, i) => {
    const pipIndex = [6, 10, 14, 18][i];
    return tip.y < landmarks[pipIndex].y;
  });

  const count = fingersUp.filter(Boolean).length;
  if (count >= 4) return { gesture: "Open Palm", pinchDistance: pinch.distance };
  if (count === 0) return { gesture: "Fist", pinchDistance: pinch.distance };
  if (fingersUp[0] && fingersUp[1] && !fingersUp[2] && !fingersUp[3]) return { gesture: "Two Finger", pinchDistance: pinch.distance };
  if (fingersUp[0] && !fingersUp[1] && !fingersUp[2] && !fingersUp[3]) {
    if (index.x < wrist.x - 0.05) return { gesture: "Point Left", pinchDistance: pinch.distance };
    if (index.x > wrist.x + 0.05) return { gesture: "Point Right", pinchDistance: pinch.distance };
    return { gesture: "Point", pinchDistance: pinch.distance };
  }
  return { gesture: "None", pinchDistance: pinch.distance };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function distanceXY(a, b) {
  const dx = (a.x ?? 0) - (b.x ?? 0);
  const dy = (a.y ?? 0) - (b.y ?? 0);
  return Math.sqrt(dx * dx + dy * dy);
}

function extractFeatureVector(landmarks) {
  if (!landmarks || landmarks.length !== 21) {
    return null;
  }

  const wrist = landmarks[0];
  const palmWidth = Math.max(distanceXY(landmarks[5], landmarks[17]), 1e-6);

  const centered = landmarks.map((point) => ({
    x: (point.x - wrist.x) / palmWidth,
    y: (point.y - wrist.y) / palmWidth,
    z: (point.z - wrist.z) / palmWidth,
  }));

  const xyVector = [];
  for (const point of centered) {
    xyVector.push(point.x, point.y);
  }

  const distances = [
    distanceXY(landmarks[0], landmarks[4]) / palmWidth,
    distanceXY(landmarks[0], landmarks[8]) / palmWidth,
    distanceXY(landmarks[0], landmarks[12]) / palmWidth,
    distanceXY(landmarks[0], landmarks[16]) / palmWidth,
    distanceXY(landmarks[0], landmarks[20]) / palmWidth,
    distanceXY(landmarks[4], landmarks[8]) / palmWidth,
    distanceXY(landmarks[8], landmarks[12]) / palmWidth,
    distanceXY(landmarks[12], landmarks[16]) / palmWidth,
    distanceXY(landmarks[16], landmarks[20]) / palmWidth,
  ];

  return {
    vector: [...xyVector, ...distances],
    palmWidth,
    featureDim: xyVector.length + distances.length,
    fingertipSpread: distances.slice(0, 5).reduce((sum, value) => sum + value, 0) / 5,
  };
}

function classifyWithFeatureEngineering(landmarks, handednessLabel = "", previousPinchDistance = null, previousLandmarks = null) {
  if (!landmarks || landmarks.length !== 21) {
    return {
      gesture: DEFAULT_GESTURE,
      confidence: 0,
      source: "feature-engineering",
      featureDim: 0,
      pinchDistance: null,
    };
  }

  const features = extractFeatureVector(landmarks);
  if (!features) {
    return {
      gesture: DEFAULT_GESTURE,
      confidence: 0,
      source: "feature-engineering",
      featureDim: 0,
      pinchDistance: null,
    };
  }

  const gestureResult = classifyGesture(landmarks, previousPinchDistance, previousLandmarks);
  const baseGesture = gestureResult.gesture;
  
  if (baseGesture === "Zoom In" || baseGesture === "Zoom Out" || baseGesture === "Swipe Left" || baseGesture === "Swipe Right") {
    return {
      gesture: baseGesture,
      confidence: 0.9,
      source: "feature-engineering",
      featureDim: features.featureDim,
      pinchDistance: gestureResult.pinchDistance,
    };
  }
  
  const fingersUp = [8, 12, 16, 20].map((tipId, index) => {
    const pipId = [6, 10, 14, 18][index];
    return landmarks[tipId].y < landmarks[pipId].y;
  });

  const countUp = fingersUp.filter(Boolean).length;
  const handedness = handednessLabel.toLowerCase();
  const thumbTipX = landmarks[4].x;
  const thumbIpX = landmarks[3].x;
  const thumbUp = handedness.includes("right") ? thumbTipX > thumbIpX : thumbTipX < thumbIpX;

  let confidence = 0.45;
  if (baseGesture === "Open Palm") {
    confidence = clamp(0.62 + (features.fingertipSpread - 1.0) * 0.2, 0.55, 0.96);
  } else if (baseGesture === "Fist") {
    confidence = clamp(0.90 - features.fingertipSpread * 0.18, 0.55, 0.95);
  } else if (baseGesture === "Two Finger") {
    confidence = clamp(0.68 + countUp * 0.04, 0.55, 0.92);
  } else if (baseGesture === "Point" || baseGesture === "Point Left" || baseGesture === "Point Right") {
    confidence = clamp(0.66 + (landmarks[6].y - landmarks[8].y) * 0.35, 0.55, 0.92);
  } else if (thumbUp) {
    confidence = 0.62;
  } else if (baseGesture === "None") {
    confidence = clamp(0.30 + countUp * 0.06, 0.2, 0.52);
  }

  return {
    gesture: baseGesture,
    confidence,
    source: "feature-engineering",
    featureDim: features.featureDim,
    pinchDistance: gestureResult.pinchDistance,
  };
}

function createSmoothingState() {
  return {
    history: [],
    stableGesture: DEFAULT_GESTURE,
    pendingGesture: DEFAULT_GESTURE,
    pendingCount: 0,
    cooldown: 0,
  };
}

function getSmoothingPreset(profile) {
  if (profile === "balanced") {
    return { windowSize: 7, minConfidence: 0.55, minConsensus: 0.6, holdFrames: 2, cooldown: 1 };
  }
  if (profile === "stable") {
    return { windowSize: 9, minConfidence: 0.62, minConsensus: 0.68, holdFrames: 3, cooldown: 2 };
  }
  return { windowSize: 5, minConfidence: 0.5, minConsensus: 0.55, holdFrames: 1, cooldown: 0 };
}

function getGestureThresholds(gesture, basePreset) {
  const overrides = {
    "Point": { minConfidence: 0.58, minConsensus: 0.62, holdFrames: 2 },
    "Point Left": { minConfidence: 0.62, minConsensus: 0.66, holdFrames: 2 },
    "Point Right": { minConfidence: 0.62, minConsensus: 0.66, holdFrames: 2 },
    "Two Finger": { minConfidence: 0.65, minConsensus: 0.68, holdFrames: 2 },
    "Open Palm": { minConfidence: 0.5, minConsensus: 0.56, holdFrames: 1 },
    "Fist": { minConfidence: 0.52, minConsensus: 0.58, holdFrames: 1 },
  };

  const custom = overrides[gesture] || {};
  return {
    minConfidence: custom.minConfidence ?? basePreset.minConfidence,
    minConsensus: custom.minConsensus ?? basePreset.minConsensus,
    holdFrames: custom.holdFrames ?? basePreset.holdFrames,
  };
}

function smoothGesture(state, rawGesture, rawConfidence, profile = "fast") {
  const preset = getSmoothingPreset(profile);
  const thresholds = getGestureThresholds(rawGesture, preset);
  const filteredGesture = rawConfidence >= thresholds.minConfidence ? rawGesture : DEFAULT_GESTURE;
  const filteredConfidence = rawConfidence >= thresholds.minConfidence ? rawConfidence : 0;

  state.history.push({ gesture: filteredGesture, confidence: filteredConfidence });
  if (state.history.length > preset.windowSize) {
    state.history.shift();
  }

  const weightedVotes = new Map();
  let total = 0;
  const size = state.history.length;

  state.history.forEach((item, index) => {
    const recencyWeight = 1 + (index / Math.max(1, size - 1)) * 0.5;
    const vote = Math.max(item.confidence, 0.01) * recencyWeight;
    weightedVotes.set(item.gesture, (weightedVotes.get(item.gesture) || 0) + vote);
    total += vote;
  });

  let candidate = state.stableGesture;
  let consensus = 0;
  if (total > 0 && weightedVotes.size > 0) {
    const sorted = [...weightedVotes.entries()].sort((a, b) => b[1] - a[1]);
    candidate = sorted[0][0];
    consensus = sorted[0][1] / total;
  }

  const candidateThresholds = getGestureThresholds(candidate, preset);
  const requiredConsensus = Math.max(thresholds.minConsensus, candidateThresholds.minConsensus);
  const requiredHold = Math.max(thresholds.holdFrames, candidateThresholds.holdFrames);

  if (consensus < requiredConsensus) {
    candidate = state.stableGesture;
  }

  let changed = false;
  if (candidate === state.stableGesture) {
    state.pendingGesture = candidate;
    state.pendingCount = 0;
    if (state.cooldown > 0) state.cooldown -= 1;
  } else if (state.cooldown > 0) {
    state.cooldown -= 1;
  } else if (candidate !== state.pendingGesture) {
    state.pendingGesture = candidate;
    state.pendingCount = 1;
  } else {
    state.pendingCount += 1;
    if (state.pendingCount >= requiredHold) {
      state.stableGesture = candidate;
      state.pendingCount = 0;
      state.cooldown = preset.cooldown;
      changed = true;
    }
  }

  const uniqueGestures = new Set(state.history.map((item) => item.gesture));
  const jitter = state.history.length > 0 ? uniqueGestures.size / state.history.length : 0;

  return {
    stableGesture: state.stableGesture,
    filteredGesture,
    consensus,
    jitter,
    changed,
    minConfidence: thresholds.minConfidence,
    minConsensus: requiredConsensus,
    holdFrames: requiredHold,
  };
}

function distance2D(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function distance3D(a, b) {
  const dx = (a.x ?? 0) - (b.x ?? 0);
  const dy = (a.y ?? 0) - (b.y ?? 0);
  const dz = (a.z ?? 0) - (b.z ?? 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function formatDistance(value, isWorld = false) {
  if (!Number.isFinite(value)) return "n/a";
  return isWorld ? `${value.toFixed(3)}m` : `${value.toFixed(1)}px`;
}

function drawRoundedRectPath(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + r);
  ctx.lineTo(x + width, y + height - r);
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  ctx.lineTo(x + r, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function drawLabel(ctx, text, x, y, bg = "rgba(18, 34, 28, 0.78)", fg = "#f3fff7") {
  ctx.save();
  ctx.font = "11px Segoe UI";
  const paddingX = 6;
  const paddingY = 4;
  const metrics = ctx.measureText(text);
  const w = metrics.width + paddingX * 2;
  const h = 18;
  ctx.fillStyle = bg;
  drawRoundedRectPath(ctx, x, y - h, w, h, 6);
  ctx.fill();
  ctx.fillStyle = fg;
  ctx.fillText(text, x + paddingX, y - 5);
  ctx.restore();
}

function drawHandOverlay(ctx, handLandmarks, worldLandmarks, width, height, handIndex, handednessLabel, handednessScore) {
  if (!handLandmarks?.length) return;

  const palette = ["#2be5a7", "#f6c15b", "#7ad7ff", "#ff8a7a"];
  const color = palette[handIndex % palette.length];

  const points = handLandmarks.map((lm) => ({
    x: width - lm.x * width,
    y: lm.y * height,
    z: lm.z,
  }));

  let minX = width;
  let minY = height;
  let maxX = 0;
  let maxY = 0;

  ctx.save();
  ctx.lineWidth = 3;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;

  for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
    const start = points[startIdx];
    const end = points[endIdx];
    if (!start || !end) continue;
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
  }

  for (let i = 0; i < points.length; i += 1) {
    const pt = points[i];
    minX = Math.min(minX, pt.x);
    minY = Math.min(minY, pt.y);
    maxX = Math.max(maxX, pt.x);
    maxY = Math.max(maxY, pt.y);

    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(15, 25, 21, 0.55)";
    ctx.lineWidth = 1;
    ctx.stroke();

    if ([0, 4, 8, 12, 16, 20].includes(i)) {
      const label = `${i}: ${LANDMARK_NAMES[i]}`;
      drawLabel(ctx, label, pt.x + 8, pt.y - 8, "rgba(8, 24, 19, 0.86)");
    }
  }

  const wrist = points[0];
  const worldWrist = worldLandmarks?.[0] ?? null;
  const worldDistances = TIP_INDICES.map((tipIdx) => {
    const tipWorld = worldLandmarks?.[tipIdx];
    const tipPoint = points[tipIdx];
    return {
      name: LANDMARK_NAMES[tipIdx],
      px: distance2D(wrist, tipPoint),
      world: worldWrist && tipWorld ? distance3D(worldWrist, tipWorld) : null,
    };
  });

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(minX - 14, minY - 34, maxX - minX + 28, maxY - minY + 48);

  const titleLines = [
    `${handednessLabel || "Unknown"} hand`,
    `Score ${(handednessScore * 100 || 0).toFixed(0)}%`,
    `Tip distances: ${worldDistances
      .map((item) => `${item.name.split(" ")[0]} ${formatDistance(item.world ?? item.px, Boolean(item.world))}`)
      .join(" | ")}`,
  ];

  const boxWidth = Math.min(width - 20, 430);
  const boxHeight = 66;
  const boxX = Math.max(10, minX - 10);
  const boxY = Math.max(42, minY - 44);
  ctx.fillStyle = "rgba(8, 24, 19, 0.78)";
  drawRoundedRectPath(ctx, boxX, boxY - boxHeight + 10, boxWidth, boxHeight, 12);
  ctx.fill();

  ctx.fillStyle = "#effef4";
  ctx.font = "12px Segoe UI";
  let lineY = boxY - boxHeight + 24;
  titleLines.forEach((line) => {
    ctx.fillText(line, boxX + 12, lineY);
    lineY += 18;
  });

  ctx.restore();
}

function drawLandmarks(ctx, result, width, height) {
  ctx.clearRect(0, 0, width, height);

  if (!result?.landmarks) {
    return;
  }

  const landmarks = result.landmarks;
  const handedness = result.handedness || [];
  const worldLandmarks = result.worldLandmarks || [];

  landmarks.forEach((handLandmarks, handIndex) => {
    const worldLmks = worldLandmarks[handIndex] || null;
    const handData = handedness[handIndex] || [];
    const categoryName = handData[0]?.categoryName || "";
    const score = handData[0]?.score ?? 0;

    drawHandOverlay(ctx, handLandmarks, worldLmks, width, height, handIndex, categoryName, score);
  });
}

export default function DashboardPage() {
  const router = useRouter();
  const { user, loading, logout } = useAuth();

  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const animationRef = useRef(null);
  const streamRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const lastFrameTimeRef = useRef(0);
  const smoothingRef = useRef(createSmoothingState());
  const previousLandmarksRef = useRef(null);
  const previousPinchDistanceRef = useRef(null);
  const prevFingerCountRef = useRef(null);
  const zoomTimeoutRef = useRef(null);

  const [running, setRunning] = useState(false);
  const [permissionOpen, setPermissionOpen] = useState(false);
  const [cameraAllowed, setCameraAllowed] = useState(true);
  const [controlsAllowed, setControlsAllowed] = useState(false);
  const [status, setStatus] = useState("Idle");
  const [gesture, setGesture] = useState(DEFAULT_GESTURE);
  const [rawGesture, setRawGesture] = useState(DEFAULT_GESTURE);
  const [rawConfidence, setRawConfidence] = useState(0);
  const [consensusRatio, setConsensusRatio] = useState(0);
  const [jitterIndex, setJitterIndex] = useState(0);
  const [featureDim, setFeatureDim] = useState(0);
  const [classificationSource, setClassificationSource] = useState("feature-engineering");
  const [handsCount, setHandsCount] = useState(0);
  const [fps, setFps] = useState(0);
  const [lastAction, setLastAction] = useState("Waiting");
  const [zoomAction, setZoomAction] = useState(null);
  const [error, setError] = useState("");
  const [activeScreen, setActiveScreen] = useState("overview");

  useEffect(() => {
    if (!loading && !user) {
      router.push('/login');
    }
  }, [user, loading, router]);

  const score = useMemo(() => {
    let s = 40;
    if (running) s += 30;
    if (gesture !== DEFAULT_GESTURE) s += 20;
    if (fps > 15) s += 10;
    return Math.min(100, s);
  }, [running, gesture, fps]);

  const metricRows = useMemo(() => {
    const fpsRatio = clamp(fps / 30, 0, 1);
    const handsRatio = clamp(handsCount / 2, 0, 1);
    const stabilityRatio = clamp(1 - jitterIndex, 0, 1);
    return [
      { label: "FPS", value: fps.toFixed(1), ratio: fpsRatio },
      { label: "Runtime Score", value: `${score}%`, ratio: score / 100 },
      { label: "Hands", value: `${handsCount}/2`, ratio: handsRatio },
      { label: "Stability", value: `${(stabilityRatio * 100).toFixed(0)}%`, ratio: stabilityRatio },
    ];
  }, [rawConfidence, consensusRatio, fps, score, handsCount, jitterIndex]);

  const moduleRows = useMemo(() => {
    const confidencePct = Math.round(rawConfidence * 100);
    const consensusPct = Math.round(consensusRatio * 100);
    const smoothPct = Math.round(clamp((1 - jitterIndex) * 100, 0, 100));
    return [
      {
        name: "Feature Engineering",
        health: Math.round(clamp((featureDim / 51) * 100, 0, 100)),
        detail: `Vector dim ${featureDim}/51`,
      },
      {
        name: "Gesture Classification",
        health: confidencePct,
        detail: `Raw ${rawGesture} (${confidencePct}%)`,
      },
      {
        name: "Gesture Smoothing & Optimization",
        health: Math.round((consensusPct + smoothPct) / 2),
        detail: `Consensus ${consensusPct}% | Smooth ${smoothPct}%`,
      },
    ];
  }, [featureDim, rawGesture, rawConfidence, consensusRatio, jitterIndex]);

  useEffect(() => {
    return () => {
      stopApp();
    };
  }, []);

  if (loading || !user) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100vh',
        background: '#0f0f0f',
        color: '#f0f0f0',
        fontFamily: '"Segoe UI", sans-serif'
      }}>
        <p>Loading...</p>
      </div>
    );
  }

  async function initLandmarker() {
    if (handLandmarkerRef.current) return handLandmarkerRef.current;

    // Dynamically import mediapipe to avoid loading wasm/wasm-resolver during
    // SSR or hydration time which can cause client errors in certain hosts.
    const mp = await import('@mediapipe/tasks-vision');
    const { FilesetResolver, HandLandmarker } = mp;

    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.65,
      minTrackingConfidence: 0.5,
    });

    return handLandmarkerRef.current;
  }

  function stopApp() {
    setRunning(false);
    setStatus("Stopped");
    setLastAction("Stopped by user");

    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }

    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) {
        track.stop();
      }
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    const canvas = overlayRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }

  function runLoop() {
    const video = videoRef.current;
    const canvas = overlayRef.current;
    const detector = handLandmarkerRef.current;

    if (!video || !canvas || !detector || video.readyState < 2) {
      animationRef.current = requestAnimationFrame(runLoop);
      return;
    }

    const now = performance.now();
    const delta = now - (lastFrameTimeRef.current || now);
    lastFrameTimeRef.current = now;

    if (delta > 0) {
      const instant = 1000 / delta;
      setFps((prev) => (prev === 0 ? instant : prev * 0.85 + instant * 0.15));
    }

    const result = detector.detectForVideo(video, now);
    const landmarks = result?.landmarks?.[0] ?? null;
    const handednessLabel = result?.handedness?.[0]?.[0]?.categoryName || "";
    const ctx = canvas.getContext("2d");

    if (ctx) {
      drawLandmarks(ctx, result, canvas.width, canvas.height);
    }

    const classification = classifyWithFeatureEngineering(
      landmarks, 
      handednessLabel, 
      previousPinchDistanceRef.current,
      previousLandmarksRef.current
    );
    
    if (landmarks) {
      previousLandmarksRef.current = landmarks;
      previousPinchDistanceRef.current = classification.pinchDistance;

      // Detect slow fist close (Zoom In) and slow fist open (Zoom Out)
      const fingersUpNow = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]].map((tip, i) => {
        const pipIndex = [6, 10, 14, 18][i];
        return tip.y < landmarks[pipIndex].y;
      });
      const fingerCountNow = fingersUpNow.filter(Boolean).length;
      if (prevFingerCountRef.current !== null) {
        const prevCount = prevFingerCountRef.current;
        if (prevCount >= 3 && fingerCountNow === 0) {
          setZoomAction("Zoom In");
          setLastAction("Zoom In");
          clearTimeout(zoomTimeoutRef.current);
          zoomTimeoutRef.current = setTimeout(() => setZoomAction(null), 1500);
        } else if (prevCount === 0 && fingerCountNow >= 3) {
          setZoomAction("Zoom Out");
          setLastAction("Zoom Out");
          clearTimeout(zoomTimeoutRef.current);
          zoomTimeoutRef.current = setTimeout(() => setZoomAction(null), 1500);
        }
      }
      prevFingerCountRef.current = fingerCountNow;
    }
    
    const smoothing = smoothGesture(
      smoothingRef.current,
      classification.gesture,
      classification.confidence,
      SMOOTH_PROFILE,
    );

    const nextStableGesture = smoothing.stableGesture;
    setGesture(nextStableGesture);
    setRawGesture(classification.gesture);
    setRawConfidence(classification.confidence);
    setConsensusRatio(smoothing.consensus);
    setJitterIndex(smoothing.jitter);
    setFeatureDim(classification.featureDim);
    setClassificationSource(classification.source);
    setHandsCount(result?.landmarks?.length ?? 0);

    if (nextStableGesture === "Zoom In") setLastAction("Zoom In");
    else if (nextStableGesture === "Zoom Out") setLastAction("Zoom Out");
    else if (nextStableGesture === "Swipe Left") setLastAction("Swipe Left");
    else if (nextStableGesture === "Swipe Right") setLastAction("Swipe Right");
    else if (nextStableGesture === "Open Palm") setLastAction("Play/Pause (simulated)");
    else if (nextStableGesture === "Fist") setLastAction("Click (simulated)");
    else if (nextStableGesture === "Point Left") setLastAction("Back (simulated)");
    else if (nextStableGesture === "Point Right") setLastAction("Forward (simulated)");
    else setLastAction("No action");

    animationRef.current = requestAnimationFrame(runLoop);
  }

  async function startAppWithPermissions() {
    setError("");

    if (!cameraAllowed) {
      setError("Camera permission is required to run the app.");
      return;
    }

    try {
      setStatus("Starting...");
      const detector = await initLandmarker();
      if (!detector) throw new Error("Failed to initialize hand detector.");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 960, height: 540, facingMode: "user" },
        audio: false,
      });

      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error("Video element not ready.");

      video.srcObject = stream;
      await video.play();

      const canvas = overlayRef.current;
      if (canvas) {
        canvas.width = video.videoWidth || 960;
        canvas.height = video.videoHeight || 540;
      }

      setRunning(true);
      setStatus("Running");
      setLastAction(controlsAllowed ? "Controls enabled" : "Controls disabled");
      lastFrameTimeRef.current = performance.now();
      animationRef.current = requestAnimationFrame(runLoop);
    } catch (e) {
      setStatus("Error");
      setError(e instanceof Error ? e.message : "Could not start camera.");
      stopApp();
    }
  }

  return (
    <ErrorBoundary>
    <main className="page">
      <aside className="sidebar">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <h1>AI-Driven Gesture Based PC Control System</h1>
          <button
            onClick={async () => {
              try {
                await logout();
              } catch {
                // ignore errors, auth context will clear and redirect
              }
            }}
            style={{
              padding: '6px 12px',
              background: '#3a3a3a',
              color: '#f0f0f0',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.75rem',
              fontWeight: '500'
            }}
          >
            Logout
          </button>
        </div>
        <p style={{ margin: '0 0 12px 0', fontSize: '0.875rem', color: '#a0a0a0' }}>Welcome, {user?.name || 'User'}</p>
        <div className="chip">{running ? "APP RUNNING" : "APP STOPPED"}</div>

        <div className="sidebar-actions">
          <button className="btn primary" onClick={() => setPermissionOpen(true)}>
            Run App
          </button>
          <button className="btn" onClick={stopApp}>
            Stop App
          </button>
        </div>

        <div className="sidebar-grid">
          <button className="card side-card side-card-btn" onClick={() => setActiveScreen("overview")}><span>Status</span><strong>{status}</strong></button>
          <button className="card side-card side-card-btn" onClick={() => setActiveScreen("overview")}><span>Gesture</span><strong>{gesture}</strong></button>
          <button className="card side-card side-card-btn" onClick={() => setActiveScreen("overview")}><span>Last Action</span><strong>{lastAction}</strong></button>
          <button className="card side-card side-card-btn" onClick={() => setActiveScreen("live")}><span>Live Feed</span><strong>View</strong></button>

          <article
            className="card side-card control-card"
            role="button"
            tabIndex={0}
            onClick={() => setActiveScreen("permissions")}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") setActiveScreen("permissions");
            }}
          >
            <h3>Permissions</h3>
            <label>
              <input
                type="checkbox"
                checked={cameraAllowed}
                onChange={(e) => setCameraAllowed(e.target.checked)}
              />
              Allow camera access (required)
            </label>
            <label>
              <input
                type="checkbox"
                checked={controlsAllowed}
                onChange={(e) => setControlsAllowed(e.target.checked)}
              />
              Allow controls (simulated in browser)
            </label>
            <p className="muted">Browser will ask actual camera permission on first run.</p>
            {error ? <p className="error">{error}</p> : null}
          </article>

          <div className="sidebar-tabs" role="tablist" aria-label="Dashboard Screens">
            <h3>Operations Dashboard</h3>
            <p className="muted">Choose a screen to view meaningful runtime data.</p>
            <button className={`sidebar-tab ${activeScreen === "overview" ? "active" : ""}`} onClick={() => setActiveScreen("overview")}>Overview</button>
            <button className={`sidebar-tab ${activeScreen === "metrics" ? "active" : ""}`} onClick={() => setActiveScreen("metrics")}>Metrics</button>
            <button className={`sidebar-tab ${activeScreen === "modules" ? "active" : ""}`} onClick={() => setActiveScreen("modules")}>Modules</button>
            <button className={`sidebar-tab ${activeScreen === "permissions" ? "active" : ""}`} onClick={() => setActiveScreen("permissions")}>Permissions</button>
            <button className={`sidebar-tab ${activeScreen === "live" ? "active" : ""}`} onClick={() => setActiveScreen("live")}>Live Feed</button>
          </div>
        </div>
      </aside>

      <section className="content">
        <article className="card screen-panel">
          {activeScreen === "overview" ? null : null}

          {activeScreen === "metrics" ? (
            <>
              <h3>Runtime Metrics</h3>
              <p className="muted">Live signal quality panel for gesture runtime diagnostics.</p>
              <div className="metric-grid">
                {metricRows.map((item) => (
                  <article className="metric-card" key={item.label}>
                    <div className="metric-head">
                      <span>{item.label}</span>
                      <strong>{item.value}</strong>
                    </div>
                    <div className="metric-track">
                      <div className="metric-fill" style={{ width: `${Math.round(item.ratio * 100)}%` }} />
                    </div>
                  </article>
                ))}
              </div>
              <p className="muted">Source: {classificationSource} | Active gesture: {gesture}</p>
            </>
          ) : null}

          {activeScreen === "modules" ? (
            <>
              <h3>Pipeline Modules</h3>
              <p className="muted">Build {MODULE_BUILD} with {SMOOTH_PROFILE} smoothing profile.</p>
              <div className="module-diagnostics-grid">
                {moduleRows.map((module) => (
                  <article className="module-diagnostic" key={module.name}>
                    <div className="metric-head">
                      <span>{module.name}</span>
                      <strong>{module.health}%</strong>
                    </div>
                    <div className="metric-track">
                      <div className="metric-fill" style={{ width: `${module.health}%` }} />
                    </div>
                    <p className="muted">{MODULE_DETAILS[module.name]}</p>
                    <p className="muted">{module.detail}</p>
                  </article>
                ))}
              </div>
            </>
          ) : null}

          {activeScreen === "permissions" ? (
            <>
              <h3>Permissions Screen</h3>
              <p className="muted">Camera access: <strong>{cameraAllowed ? "Allowed" : "Blocked"}</strong></p>
              <p className="muted">Control access: <strong>{controlsAllowed ? "Allowed" : "Blocked"}</strong></p>
              <p className="muted">You can also toggle both in the sidebar permissions card.</p>
            </>
          ) : null}

          {activeScreen === "live" ? (
            <>
              <h3>Live Feed Screen</h3>
              <p className="muted">Camera stream and hand landmark overlay are shown below.</p>
            </>
          ) : null}
        </article>

        <article className="card video-card">
          <div className="video-wrap">
            <video ref={videoRef} playsInline muted className="mirror-feed" />
            <canvas ref={overlayRef} />
            {zoomAction && (
              <div style={{
                position: "absolute",
                top: "50%",
                left: "50%",
                transform: "translate(-50%, -50%)",
                fontSize: "2.2rem",
                fontWeight: "700",
                color: zoomAction === "Zoom In" ? "#00ff88" : "#ff9900",
                textShadow: "0 2px 20px rgba(0,0,0,0.8)",
                background: "rgba(0,0,0,0.55)",
                padding: "10px 32px",
                borderRadius: "14px",
                pointerEvents: "none",
                zIndex: 20,
                letterSpacing: "2px",
                whiteSpace: "nowrap",
              }}>
                {zoomAction === "Zoom In" ? "🔍+ Zoom In" : "🔍− Zoom Out"}
              </div>
            )}
          </div>
        </article>
      </section>

      {permissionOpen ? (
        <div className="modal-backdrop">
          <div className="modal">
            <h3>Permission Request</h3>
            <p>Before starting, confirm camera permission and optional controls.</p>
            <label>
              <input
                type="checkbox"
                checked={cameraAllowed}
                onChange={(e) => setCameraAllowed(e.target.checked)}
              />
              Allow camera access (required)
            </label>
            <label>
              <input
                type="checkbox"
                checked={controlsAllowed}
                onChange={(e) => setControlsAllowed(e.target.checked)}
              />
              Allow controls (optional)
            </label>
            <div className="modal-actions">
              <button className="btn" onClick={() => setPermissionOpen(false)}>Cancel</button>
              <button
                className="btn primary"
                onClick={async () => {
                  setPermissionOpen(false);
                  await startAppWithPermissions();
                }}
              >
                Allow and Run
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </main>
    </ErrorBoundary>
  );
}
