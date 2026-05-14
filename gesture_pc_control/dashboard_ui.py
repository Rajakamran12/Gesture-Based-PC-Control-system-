import time
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
from PIL import Image, ImageTk


class DashboardUI:
    """Tkinter UI with a consent landing page and live dashboard view."""

    def __init__(
        self,
        title: str = "Gesture-Based PC Control Dashboard",
        width: int = 960,
        height: int = 540,
        auto_start_permissions: Optional[Tuple[bool, bool]] = None,
    ):
        self.width = width
        self.height = height

        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width + 80}x{height + 320}")
        self.root.minsize(width + 40, height + 260)

        self._running = False
        self._frame_callback: Optional[Callable[[], Optional[Dict[str, Any]]]] = None
        self._controls_toggle_callback: Optional[Callable[[bool], None]] = None
        self._start_callback: Optional[Callable[[bool, bool], bool]] = None
        self._last_time = time.perf_counter()
        self._fps = 0.0
        self._controls_enabled = False
        self._dashboard_ready = False
        self._target_frame_interval_ms = 33
        self._camera_allowed = tk.BooleanVar(value=False)
        self._pc_allowed = tk.BooleanVar(value=False)
        self._auto_start_permissions = auto_start_permissions

        if self._auto_start_permissions is not None:
            self._camera_allowed.set(bool(self._auto_start_permissions[0]))
            self._pc_allowed.set(bool(self._auto_start_permissions[1]))

        self._build_layout()

        self._photo: Optional[ImageTk.PhotoImage] = None
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        self.root.bind("<c>", lambda _event: self._toggle_controls())
        self.root.bind("<C>", lambda _event: self._toggle_controls())
        self.root.bind("<q>", lambda _event: self.stop())
        self.root.bind("<Q>", lambda _event: self.stop())

        self.show_consent_view()

    def _build_layout(self) -> None:
        self.container = ttk.Frame(self.root, padding=16)
        self.container.pack(fill=tk.BOTH, expand=True)

        self.consent_frame = ttk.Frame(self.container)
        self.dashboard_frame = ttk.Frame(self.container)

        self._build_consent_frame()
        self._build_dashboard_frame()

    def _build_consent_frame(self) -> None:
        header = ttk.Label(
            self.consent_frame,
            text="Gesture-Based PC Control",
            font=("Segoe UI", 20, "bold"),
        )
        header.pack(anchor=tk.W, pady=(0, 10))

        desc = ttk.Label(
            self.consent_frame,
            text=(
                "This desktop app needs your approval before it starts the camera feed "
                "and enables PC control features."
            ),
            wraplength=self.width,
            font=("Segoe UI", 11),
        )
        desc.pack(anchor=tk.W, pady=(0, 16))

        permissions_box = ttk.LabelFrame(self.consent_frame, text="Permissions")
        permissions_box.pack(fill=tk.X, pady=(0, 16))

        ttk.Checkbutton(
            permissions_box,
            text="Allow camera access and automatically open webcam feed",
            variable=self._camera_allowed,
        ).pack(anchor=tk.W, padx=12, pady=(10, 4))

        ttk.Checkbutton(
            permissions_box,
            text="Allow access to PC controls (mouse / keyboard actions)",
            variable=self._pc_allowed,
        ).pack(anchor=tk.W, padx=12, pady=(4, 10))

        self.permission_hint = ttk.Label(
            self.consent_frame,
            text=(
                "Tip: Desktop apps cannot request browser-style OS permissions, so this screen "
                "serves as the app's access gate."
            ),
            wraplength=self.width,
            foreground="#666666",
            font=("Segoe UI", 10),
        )
        self.permission_hint.pack(anchor=tk.W, pady=(0, 12))

        self.start_button = ttk.Button(self.consent_frame, text="Grant Access and Start", command=self._on_start_clicked)
        self.start_button.pack(anchor=tk.W)

    def _build_dashboard_frame(self) -> None:
        self.video_label = ttk.Label(self.dashboard_frame)
        self.video_label.pack()

        info_row = ttk.Frame(self.dashboard_frame)
        info_row.pack(fill=tk.X, pady=(12, 0))

        self.gesture_var = tk.StringVar(value="Gesture: None")
        self.raw_gesture_var = tk.StringVar(value="Raw: None")
        self.confidence_var = tk.StringVar(value="Confidence: 0.00")
        self.fps_var = tk.StringVar(value="FPS: 0.00")
        self.hands_var = tk.StringVar(value="Hands: 0")
        self.distance_var = tk.StringVar(value="Distance: n/a")
        self.controls_var = tk.StringVar(value="Controls: OFF (Press C)")
        self.action_var = tk.StringVar(value="Action: No action")
        self.latency_var = tk.StringVar(value="Latency: 0.0ms")
        self.optimization_var = tk.StringVar(value="Consensus: 0.00 | Jitter: 0.00")
        self.features_var = tk.StringVar(value="Features: 0")
        self.runtime_build_var = tk.StringVar(value="Build: legacy")
        self.runtime_profile_var = tk.StringVar(value="Profile: unknown")
        self.runtime_model_var = tk.StringVar(value="Model: rule")

        ttk.Label(info_row, textvariable=self.gesture_var, font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(info_row, textvariable=self.raw_gesture_var, font=("Segoe UI", 11)).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(info_row, textvariable=self.confidence_var, font=("Segoe UI", 11)).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(info_row, textvariable=self.fps_var, font=("Segoe UI", 11)).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(info_row, textvariable=self.hands_var, font=("Segoe UI", 11)).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(info_row, textvariable=self.distance_var, font=("Segoe UI", 11)).pack(side=tk.LEFT)

        status_row = ttk.Frame(self.dashboard_frame)
        status_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(status_row, textvariable=self.controls_var, font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(status_row, textvariable=self.action_var, font=("Segoe UI", 11)).pack(side=tk.LEFT)

        perf_row = ttk.Frame(self.dashboard_frame)
        perf_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(perf_row, textvariable=self.features_var, font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(perf_row, textvariable=self.latency_var, font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(perf_row, textvariable=self.optimization_var, font=("Segoe UI", 10)).pack(side=tk.LEFT)

        runtime_row = ttk.Frame(self.dashboard_frame)
        runtime_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(runtime_row, textvariable=self.runtime_build_var, font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(runtime_row, textvariable=self.runtime_profile_var, font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(runtime_row, textvariable=self.runtime_model_var, font=("Segoe UI", 10)).pack(side=tk.LEFT)

        ttk.Label(self.dashboard_frame, text="Landmark Debug", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(8, 0))
        self.landmark_text = tk.Text(self.dashboard_frame, height=4, wrap="word", font=("Consolas", 9))
        self.landmark_text.pack(fill=tk.X)
        self.landmark_text.insert("1.0", "No landmarks detected")
        self.landmark_text.configure(state=tk.DISABLED)

        hint = ttk.Label(
            self.dashboard_frame,
            text="Press C to toggle PC controls, Press Q to exit",
            foreground="#666666",
            font=("Segoe UI", 10),
        )
        hint.pack(anchor=tk.W, pady=(8, 0))

    def show_consent_view(self) -> None:
        self._dashboard_ready = False
        self.dashboard_frame.pack_forget()
        self.consent_frame.pack(fill=tk.BOTH, expand=True)

    def show_dashboard_view(self) -> None:
        self.consent_frame.pack_forget()
        self.dashboard_frame.pack(fill=tk.BOTH, expand=True)
        self._dashboard_ready = True

    def set_start_callback(self, start_callback: Callable[[bool, bool], bool]) -> None:
        self._start_callback = start_callback

    def set_control_toggle_callback(self, toggle_callback: Callable[[bool], None]) -> None:
        self._controls_toggle_callback = toggle_callback

    def set_frame_provider(self, frame_callback: Callable[[], Optional[Dict[str, Any]]]) -> None:
        self._frame_callback = frame_callback

    def start(self) -> None:
        self._running = True
        self._schedule_next_frame()
        if self._auto_start_permissions is not None:
            self.root.after(120, self._attempt_auto_start)
        self.root.mainloop()

    def _attempt_auto_start(self) -> None:
        if self._start_callback is None or self._auto_start_permissions is None:
            return

        camera_allowed, pc_allowed = self._auto_start_permissions
        if not camera_allowed:
            self.permission_hint.configure(text="Camera access is required to start the app.", foreground="#a94442")
            return

        started = self._start_callback(camera_allowed, pc_allowed)
        if started:
            self.show_dashboard_view()
        else:
            self.permission_hint.configure(text="Camera could not be opened. Close other apps and try again.", foreground="#a94442")

    def stop(self) -> None:
        self._running = False
        self.root.quit()
        self.root.destroy()

    def _schedule_next_frame(self) -> None:
        if not self._running:
            return

        loop_start = time.perf_counter()

        if self._dashboard_ready and self._frame_callback is not None:
            payload = self._frame_callback()
            if payload is not None:
                self._update_dashboard(payload)

        elapsed_ms = int((time.perf_counter() - loop_start) * 1000)
        next_delay = max(1, self._target_frame_interval_ms - elapsed_ms)
        self.root.after(next_delay, self._schedule_next_frame)

    def _on_start_clicked(self) -> None:
        camera_allowed = bool(self._camera_allowed.get())
        pc_allowed = bool(self._pc_allowed.get())

        if not camera_allowed:
            self.permission_hint.configure(text="Camera access is required to start the app.", foreground="#a94442")
            return

        if self._start_callback is None:
            return

        started = self._start_callback(camera_allowed, pc_allowed)
        if started:
            self.show_dashboard_view()
        else:
            self.permission_hint.configure(text="Camera could not be opened. Close other apps and try again.", foreground="#a94442")

    def _toggle_controls(self) -> None:
        self._controls_enabled = not self._controls_enabled
        state_text = "ON" if self._controls_enabled else "OFF"
        self.controls_var.set(f"Controls: {state_text} (Press C)")
        if self._controls_toggle_callback is not None:
            self._controls_toggle_callback(self._controls_enabled)

    def _update_dashboard(self, payload: Dict[str, Any]) -> None:
        frame = payload.get("frame")
        if frame is None:
            return

        now = time.perf_counter()
        delta = now - self._last_time
        if delta > 0:
            instant_fps = 1.0 / delta
            self._fps = 0.9 * self._fps + 0.1 * instant_fps if self._fps > 0 else instant_fps
        self._last_time = now

        gesture_text = payload.get("gesture", "None")
        raw_gesture_text = payload.get("raw_gesture", "None")
        confidence = float(payload.get("gesture_confidence", 0.0))
        classification_source = str(payload.get("classification_source", "rule"))
        landmarks: List[Dict[str, Any]] = payload.get("landmarks", [])
        distance_ft = payload.get("distance_ft")
        controls_enabled = bool(payload.get("controls_enabled", False))
        action_text = payload.get("action", "No action")
        landmark_debug = payload.get("landmark_debug", "No landmarks detected")
        consensus_ratio = float(payload.get("consensus_ratio", 0.0))
        jitter_index = float(payload.get("jitter_index", 0.0))
        latency_ms = float(payload.get("pipeline_latency_ms", 0.0))
        feature_dim = int(payload.get("feature_dim", 0))
        modules_version = str(payload.get("modules_version", "legacy"))
        smooth_profile = str(payload.get("smooth_profile", "unknown"))
        model_loaded = bool(payload.get("model_loaded", False))

        self.gesture_var.set(f"Gesture: {gesture_text}")
        self.raw_gesture_var.set(f"Raw: {raw_gesture_text}")
        self.confidence_var.set(f"Confidence: {confidence:.2f}")
        self.fps_var.set(f"FPS: {self._fps:.2f}")
        self.hands_var.set(f"Hands: {len(landmarks)}")
        self.distance_var.set(
            f"Distance: {distance_ft:.2f} ft" if isinstance(distance_ft, (int, float)) else "Distance: n/a"
        )
        self._controls_enabled = controls_enabled
        self.controls_var.set(f"Controls: {'ON' if controls_enabled else 'OFF'} (Press C)")
        self.action_var.set(f"Action: {action_text}")
        self.features_var.set(f"Features: {feature_dim}")
        self.latency_var.set(f"Latency: {latency_ms:.1f}ms")
        self.optimization_var.set(f"Consensus: {consensus_ratio:.2f} | Jitter: {jitter_index:.2f}")
        self.runtime_build_var.set(f"Build: modules {modules_version}")
        self.runtime_profile_var.set(f"Profile: {smooth_profile}")
        self.runtime_model_var.set(f"Model: {classification_source} ({'loaded' if model_loaded else 'fallback'})")

        self.landmark_text.configure(state=tk.NORMAL)
        self.landmark_text.delete("1.0", tk.END)
        self.landmark_text.insert("1.0", landmark_debug)
        self.landmark_text.configure(state=tk.DISABLED)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=self._photo)
