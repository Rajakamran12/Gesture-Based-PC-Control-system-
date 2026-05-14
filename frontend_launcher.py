import datetime
import os
import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk


class FrontendLauncher:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("AI Driving Command Center")
        self.root.geometry("1240x760")
        self.root.minsize(1120, 700)
        self.root.configure(bg="#edf1ea")

        self.project_root = Path(__file__).resolve().parent
        self.python_exe = self.project_root / ".venv" / "Scripts" / "python.exe"

        self.process: subprocess.Popen[str] | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()

        self.target_var = tk.StringVar(value="Gesture Runtime Dashboard v2 (Recommended)")
        self.open_quiet_var = tk.BooleanVar(value=False)
        self.auto_stop_var = tk.BooleanVar(value=True)
        self.app_state_var = tk.StringVar(value="Stopped")
        self.app_target_var = tk.StringVar(value="Gesture Runtime Dashboard v2")
        self.time_var = tk.StringVar(value="--:--:--")
        self.activity_count_var = tk.StringVar(value="0")
        self.log_count_var = tk.StringVar(value="0")

        self.activity_count = 0
        self.log_count = 0
        self.health_score = 0
        self.current_section = "Overview"
        self._max_log_lines = 900
        self._lines_since_trim = 0

        self.preview_capture: cv2.VideoCapture | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.preview_message_var = tk.StringVar(value="Initializing camera preview...")
        self.permission_camera_var = tk.BooleanVar(value=True)
        self.permission_pc_var = tk.BooleanVar(value=False)

        self.card_status: dict[str, tk.Label] = {}
        self.status_health_canvas: tk.Canvas | None = None
        self.nav_buttons: dict[str, tk.Button] = {}
        self.metric_row_frame: tk.Frame | None = None
        self.runtime_row_frame: tk.Frame | None = None
        self.logs_outer_frame: tk.Frame | None = None
        self.donut_canvas: tk.Canvas | None = None
        self.preview_canvas: tk.Canvas | None = None

        self._configure_styles()
        self._build_ui()
        self._set_section("Overview")
        self._refresh_status()
        self._poll_logs()
        self._tick_clock()
        self._update_preview_tile()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure(
            "Modern.TCombobox",
            fieldbackground="#f6f8f4",
            background="#f6f8f4",
            foreground="#12332a",
            bordercolor="#ccd9cf",
            lightcolor="#ccd9cf",
            darkcolor="#ccd9cf",
            arrowsize=14,
            padding=6,
        )

    def _build_ui(self) -> None:
        main = tk.Frame(self.root, bg="#edf1ea")
        main.pack(fill=tk.BOTH, expand=True)

        self._build_sidebar(main)
        self._build_content(main)

    def _build_sidebar(self, parent: tk.Widget) -> None:
        sidebar = tk.Frame(parent, bg="#183c32", width=235)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        brand = tk.Label(
            sidebar,
            text="DriveFlow",
            bg="#183c32",
            fg="#f4f7f1",
            font=("Bahnschrift", 20, "bold"),
            anchor="w",
            padx=20,
            pady=22,
        )
        brand.pack(fill=tk.X)

        nav_title = tk.Label(
            sidebar,
            text="Navigation",
            bg="#183c32",
            fg="#9cc3b0",
            font=("Bahnschrift", 10),
            anchor="w",
            padx=20,
            pady=8,
        )
        nav_title.pack(fill=tk.X)

        for label, active in [
            ("Overview", True),
            ("Runtime", False),
            ("Options", False),
            ("Logs", False),
        ]:
            bg = "#235344" if active else "#183c32"
            fg = "#f2f7f3" if active else "#b5d0c4"
            button = tk.Button(
                sidebar,
                text=label,
                command=lambda section=label: self._set_section(section),
                bg=bg,
                fg=fg,
                activebackground="#2a5f4f",
                activeforeground="#ffffff",
                relief=tk.FLAT,
                borderwidth=0,
                font=("Bahnschrift", 12),
                anchor="w",
                padx=22,
                pady=10,
            )
            button.pack(fill=tk.X, padx=10, pady=2)
            self.nav_buttons[label] = button

        spacer = tk.Frame(sidebar, bg="#183c32")
        spacer.pack(fill=tk.BOTH, expand=True)

        self.sidebar_status = tk.Label(
            sidebar,
            text="System ready",
            bg="#183c32",
            fg="#9cc3b0",
            font=("Bahnschrift", 10),
            anchor="w",
            padx=20,
            pady=14,
        )
        self.sidebar_status.pack(fill=tk.X)

    def _build_content(self, parent: tk.Widget) -> None:
        content = tk.Frame(parent, bg="#edf1ea")
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_header(content)
        self._build_metric_cards(content)
        self._build_runtime_row(content)
        self._build_logs_panel(content)

    def _build_header(self, parent: tk.Widget) -> None:
        header = tk.Frame(parent, bg="#edf1ea", padx=20, pady=16)
        header.pack(fill=tk.X)

        left = tk.Frame(header, bg="#edf1ea")
        left.pack(side=tk.LEFT)

        tk.Label(
            left,
            text="Operations Dashboard",
            bg="#edf1ea",
            fg="#102820",
            font=("Bahnschrift", 26, "bold"),
        ).pack(anchor="w")

        tk.Label(
            left,
            text="Launch and monitor your gesture control stack from one control room.",
            bg="#edf1ea",
            fg="#48645a",
            font=("Candara", 12),
        ).pack(anchor="w", pady=(2, 0))

        right = tk.Frame(header, bg="#edf1ea")
        right.pack(side=tk.RIGHT)

        self.badge = tk.Label(
            right,
            text="APP STOPPED",
            bg="#dfe9e1",
            fg="#324f42",
            padx=12,
            pady=6,
            font=("Bahnschrift", 10, "bold"),
        )
        self.badge.pack(anchor="e")

        tk.Label(
            right,
            textvariable=self.time_var,
            bg="#edf1ea",
            fg="#3a5a4f",
            font=("Consolas", 12, "bold"),
            pady=6,
        ).pack(anchor="e")

    def _create_metric_card(self, parent: tk.Widget, title: str, value_var: tk.StringVar, value_color: str) -> tk.Frame:
        card = tk.Frame(parent, bg="#f7faf5", highlightbackground="#d8e3da", highlightthickness=1, padx=14, pady=10)
        tk.Label(card, text=title, bg="#f7faf5", fg="#4e675d", font=("Candara", 11)).pack(anchor="w")
        value_label = tk.Label(card, textvariable=value_var, bg="#f7faf5", fg=value_color, font=("Bahnschrift", 16, "bold"))
        value_label.pack(anchor="w", pady=(4, 0))
        return card

    def _build_metric_cards(self, parent: tk.Widget) -> None:
        row = tk.Frame(parent, bg="#edf1ea", padx=20)
        row.pack(fill=tk.X, pady=(0, 10))
        self.metric_row_frame = row

        self.card_status_vars = {
            "Runtime State": self.app_state_var,
            "Selected Target": self.app_target_var,
            "Activity Events": self.activity_count_var,
            "Log Entries": self.log_count_var,
        }

        colors = {
            "Runtime State": "#1b6f4d",
            "Selected Target": "#2f4a8f",
            "Activity Events": "#7f5d16",
            "Log Entries": "#5e2f8f",
        }

        for key in self.card_status_vars:
            card = self._create_metric_card(row, key, self.card_status_vars[key], colors[key])
            card.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
            self.card_status[key] = card

    def _build_runtime_row(self, parent: tk.Widget) -> None:
        row = tk.Frame(parent, bg="#edf1ea", padx=20)
        row.pack(fill=tk.BOTH, expand=False)
        self.runtime_row_frame = row

        options_card = tk.Frame(row, bg="#f8fbf7", highlightbackground="#d8e3da", highlightthickness=1, padx=14, pady=14)
        options_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            options_card,
            text="Launch Control",
            bg="#f8fbf7",
            fg="#1a3f35",
            font=("Bahnschrift", 15, "bold"),
        ).pack(anchor="w")

        tk.Label(
            options_card,
            text="Target",
            bg="#f8fbf7",
            fg="#4b665c",
            font=("Candara", 11),
            pady=6,
        ).pack(anchor="w")

        self.target_combo = ttk.Combobox(
            options_card,
            textvariable=self.target_var,
            style="Modern.TCombobox",
            state="readonly",
            values=[
                "Gesture Runtime Dashboard v2 (Recommended)",
                "Legacy app.py",
                "Robot test.py",
            ],
        )
        self.target_combo.pack(fill=tk.X)

        toggles = tk.Frame(options_card, bg="#f8fbf7")
        toggles.pack(fill=tk.X, pady=(12, 8))

        tk.Checkbutton(
            toggles,
            text="Run without attached console",
            variable=self.open_quiet_var,
            bg="#f8fbf7",
            fg="#2d4b40",
            activebackground="#f8fbf7",
            activeforeground="#2d4b40",
            selectcolor="#ecf4ee",
            font=("Candara", 11),
        ).pack(anchor="w")

        tk.Checkbutton(
            toggles,
            text="Auto-stop existing instance before launch",
            variable=self.auto_stop_var,
            bg="#f8fbf7",
            fg="#2d4b40",
            activebackground="#f8fbf7",
            activeforeground="#2d4b40",
            selectcolor="#ecf4ee",
            font=("Candara", 11),
        ).pack(anchor="w", pady=(4, 0))

        action_row = tk.Frame(options_card, bg="#f8fbf7")
        action_row.pack(fill=tk.X, pady=(10, 0))

        self.run_btn = tk.Button(
            action_row,
            text="Run App",
            command=self._run_app,
            bg="#1f8f5f",
            fg="#ffffff",
            activebackground="#18734d",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=18,
            pady=10,
            font=("Bahnschrift", 11, "bold"),
            cursor="hand2",
        )
        self.run_btn.pack(side=tk.LEFT)

        self.stop_btn = tk.Button(
            action_row,
            text="Stop App",
            command=self._stop_app,
            bg="#d8e0da",
            fg="#25463a",
            activebackground="#c4d2c8",
            activeforeground="#1f3b32",
            relief=tk.FLAT,
            padx=18,
            pady=10,
            font=("Bahnschrift", 11, "bold"),
            cursor="hand2",
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(8, 0))

        refresh_btn = tk.Button(
            action_row,
            text="Refresh",
            command=self._refresh_status,
            bg="#ebefe9",
            fg="#2d4b40",
            activebackground="#dde7df",
            relief=tk.FLAT,
            padx=16,
            pady=10,
            font=("Bahnschrift", 11),
            cursor="hand2",
        )
        refresh_btn.pack(side=tk.LEFT, padx=(8, 0))

        right_panel = tk.Frame(row, bg="#edf1ea")
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(10, 0), expand=True)

        health_card = tk.Frame(right_panel, bg="#f8fbf7", highlightbackground="#d8e3da", highlightthickness=1, padx=14, pady=12)
        health_card.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            health_card,
            text="System Health",
            bg="#f8fbf7",
            fg="#1a3f35",
            font=("Bahnschrift", 15, "bold"),
        ).pack(anchor="w")

        self.health_lines = {
            "venv": tk.StringVar(value=".venv: checking"),
            "python": tk.StringVar(value="python: checking"),
            "models": tk.StringVar(value="models: checking"),
        }

        for key in ["venv", "python", "models"]:
            tk.Label(
                health_card,
                textvariable=self.health_lines[key],
                bg="#f8fbf7",
                fg="#2f4b40",
                font=("Candara", 11),
                pady=2,
            ).pack(anchor="w")

        self.status_health_canvas = tk.Canvas(
            health_card,
            width=420,
            height=72,
            bg="#f8fbf7",
            bd=0,
            highlightthickness=0,
        )
        self.status_health_canvas.pack(anchor="w", pady=(10, 2))

        analytics_row = tk.Frame(health_card, bg="#f8fbf7")
        analytics_row.pack(fill=tk.X, pady=(8, 0))

        donut_card = tk.Frame(analytics_row, bg="#f3f8f2", highlightbackground="#d8e3da", highlightthickness=1, padx=8, pady=8)
        donut_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            donut_card,
            text="Runtime Score",
            bg="#f3f8f2",
            fg="#2c4d40",
            font=("Bahnschrift", 11, "bold"),
        ).pack(anchor="w")

        self.donut_canvas = tk.Canvas(
            donut_card,
            width=160,
            height=110,
            bg="#f3f8f2",
            bd=0,
            highlightthickness=0,
        )
        self.donut_canvas.pack(anchor="w", pady=(4, 0))

        preview_card = tk.Frame(analytics_row, bg="#f3f8f2", highlightbackground="#d8e3da", highlightthickness=1, padx=8, pady=8)
        preview_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

        tk.Label(
            preview_card,
            text="Live Camera Tile",
            bg="#f3f8f2",
            fg="#2c4d40",
            font=("Bahnschrift", 11, "bold"),
        ).pack(anchor="w")

        self.preview_canvas = tk.Canvas(
            preview_card,
            width=190,
            height=110,
            bg="#101a1a",
            bd=0,
            highlightthickness=0,
        )
        self.preview_canvas.pack(anchor="w", pady=(4, 0))

        tk.Label(
            preview_card,
            textvariable=self.preview_message_var,
            bg="#f3f8f2",
            fg="#4f6b60",
            font=("Candara", 10),
        ).pack(anchor="w", pady=(4, 0))

        activity_card = tk.Frame(right_panel, bg="#f8fbf7", highlightbackground="#d8e3da", highlightthickness=1, padx=14, pady=12)
        activity_card.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        tk.Label(
            activity_card,
            text="Activity Timeline",
            bg="#f8fbf7",
            fg="#1a3f35",
            font=("Bahnschrift", 15, "bold"),
        ).pack(anchor="w")

        self.activity_list = tk.Listbox(
            activity_card,
            height=5,
            bg="#f3f8f2",
            fg="#27463a",
            selectbackground="#d2e4d8",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground="#d8e3da",
            font=("Consolas", 10),
        )
        self.activity_list.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self._push_activity("dashboard ready")

    def _build_logs_panel(self, parent: tk.Widget) -> None:
        logs_outer = tk.Frame(parent, bg="#edf1ea", padx=20, pady=12)
        logs_outer.pack(fill=tk.BOTH, expand=True)
        self.logs_outer_frame = logs_outer

        logs_card = tk.Frame(logs_outer, bg="#101a1a", highlightbackground="#243838", highlightthickness=1)
        logs_card.pack(fill=tk.BOTH, expand=True)

        header = tk.Frame(logs_card, bg="#142322", padx=12, pady=8)
        header.pack(fill=tk.X)

        tk.Label(
            header,
            text="Runtime Console",
            bg="#142322",
            fg="#d7eee0",
            font=("Bahnschrift", 13, "bold"),
        ).pack(side=tk.LEFT)

        self.logs_state_label = tk.Label(
            header,
            text="waiting",
            bg="#27423f",
            fg="#daf7e8",
            font=("Consolas", 10, "bold"),
            padx=10,
            pady=3,
        )
        self.logs_state_label.pack(side=tk.RIGHT)

        self.logs = tk.Text(
            logs_card,
            height=14,
            wrap="word",
            bg="#101a1a",
            fg="#b7dbca",
            insertbackground="#e8fff2",
            borderwidth=0,
            padx=12,
            pady=10,
            font=("Consolas", 10),
        )
        self.logs.pack(fill=tk.BOTH, expand=True)
        self.logs.configure(state=tk.DISABLED)

    def _tick_clock(self) -> None:
        self.time_var.set(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.root.after(1000, self._tick_clock)

    def _draw_runtime_donut(self, score: int) -> None:
        if self.donut_canvas is None:
            return

        c = self.donut_canvas
        c.delete("all")

        center_x = 78
        center_y = 58
        radius = 38

        c.create_oval(center_x - radius, center_y - radius, center_x + radius, center_y + radius, outline="#d7e4db", width=12)
        c.create_arc(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
            start=90,
            extent=-(score * 3.6),
            style=tk.ARC,
            outline="#1f8f5f",
            width=12,
        )
        c.create_text(center_x, center_y, text=f"{score}%", fill="#21483b", font=("Bahnschrift", 14, "bold"))
        c.create_text(136, 24, text="stability", fill="#4f6b60", font=("Consolas", 9))
        c.create_text(136, 44, text="launch", fill="#4f6b60", font=("Consolas", 9))
        c.create_text(136, 64, text="health", fill="#4f6b60", font=("Consolas", 9))

    def _open_preview_camera(self) -> cv2.VideoCapture | None:
        backend_choices: list[int | None] = [None]
        if os.name == "nt":
            backend_choices = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]

        for index in [0, 1, 2]:
            for backend in backend_choices:
                cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
                if cap is not None and cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                    self.preview_message_var.set(f"Camera index {index} active")
                    return cap
                if cap is not None:
                    cap.release()
        return None

    def _update_preview_tile(self) -> None:
        if self.preview_canvas is None:
            self.root.after(900, self._update_preview_tile)
            return

        if self.process is not None and self.process.poll() is None:
            if self.preview_capture is not None:
                self.preview_capture.release()
                self.preview_capture = None
            self.preview_message_var.set("Preview paused while app is running")
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(96, 54, text="preview paused", fill="#c2d5ca", font=("Consolas", 10))
            self.root.after(1200, self._update_preview_tile)
            return

        if self.preview_capture is None:
            self.preview_capture = self._open_preview_camera()

        canvas = self.preview_canvas
        canvas.delete("all")

        if self.preview_capture is None:
            self.preview_message_var.set("No camera available")
            canvas.create_text(96, 54, text="camera unavailable", fill="#c2d5ca", font=("Consolas", 10))
            self.root.after(1500, self._update_preview_tile)
            return

        ok, frame = self.preview_capture.read()
        if not ok:
            self.preview_message_var.set("Waiting for camera frame")
            canvas.create_text(96, 54, text="waiting...", fill="#c2d5ca", font=("Consolas", 10))
            self.root.after(1200, self._update_preview_tile)
            return

        frame = cv2.resize(frame, (190, 110))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self.preview_photo = ImageTk.PhotoImage(image=image)
        canvas.create_image(0, 0, image=self.preview_photo, anchor=tk.NW)

        self.root.after(900, self._update_preview_tile)

    def _set_section(self, section: str) -> None:
        self.current_section = section

        for name, button in self.nav_buttons.items():
            if name == section:
                button.configure(bg="#235344", fg="#f2f7f3")
            else:
                button.configure(bg="#183c32", fg="#b5d0c4")

        if self.metric_row_frame is None or self.runtime_row_frame is None or self.logs_outer_frame is None:
            return

        self.metric_row_frame.pack_forget()
        self.runtime_row_frame.pack_forget()
        self.logs_outer_frame.pack_forget()

        if section == "Logs":
            self.logs_outer_frame.pack(fill=tk.BOTH, expand=True)
        elif section == "Runtime":
            self.metric_row_frame.pack(fill=tk.X, pady=(0, 10))
            self.runtime_row_frame.pack(fill=tk.BOTH, expand=True)
        elif section == "Options":
            self.runtime_row_frame.pack(fill=tk.BOTH, expand=True)
            self.logs_outer_frame.pack(fill=tk.BOTH, expand=True)
        else:
            self.metric_row_frame.pack(fill=tk.X, pady=(0, 10))
            self.runtime_row_frame.pack(fill=tk.BOTH, expand=False)
            self.logs_outer_frame.pack(fill=tk.BOTH, expand=True)

        self._push_activity(f"section opened: {section.lower()}")

    def _draw_health_chart(self, values: list[int]) -> None:
        if self.status_health_canvas is None:
            return

        canvas = self.status_health_canvas
        canvas.delete("all")

        labels = ["venv", "python", "models"]
        colors = ["#2b9b68" if score else "#cc7f3f" for score in values]
        bar_width = 100
        gap = 25
        start_x = 8
        bottom_y = 58

        for idx, score in enumerate(values):
            x0 = start_x + idx * (bar_width + gap)
            x1 = x0 + bar_width
            bar_height = 36 if score else 14
            y0 = bottom_y - bar_height
            y1 = bottom_y
            canvas.create_rectangle(x0, y0, x1, y1, fill=colors[idx], width=0)
            canvas.create_text(x0 + bar_width / 2, 66, text=labels[idx], fill="#4b665c", font=("Consolas", 9))

    def _refresh_status(self) -> None:
        venv_ok = self.python_exe.exists()
        py_ok = False
        py_version = "n/a"

        if venv_ok:
            try:
                result = subprocess.run(
                    [str(self.python_exe), "-c", "import sys; print(sys.version.split()[0])"],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                    timeout=8,
                )
                if result.returncode == 0:
                    py_ok = True
                    py_version = result.stdout.strip()
            except Exception:
                py_ok = False

        model_a = self.project_root / "model" / "keypoint_classifier" / "keypoint_classifier.tflite"
        model_b = self.project_root / "gesture_pc_control" / "models" / "hand_landmarker.task"
        models_ok = model_a.exists() and model_b.exists()

        self.health_lines["venv"].set(f".venv: {'found' if venv_ok else 'missing'}")
        self.health_lines["python"].set(f"python: {py_version if py_ok else 'unavailable'}")
        self.health_lines["models"].set(f"models: {'ready' if models_ok else 'missing files'}")

        self._draw_health_chart([1 if venv_ok else 0, 1 if py_ok else 0, 1 if models_ok else 0])

        self.health_score = 35
        self.health_score += 25 if venv_ok else 0
        self.health_score += 20 if py_ok else 0
        self.health_score += 20 if models_ok else 0
        self._draw_runtime_donut(self.health_score)

        summary = "healthy" if (venv_ok and py_ok and models_ok) else "needs attention"
        self.sidebar_status.configure(text=f"System {summary}")
        self._push_activity("status refreshed")

    def _command_for_target(self) -> list[str]:
        target = self.target_var.get()
        if target == "Legacy app.py":
            return [str(self.python_exe), "app.py"]
        if target == "Robot test.py":
            return [str(self.python_exe), "test.py"]
        return [str(self.python_exe), "gesture_pc_control/main.py"]

    def _request_permissions(self) -> tuple[bool, bool] | None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Permission Request")
        dialog.configure(bg="#f6faf6")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        panel = tk.Frame(dialog, bg="#f6faf6", padx=16, pady=14)
        panel.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            panel,
            text="Allow Access Before Starting",
            bg="#f6faf6",
            fg="#16372d",
            font=("Bahnschrift", 14, "bold"),
        ).pack(anchor="w")

        tk.Label(
            panel,
            text="Run App will open the camera. Choose what you want to allow:",
            bg="#f6faf6",
            fg="#4b665c",
            font=("Candara", 11),
            pady=6,
        ).pack(anchor="w")

        hint_var = tk.StringVar(value="")

        tk.Checkbutton(
            panel,
            text="Allow camera access (required)",
            variable=self.permission_camera_var,
            bg="#f6faf6",
            fg="#27463a",
            activebackground="#f6faf6",
            activeforeground="#27463a",
            selectcolor="#e9f2eb",
            font=("Candara", 11),
        ).pack(anchor="w", pady=(6, 2))

        tk.Checkbutton(
            panel,
            text="Allow PC controls (mouse/keyboard actions)",
            variable=self.permission_pc_var,
            bg="#f6faf6",
            fg="#27463a",
            activebackground="#f6faf6",
            activeforeground="#27463a",
            selectcolor="#e9f2eb",
            font=("Candara", 11),
        ).pack(anchor="w", pady=(2, 6))

        tk.Label(
            panel,
            textvariable=hint_var,
            bg="#f6faf6",
            fg="#a94442",
            font=("Candara", 10),
        ).pack(anchor="w")

        result: dict[str, tuple[bool, bool] | None] = {"value": None}

        def on_allow() -> None:
            camera_allowed = bool(self.permission_camera_var.get())
            pc_allowed = bool(self.permission_pc_var.get())

            if not camera_allowed:
                hint_var.set("Camera permission is required to start the dashboard.")
                return

            result["value"] = (camera_allowed, pc_allowed)
            dialog.destroy()

        def on_cancel() -> None:
            result["value"] = None
            dialog.destroy()

        actions = tk.Frame(panel, bg="#f6faf6")
        actions.pack(anchor="e", fill=tk.X, pady=(10, 0))

        tk.Button(
            actions,
            text="Cancel",
            command=on_cancel,
            bg="#dce6df",
            fg="#27463a",
            activebackground="#ccd9d1",
            relief=tk.FLAT,
            padx=12,
            pady=6,
            font=("Bahnschrift", 10),
            cursor="hand2",
        ).pack(side=tk.RIGHT)

        tk.Button(
            actions,
            text="Allow and Run",
            command=on_allow,
            bg="#1f8f5f",
            fg="#ffffff",
            activebackground="#18734d",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6,
            font=("Bahnschrift", 10, "bold"),
            cursor="hand2",
        ).pack(side=tk.RIGHT, padx=(0, 8))

        dialog.protocol("WM_DELETE_WINDOW", on_cancel)
        self.root.wait_window(dialog)
        return result["value"]

    def _set_running_visual_state(self, running: bool) -> None:
        if running:
            self.badge.configure(text="APP RUNNING", bg="#1f8f5f", fg="#f5fff8")
            self.logs_state_label.configure(text="running", bg="#1f8f5f", fg="#f5fff8")
            self.app_state_var.set("Running")
            self._draw_runtime_donut(min(99, self.health_score + 8))
        else:
            self.badge.configure(text="APP STOPPED", bg="#dfe9e1", fg="#324f42")
            self.logs_state_label.configure(text="stopped", bg="#27423f", fg="#daf7e8")
            self.app_state_var.set("Stopped")
            self._draw_runtime_donut(self.health_score)

    def _run_app(self) -> None:
        if not self.python_exe.exists():
            messagebox.showerror("Missing Environment", "Could not find .venv/Scripts/python.exe")
            self._push_activity("launch blocked: missing .venv")
            return

        if self.process is not None and self.process.poll() is None:
            if self.auto_stop_var.get():
                self._stop_app()
            else:
                messagebox.showwarning("App Running", "An app instance is already running.")
                self._push_activity("launch blocked: app already running")
                return

        cmd = self._command_for_target()
        self.app_target_var.set(self.target_var.get().replace(" (Recommended)", ""))

        launch_env = os.environ.copy()
        if self.target_var.get().startswith("Gesture Runtime Dashboard"):
            permissions = self._request_permissions()
            if permissions is None:
                self._push_activity("launch canceled: permission dialog")
                return

            camera_allowed, pc_allowed = permissions
            launch_env["DRIVEFLOW_PERMISSION_PRESET"] = "1"
            launch_env["DRIVEFLOW_CAMERA_ALLOWED"] = "1" if camera_allowed else "0"
            launch_env["DRIVEFLOW_PC_ALLOWED"] = "1" if pc_allowed else "0"

        if self.preview_capture is not None:
            self.preview_capture.release()
            self.preview_capture = None
            self.preview_message_var.set("Preview released for app camera")

        creationflags = 0
        if os.name == "nt" and self.open_quiet_var.get():
            creationflags = subprocess.CREATE_NO_WINDOW

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                env=launch_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=creationflags,
            )
        except Exception as exc:
            messagebox.showerror("Launch Failed", str(exc))
            self._push_activity("launch failed")
            return

        self._set_running_visual_state(True)
        self._append_log(f"[Launcher] started: {' '.join(cmd)}")
        self._push_activity(f"launch started: {self.app_target_var.get().lower()}")

        if self.process.stdout is not None:
            thread = threading.Thread(target=self._stream_logs, args=(self.process.stdout,), daemon=True)
            thread.start()

    def _stream_logs(self, stdout) -> None:
        try:
            for line in stdout:
                self.log_queue.put(line.rstrip("\n"))
        finally:
            self.log_queue.put("[Launcher] process exited")
            self.log_queue.put("__PROCESS_EXIT__")

    def _poll_logs(self) -> None:
        processed = 0
        max_per_tick = 80
        while True:
            if processed >= max_per_tick:
                break
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break

            if line == "__PROCESS_EXIT__":
                self._set_running_visual_state(False)
                self._push_activity("process exited")
                continue

            self._append_log(line)
            processed += 1

        next_delay = 40 if processed >= max_per_tick else 120
        self.root.after(next_delay, self._poll_logs)

    def _append_log(self, line: str) -> None:
        self.logs.configure(state=tk.NORMAL)
        self.logs.insert(tk.END, f"{line}\n")
        self.logs.see(tk.END)

        self._lines_since_trim += 1
        if self._lines_since_trim >= 40:
            line_count = int(self.logs.index("end-1c").split(".")[0])
            if line_count > self._max_log_lines:
                trim_to = line_count - self._max_log_lines
                self.logs.delete("1.0", f"{trim_to + 1}.0")
            self._lines_since_trim = 0

        self.logs.configure(state=tk.DISABLED)

        self.log_count += 1
        self.log_count_var.set(str(self.log_count))

    def _push_activity(self, event: str) -> None:
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"{stamp} | {event}"
        self.activity_list.insert(0, entry)
        if self.activity_list.size() > 12:
            self.activity_list.delete(12, tk.END)

        self.activity_count += 1
        self.activity_count_var.set(str(self.activity_count))

    def _stop_app(self) -> None:
        if self.process is None:
            self._set_running_visual_state(False)
            return

        if self.process.poll() is None:
            self.process.terminate()
            self._append_log("[Launcher] terminate signal sent")
            self._push_activity("stop requested")

        self.process = None
        self._set_running_visual_state(False)

    def _on_close(self) -> None:
        self._stop_app()
        if self.preview_capture is not None:
            self.preview_capture.release()
            self.preview_capture = None
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    FrontendLauncher().run()
