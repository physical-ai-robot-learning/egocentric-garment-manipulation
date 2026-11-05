"""Recording UI (PySide6). Global RGB-D on top (RGB / depth toggle), left + right wrist below; protocol banner with countdown,
Start / Stop / Cancel / Save, episode metadata form, live camera health, and the QA report of the last take.

Keys: Space = Start / Stop, Esc = Cancel, Ctrl+S = Save, D = depth toggle.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .config import AppConfig
from .session import PROMPT, CollectorSession, State

STATE_COLOR = {State.IDLE: "#57606a", State.INIT: "#bf8700", State.MANIPULATION: "#1a7f37", State.END: "#bf8700", State.REVIEW: "#0969da"}


def to_pixmap(bgr: np.ndarray, w: int, h: int) -> QtGui.QPixmap:
    s = min(w / bgr.shape[1], h / bgr.shape[0])
    img = cv2.resize(bgr, (max(1, int(bgr.shape[1] * s)), max(1, int(bgr.shape[0] * s))), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    q = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(q.copy())


def depth_vis(d: np.ndarray, scale_mm: float = 1.0) -> np.ndarray:
    mm = d.astype(np.float32) * scale_mm
    v = np.clip((mm - 300) / (2000 - 300), 0, 1)
    out = cv2.applyColorMap((255 - v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[d == 0] = 0
    return out


class View(QtWidgets.QLabel):
    def __init__(self, title: str):
        super().__init__(title)
        self.setAlignment(QtCore.Qt.AlignCenter); self.setMinimumSize(320, 180)
        self.setStyleSheet("background:#111;color:#888;border-radius:6px")
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)

    def show_bgr(self, bgr):
        self.setPixmap(to_pixmap(bgr, self.width(), self.height()))


class Main(QtWidgets.QWidget):
    def __init__(self, cfg: AppConfig, session: CollectorSession):
        super().__init__()
        self.cfg, self.s = cfg, session
        self.setWindowTitle(f"Laundry collector v1 - {cfg.hardware.profile} - {cfg.dataset_dir}")
        self.show_depth = False
        # views
        self.v_global, self.v_left, self.v_right = View("global"), View("left wrist"), View("right wrist")
        views = QtWidgets.QGridLayout(); views.addWidget(self.v_global, 0, 0, 1, 2); views.addWidget(self.v_left, 1, 0); views.addWidget(self.v_right, 1, 1)
        views.setRowStretch(0, 3); views.setRowStretch(1, 2)
        # side panel
        self.banner = QtWidgets.QLabel(); self.banner.setWordWrap(True); self.banner.setMinimumHeight(70)
        self.b_start, self.b_stop = QtWidgets.QPushButton("Start Recording  [Space]"), QtWidgets.QPushButton("Stop  [Space]")
        self.b_cancel, self.b_save = QtWidgets.QPushButton("Cancel Episode  [Esc]"), QtWidgets.QPushButton("Save Episode  [Ctrl+S]")
        self.b_depth = QtWidgets.QPushButton("RGB / Depth  [D]"); self.b_depth.setCheckable(True)
        for b in (self.b_start, self.b_stop, self.b_cancel, self.b_save):
            b.setMinimumHeight(36)
        self.b_start.clicked.connect(self.on_start); self.b_stop.clicked.connect(self.on_stop)
        self.b_cancel.clicked.connect(self.on_cancel); self.b_save.clicked.connect(self.on_save)
        self.b_depth.toggled.connect(lambda v: setattr(self, "show_depth", v))
        form = QtWidgets.QFormLayout()
        self.f_id = QtWidgets.QLabel(); self.f_task = QtWidgets.QLineEdit(cfg.collector.task)
        self.f_garment = QtWidgets.QComboBox(); self.f_garment.setEditable(True); self.f_garment.addItems(cfg.collector.garments)
        self.f_init = QtWidgets.QComboBox(); self.f_init.addItems(cfg.collector.initial_states)
        self.f_ok, self.f_fail = QtWidgets.QRadioButton("Success"), QtWidgets.QRadioButton("Failure")
        self.f_grp = QtWidgets.QButtonGroup(self); self.f_grp.setExclusive(False); self.f_grp.addButton(self.f_ok); self.f_grp.addButton(self.f_fail)
        self.f_ok.toggled.connect(lambda v: v and self.f_fail.setChecked(False)); self.f_fail.toggled.connect(lambda v: v and self.f_ok.setChecked(False))
        sf = QtWidgets.QHBoxLayout(); sf.addWidget(self.f_ok); sf.addWidget(self.f_fail)
        self.f_notes = QtWidgets.QPlainTextEdit(); self.f_notes.setMaximumHeight(70)
        self.f_operator = QtWidgets.QLineEdit(cfg.collector.operator)
        form.addRow("Episode ID", self.f_id); form.addRow("Task", self.f_task); form.addRow("Garment ID", self.f_garment)
        form.addRow("Initial condition", self.f_init); form.addRow("Result", sf); form.addRow("Operator", self.f_operator); form.addRow("Notes", self.f_notes)
        self.health = QtWidgets.QLabel(); self.health.setTextFormat(QtCore.Qt.RichText)
        self.report = QtWidgets.QPlainTextEdit(); self.report.setReadOnly(True); self.report.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        side = QtWidgets.QVBoxLayout()
        side.addWidget(self.banner)
        for row in ((self.b_start, self.b_stop), (self.b_cancel, self.b_save)):
            h = QtWidgets.QHBoxLayout(); [h.addWidget(b) for b in row]; side.addLayout(h)
        side.addWidget(self.b_depth); side.addLayout(form)
        side.addWidget(QtWidgets.QLabel("<b>Cameras</b>")); side.addWidget(self.health)
        side.addWidget(QtWidgets.QLabel("<b>Last take QA</b>")); side.addWidget(self.report, 1)
        sw = QtWidgets.QWidget(); sw.setLayout(side); sw.setFixedWidth(420)
        root = QtWidgets.QHBoxLayout(self); root.addLayout(views, 1); root.addWidget(sw)
        for key, fn in (("Space", self.on_space), ("Escape", self.on_cancel), ("Ctrl+S", self.on_save), ("D", self.b_depth.toggle)):
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=fn)
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.refresh); self.timer.start(int(1000 / cfg.collector.preview_hz))
        self.resize(1500, 900)

    # ------------------------------------------------------------------ actions
    def form(self) -> dict:
        return dict(task=self.f_task.text().strip(), garment_id=self.f_garment.currentText().strip(), initial_state=self.f_init.currentText(),
                    success=True if self.f_ok.isChecked() else False if self.f_fail.isChecked() else None,
                    notes=self.f_notes.toPlainText().strip(), operator=self.f_operator.text().strip())

    def on_space(self):
        (self.on_start if self.s.state == State.IDLE else self.on_stop)()

    def on_start(self):
        if self.s.start():
            self.f_ok.setChecked(False); self.f_fail.setChecked(False); self.f_notes.clear(); self.report.clear()

    def on_stop(self):
        self.s.stop()

    def on_cancel(self):
        if self.s.state in (State.INIT, State.MANIPULATION, State.END, State.REVIEW):
            if QtWidgets.QMessageBox.question(self, "Cancel episode", "Discard this episode? (moved to _discarded/)") == QtWidgets.QMessageBox.Yes:
                self.s.cancel()

    def on_save(self):
        self.s.save(self.form())

    # ------------------------------------------------------------------ refresh
    def refresh(self):
        s = self.s
        s.tick()
        g = s.sources.get("global"); f = g.latest() if g else None
        if f is not None:
            self.v_global.show_bgr(depth_vis(f.depth, g.info.get("depth_scale_mm_per_unit") or 1.0) if (self.show_depth and f.depth is not None) else f.image)
        for v, n in ((self.v_left, "left_wrist"), (self.v_right, "right_wrist")):
            fr = s.sources[n].latest() if n in s.sources else None
            if fr is not None:
                v.show_bgr(fr.image)
        cd = s.countdown()
        extra = f"  {max(0.0, cd):.1f} s" if s.state in (State.INIT, State.END) else (f"  {cd:.0f} s" if s.state == State.MANIPULATION else "")
        self.banner.setText(f"<div style='font-size:15px'><b>{s.state.value}</b>{extra}</div><div style='font-size:13px'>{PROMPT[s.state]}</div>"
                            + (f"<div style='font-size:11px;color:#ddd'>{s.message}</div>" if s.message else ""))
        self.banner.setStyleSheet(f"background:{STATE_COLOR[s.state]};color:white;padding:8px;border-radius:6px")
        self.f_id.setText(f"{s.eid:06d}" if s.state != State.IDLE and s.eid else f"{s.store.next_id():06d} (next)")
        self.b_start.setEnabled(s.state == State.IDLE); self.b_stop.setEnabled(s.state == State.MANIPULATION)
        self.b_cancel.setEnabled(s.state != State.IDLE); self.b_save.setEnabled(s.state == State.REVIEW)
        rows = []
        for n, src in s.sources.items():
            st = src.snapshot_status()
            col = "#1a7f37" if st.connected and not st.error else "#cf222e"
            rows.append(f"<span style='color:{col}'>●</span> {n}: {st.fps:.1f} fps, drops {st.drops}, age {st.age_ms or 0:.0f} ms"
                        + (f" <span style='color:#cf222e'>{st.error}</span>" if st.error else ""))
        self.health.setText("<br>".join(rows))
        if s.state == State.REVIEW and not self.report.toPlainText() and s.last_report_txt:
            self.report.setPlainText(s.last_report_txt)

    def closeEvent(self, e):
        if self.s.state in (State.INIT, State.MANIPULATION, State.END, State.REVIEW):
            if QtWidgets.QMessageBox.question(self, "Quit", "An episode is open. Discard it and quit?") != QtWidgets.QMessageBox.Yes:
                e.ignore(); return
            self.s.cancel()
        self.s.close(); e.accept()


def run_ui(cfg: AppConfig) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    s = CollectorSession(cfg)
    s.open()
    w = Main(cfg, s); w.show()
    sys.exit(app.exec())
