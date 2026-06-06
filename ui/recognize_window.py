import cv2
import numpy as np
import sys
import os
import time
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QFrame, QProgressBar, QSizePolicy, QSplitter
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from database.db_manager import load_all_encodings
from utils.face_processor import (
    detect_faces, get_landmarks, get_face_encoding,
    draw_landmarks, draw_face_box, compare_encodings,
    compare_encodings_multi
)

SMOOTHING_FRAMES = 5


class VideoLabel(QLabel):
    """Aspect-ratio saqlagan holda kamera oqimini ko'rsatuvchi label"""
    def __init__(self, placeholder="Kamera o'chirilgan"):
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            "background:#0a0a0a;color:#555;font-size:13px;"
            "border-radius:8px;border:1px solid #333;"
        )
        self._pixmap_orig = None

    def set_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._pixmap_orig = QPixmap.fromImage(qimg)
        self._update_scaled()

    def _update_scaled(self):
        if self._pixmap_orig is None:
            return
        pix = self._pixmap_orig.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        super().setPixmap(pix)

    def resizeEvent(self, event):
        self._update_scaled()
        super().resizeEvent(event)

    def clear_frame(self, text="Kamera o'chirilgan"):
        self._pixmap_orig = None
        self.clear()
        self.setText(text)


class RecognizeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.known_encodings = []
        self.current_frame = None
        self._result_buffer = []
        self._build_ui()
        self._reload_encodings()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        # --- Yuqori panel ---
        top = QHBoxLayout()
        title = QLabel("Tanib olish rejimi")
        title.setFont(QFont("Arial", 13, QFont.Bold))
        top.addWidget(title)
        top.addStretch()

        self.btn_reload = QPushButton("🔄  Bazani yangilash")
        self.btn_reload.setFixedHeight(34)
        self.btn_reload.setStyleSheet(
            "border:1px solid #ccc;border-radius:6px;padding:0 14px;font-size:13px;"
        )
        self.btn_reload.clicked.connect(self._reload_encodings)
        top.addWidget(self.btn_reload)

        self.btn_cam = QPushButton("▶  Kamerani yoq")
        self.btn_cam.setFixedHeight(34)
        self.btn_cam.setStyleSheet(
            "background:#1D9E75;color:white;border-radius:6px;"
            "padding:0 16px;font-size:13px;"
        )
        self.btn_cam.clicked.connect(self._toggle_camera)
        top.addWidget(self.btn_cam)
        root.addLayout(top)

        # --- Kamera panellari (50/50 bo'linadi, oynaga moslashadi) ---
        panels = QHBoxLayout()
        panels.setSpacing(12)

        self.cam_label = VideoLabel("Kamera o'chirilgan")
        self.lm_label  = VideoLabel("Kamera o'chirilgan")

        panels.addWidget(self._wrap_panel(self.cam_label, "Kamera oqimi"))
        panels.addWidget(self._wrap_panel(self.lm_label,  "Landmark nuqtalar (68 ta)"))
        root.addLayout(panels, stretch=3)   # kameralar ko'proq joy oladi

        # --- Natija paneli ---
        result_frame = QFrame()
        result_frame.setStyleSheet(
            "background:white;border:1px solid #e0e0e0;border-radius:10px;"
        )
        result_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        rl = QHBoxLayout(result_frame)
        rl.setContentsMargins(16, 12, 16, 12)
        rl.setSpacing(16)

        # Avatar
        self.avatar_label = QLabel("?")
        self.avatar_label.setFixedSize(52, 52)
        self.avatar_label.setAlignment(Qt.AlignCenter)
        self.avatar_label.setFont(QFont("Arial", 18, QFont.Bold))
        self.avatar_label.setStyleSheet(
            "background:#eee;border-radius:26px;color:#999;"
        )
        rl.addWidget(self.avatar_label)

        # Ism
        info_col = QVBoxLayout()
        self.name_label = QLabel("Kutilmoqda...")
        self.name_label.setFont(QFont("Arial", 15, QFont.Bold))
        self.name_label.setStyleSheet("color:#222;")
        self.info_label = QLabel("Kamerani yoqing va yuzingizni ko'rsating")
        self.info_label.setStyleSheet("color:#888;font-size:12px;")
        info_col.addWidget(self.name_label)
        info_col.addWidget(self.info_label)
        rl.addLayout(info_col, stretch=1)

        # Confidence
        conf_col = QVBoxLayout()
        conf_col.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.conf_label = QLabel("—")
        self.conf_label.setFont(QFont("Arial", 26, QFont.Bold))
        self.conf_label.setAlignment(Qt.AlignRight)
        self.conf_label.setStyleSheet("color:#1D9E75;")
        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setValue(0)
        self.conf_bar.setMinimumWidth(120)
        self.conf_bar.setMaximumWidth(200)
        self.conf_bar.setTextVisible(False)
        self.conf_bar.setStyleSheet("""
            QProgressBar { border:1px solid #ddd;border-radius:4px;height:8px; }
            QProgressBar::chunk { background:#1D9E75;border-radius:3px; }
        """)
        conf_col.addWidget(self.conf_label)
        conf_col.addWidget(self.conf_bar)
        rl.addLayout(conf_col)

        root.addWidget(result_frame, stretch=0)

        # --- Metrikalar ---
        metrics = QHBoxLayout()
        metrics.setSpacing(12)
        self.metric_widgets = {}
        for key, label in [("faces", "Yuzlar"), ("landmarks", "Nuqtalar"), ("time", "Vaqt (ms)")]:
            m_layout, m_val = self._make_metric(label, "—")
            self.metric_widgets[key] = m_val
            metrics.addLayout(m_layout)
        metrics.addStretch()
        root.addLayout(metrics, stretch=0)

    def _wrap_panel(self, widget, title):
        frame = QFrame()
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        t = QLabel(title)
        t.setStyleSheet("font-size:12px;color:#666;font-weight:bold;")
        layout.addWidget(t)
        layout.addWidget(widget)
        return frame

    def _make_metric(self, label, value):
        layout = QVBoxLayout()
        layout.setSpacing(2)
        lbl = QLabel(label)
        lbl.setStyleSheet(
            "font-size:11px;color:#999;background:#f0f0f0;"
            "border-radius:5px;padding:2px 10px;"
        )
        val = QLabel(value)
        val.setFont(QFont("Arial", 14, QFont.Bold))
        val.setStyleSheet("color:#333;")
        lbl.setAlignment(Qt.AlignCenter)
        val.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)
        layout.addWidget(val)
        return layout, val

    def _reload_encodings(self):
        self.known_encodings = load_all_encodings()
        self._result_buffer.clear()
        self.btn_reload.setText(f"🔄  Bazani yangilash ({len(self.known_encodings)} ta)")

    def _toggle_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Xato", "Kamera topilmadi!")
                return
            self.timer.start(33)
            self.btn_cam.setText("⏹  To'xtatish")
            self.btn_cam.setStyleSheet(
                "background:#c0392b;color:white;border-radius:6px;"
                "padding:0 16px;font-size:13px;"
            )
        else:
            self._stop_camera()

    def _stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.cam_label.clear_frame()
        self.lm_label.clear_frame()
        self.btn_cam.setText("▶  Kamerani yoq")
        self.btn_cam.setStyleSheet(
            "background:#1D9E75;color:white;border-radius:6px;"
            "padding:0 16px;font-size:13px;"
        )
        self._reset_result()
        self._result_buffer.clear()

    def _reset_result(self):
        self.name_label.setText("Kutilmoqda...")
        self.info_label.setText("Kamerani yoqing va yuzingizni ko'rsating")
        self.avatar_label.setText("?")
        self.avatar_label.setStyleSheet("background:#eee;border-radius:26px;color:#999;")
        self.conf_label.setText("—")
        self.conf_bar.setValue(0)
        for w in self.metric_widgets.values():
            w.setText("—")

    def _update_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_frame = frame.copy()

        t0 = time.time()
        faces = detect_faces(frame)
        elapsed = int((time.time() - t0) * 1000)

        display1 = frame.copy()
        lm_display = frame.copy()  # qora fon emas, kamera tasviri

        if len(faces) > 0:
            face = faces[0]
            x, y, w, h = face

            vec = get_face_encoding(frame, face)
            result_person, confidence = compare_encodings_multi(self.known_encodings, vec)

            self._result_buffer.append((result_person, confidence))
            if len(self._result_buffer) > SMOOTHING_FRAMES:
                self._result_buffer.pop(0)

            smooth_person, smooth_conf = self._get_smoothed_result()

            color = (29, 158, 117) if smooth_person else (60, 60, 200)
            label = smooth_person['full_name'] if smooth_person else "Noma'lum"
            display1 = draw_face_box(display1, face, label,
                                     smooth_conf if smooth_person else None, color)

            landmarks = get_landmarks(frame, face)
            if landmarks and len(landmarks) >= 68:
                lm_display = draw_landmarks(lm_display, landmarks)
                self.metric_widgets['landmarks'].setText("68")
            else:
                cv2.putText(lm_display, "Landmark modeli yo'q",
                            (10, lm_display.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
                self.metric_widgets['landmarks'].setText("0")

            self._update_result(smooth_person, smooth_conf)
        else:
            self._result_buffer.clear()
            self.metric_widgets['landmarks'].setText("0")
            self._reset_result()

        self.cam_label.set_frame(display1)
        self.lm_label.set_frame(lm_display)

        self.metric_widgets['faces'].setText(str(len(faces)))
        self.metric_widgets['time'].setText(str(elapsed))

    def _get_smoothed_result(self):
        if not self._result_buffer:
            return None, 0.0
        found = [(p, c) for p, c in self._result_buffer if p is not None]
        if not found:
            return None, 0.0
        if len(found) >= (SMOOTHING_FRAMES // 2 + 1):
            names = [p['full_name'] for p, c in found]
            most_common = max(set(names), key=names.count)
            best = [(p, c) for p, c in found if p['full_name'] == most_common]
            avg_conf = sum(c for _, c in best) / len(best)
            return best[0][0], avg_conf
        return None, 0.0

    def _update_result(self, person, confidence):
        if person is None:
            self.name_label.setText("Noma'lum shaxs")
            self.name_label.setStyleSheet("color:#c0392b;")
            self.info_label.setText("Bazada topilmadi")
            self.avatar_label.setText("?")
            self.avatar_label.setStyleSheet(
                "background:#fdecea;border-radius:26px;color:#c0392b;"
            )
            self.conf_label.setText("—")
            self.conf_bar.setValue(0)
        else:
            self.name_label.setText(person['full_name'])
            self.name_label.setStyleSheet("color:#1a1a1a;")
            self.info_label.setText(
                f"{person.get('role','—')}  ·  {person.get('person_code','—')}"
            )
            initials = "".join(word[0].upper() for word in person['full_name'].split()[:2])
            self.avatar_label.setText(initials)
            self.avatar_label.setStyleSheet(
                "background:#E1F5EE;border-radius:26px;color:#085041;"
            )
            conf_int = int(confidence)
            self.conf_label.setText(f"{conf_int}%")
            color = "#1D9E75" if conf_int >= 75 else "#BA7517" if conf_int >= 50 else "#c0392b"
            self.conf_label.setStyleSheet(
                f"color:{color};font-size:26px;font-weight:bold;"
            )
            self.conf_bar.setValue(conf_int)
            self.conf_bar.setStyleSheet(f"""
                QProgressBar {{ border:1px solid #ddd;border-radius:4px;height:8px; }}
                QProgressBar::chunk {{ background:{color};border-radius:3px; }}
            """)

    def closeEvent(self, event):
        self._stop_camera()
        super().closeEvent(event)