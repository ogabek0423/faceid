import cv2
import numpy as np
import sys
import os
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QFrame, QProgressBar, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from database.db_manager import load_all_encodings
from utils.face_processor import (
    detect_faces, get_landmarks, get_face_encoding,
    draw_landmarks, draw_face_box, compare_encodings
)


class RecognizeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.known_encodings = []
        self.current_frame = None
        self.last_result = None
        self._build_ui()
        self._reload_encodings()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # Yuqori: boshqaruv
        top = QHBoxLayout()
        title = QLabel("Tanib olish rejimi")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        top.addWidget(title)
        top.addStretch()

        self.btn_reload = QPushButton("🔄  Bazani yangilash")
        self.btn_reload.setFixedHeight(32)
        self.btn_reload.setStyleSheet(
            "border:1px solid #ccc;border-radius:5px;padding:0 12px;font-size:12px;"
        )
        self.btn_reload.clicked.connect(self._reload_encodings)
        top.addWidget(self.btn_reload)

        self.btn_cam = QPushButton("▶  Kamerani yoq")
        self.btn_cam.setFixedHeight(32)
        self.btn_cam.setStyleSheet(
            "background:#1D9E75;color:white;border-radius:5px;padding:0 14px;font-size:12px;"
        )
        self.btn_cam.clicked.connect(self._toggle_camera)
        top.addWidget(self.btn_cam)

        root.addLayout(top)

        # Asosiy 3 panel
        panels = QHBoxLayout()
        panels.setSpacing(12)

        # Panel 1 — original kamera
        self.cam_label = self._make_panel("Kamera oqimi", 400, 300)
        panels.addWidget(self._wrap(self.cam_label, "Kamera oqimi"))

        # Panel 2 — landmark ko'rinish
        self.lm_label = self._make_panel("Landmark nuqtalar", 400, 300)
        panels.addWidget(self._wrap(self.lm_label, "Landmark nuqtalar (68 ta)"))

        root.addLayout(panels)

        # Pastki: natija
        result_frame = QFrame()
        result_frame.setStyleSheet(
            "background:white;border:1px solid #e0e0e0;border-radius:10px;"
        )
        result_layout = QHBoxLayout(result_frame)
        result_layout.setContentsMargins(20, 16, 20, 16)
        result_layout.setSpacing(24)

        # Avatar
        self.avatar_label = QLabel("?")
        self.avatar_label.setFixedSize(60, 60)
        self.avatar_label.setAlignment(Qt.AlignCenter)
        self.avatar_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.avatar_label.setStyleSheet(
            "background:#eee;border-radius:30px;color:#999;"
        )
        result_layout.addWidget(self.avatar_label)

        # Ism va ma'lumot
        info_col = QVBoxLayout()
        self.name_label = QLabel("Kutilmoqda...")
        self.name_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.name_label.setStyleSheet("color:#222;")

        self.info_label = QLabel("Kamerani yoqing va yuzingizni ko'rsating")
        self.info_label.setStyleSheet("color:#888;font-size:13px;")

        info_col.addWidget(self.name_label)
        info_col.addWidget(self.info_label)
        result_layout.addLayout(info_col)
        result_layout.addStretch()

        # Moslik
        conf_col = QVBoxLayout()
        conf_col.setAlignment(Qt.AlignRight)
        self.conf_label = QLabel("—")
        self.conf_label.setFont(QFont("Arial", 28, QFont.Bold))
        self.conf_label.setAlignment(Qt.AlignRight)
        self.conf_label.setStyleSheet("color:#1D9E75;")

        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setValue(0)
        self.conf_bar.setFixedWidth(160)
        self.conf_bar.setTextVisible(False)
        self.conf_bar.setStyleSheet("""
            QProgressBar { border:1px solid #ddd; border-radius:4px; height:8px; }
            QProgressBar::chunk { background:#1D9E75; border-radius:3px; }
        """)

        conf_col.addWidget(self.conf_label)
        conf_col.addWidget(self.conf_bar)
        result_layout.addLayout(conf_col)

        # Metrikalar
        metrics = QHBoxLayout()
        metrics.setSpacing(16)
        self.metric_widgets = {}
        for key, label in [("faces", "Yuzlar"), ("landmarks", "Nuqtalar"), ("time", "Vaqt (ms)")]:
            m = self._make_metric(label, "—")
            self.metric_widgets[key] = m[1]
            metrics.addLayout(m[0])

        result_col = QVBoxLayout()
        result_col.addWidget(result_frame)
        result_col.addLayout(metrics)

        root.addLayout(result_col)

    def _make_panel(self, title, w, h):
        lbl = QLabel("Kamera o'chirilgan")
        lbl.setFixedSize(w, h)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(
            "background:#111;color:#555;font-size:13px;"
            "border-radius:8px;border:1px solid #333;"
        )
        return lbl

    def _wrap(self, widget, title):
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        t = QLabel(title)
        t.setStyleSheet("font-size:12px;color:#888;font-weight:bold;")
        layout.addWidget(t)
        layout.addWidget(widget)
        return frame

    def _make_metric(self, label, value):
        layout = QVBoxLayout()
        layout.setSpacing(2)
        lbl = QLabel(label)
        lbl.setStyleSheet(
            "font-size:11px;color:#999;background:#f8f8f8;"
            "border-radius:5px;padding:2px 8px;"
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
        self.btn_reload.setText(f"🔄  Bazani yangilash ({len(self.known_encodings)} ta)")

    def _toggle_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Xato", "Kamera topilmadi!")
                return
            self.timer.start(30)
            self.btn_cam.setText("⏹  To'xtatish")
            self.btn_cam.setStyleSheet(
                "background:#c0392b;color:white;border-radius:5px;padding:0 14px;font-size:12px;"
            )
        else:
            self._stop_camera()

    def _stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        for lbl in [self.cam_label, self.lm_label]:
            lbl.clear()
            lbl.setText("Kamera o'chirilgan")
        self.btn_cam.setText("▶  Kamerani yoq")
        self.btn_cam.setStyleSheet(
            "background:#1D9E75;color:white;border-radius:5px;padding:0 14px;font-size:12px;"
        )
        self._reset_result()

    def _reset_result(self):
        self.name_label.setText("Kutilmoqda...")
        self.info_label.setText("Kamerani yoqing va yuzingizni ko'rsating")
        self.avatar_label.setText("?")
        self.avatar_label.setStyleSheet("background:#eee;border-radius:30px;color:#999;")
        self.conf_label.setText("—")
        self.conf_bar.setValue(0)
        for w in self.metric_widgets.values():
            w.setText("—")

    def _update_frame(self):
        if self.cap is None:
            return
        import time
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_frame = frame.copy()

        t0 = time.time()
        faces = detect_faces(frame)
        elapsed = int((time.time() - t0) * 1000)

        # Panel 1: kamera + yuz ramkasi
        display1 = frame.copy()

        if len(faces) > 0:
            face = faces[0]
            # Encoding va solishtirish
            vec = get_face_encoding(frame, face)

            # Diqqat: compare_encodings funksiyasiga ma'lumot yuborilmoqda
            result_person, confidence = compare_encodings(self.known_encodings, vec)

            # Agar shaxs aniqlansa yashil, aks holda qizil ramka
            color = (29, 158, 117) if result_person else (60, 60, 200)  # BGR formatda
            label = result_person['full_name'] if result_person else "Noma'lum"
            display1 = draw_face_box(display1, face, label, confidence, color)

            # Panel 2: landmark
            landmarks = get_landmarks(frame, face)
            lm_frame = np.zeros_like(frame)
            if landmarks:
                lm_frame = draw_landmarks(lm_frame, landmarks)
            self._show_frame(lm_frame, self.lm_label)
            self.metric_widgets['landmarks'].setText("68")

            # FAQAT YUZ BOR BO'LSAGINA NATIJANI YANGILASH
            self._update_result(result_person, confidence)
        else:
            self.lm_label.setText("Yuz topilmadi")
            self.metric_widgets['landmarks'].setText("0")
            # Kadrda yuz bo'lmasa, oynani boshlang'ich holatga qaytarish
            self._reset_result()

        self._show_frame(display1, self.cam_label)

        # Metrikalar
        self.metric_widgets['faces'].setText(str(len(faces)))
        self.metric_widgets['time'].setText(f"{elapsed}")

    def _show_frame(self, frame, label):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            label.width(), label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(pix)

    def _update_result(self, person, confidence):
        if person is None:
            self.name_label.setText("Noma'lum shaxs")
            self.name_label.setStyleSheet("color:#c0392b;")
            self.info_label.setText("Bazada topilmadi")
            self.avatar_label.setText("?")
            self.avatar_label.setStyleSheet(
                "background:#fdecea;border-radius:30px;color:#c0392b;"
            )
            self.conf_label.setText("—")
            self.conf_bar.setValue(0)
            self.conf_bar.setStyleSheet("""
                QProgressBar { border:1px solid #ddd; border-radius:4px; height:8px; }
                QProgressBar::chunk { background:#c0392b; border-radius:3px; }
            """)
        else:
            self.name_label.setText(person['full_name'])
            self.name_label.setStyleSheet("color:#1a1a1a;")
            self.info_label.setText(
                f"{person['role']}  ·  {person['person_code']}"
            )
            initials = "".join(w[0].upper() for w in person['full_name'].split()[:2])
            self.avatar_label.setText(initials)
            self.avatar_label.setStyleSheet(
                "background:#E1F5EE;border-radius:30px;color:#085041;"
            )
            conf_int = int(confidence)
            self.conf_label.setText(f"{conf_int}%")
            color = "#1D9E75" if conf_int >= 80 else "#BA7517" if conf_int >= 60 else "#c0392b"
            self.conf_label.setStyleSheet(f"color:{color};")
            self.conf_bar.setValue(conf_int)
            self.conf_bar.setStyleSheet(f"""
                QProgressBar {{ border:1px solid #ddd; border-radius:4px; height:8px; }}
                QProgressBar::chunk {{ background:{color}; border-radius:3px; }}
            """)

    def closeEvent(self, event):
        self._stop_camera()
        super().closeEvent(event)