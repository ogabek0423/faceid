import cv2
import os
import sys
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPushButton,
    QFrame, QProgressBar, QMessageBox, QSizePolicy, QSplitter
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from database.db_manager import add_person, add_face_image, save_encoding, get_image_count
from utils.face_processor import detect_faces, get_face_encoding, draw_face_box, save_face_image

ANGLES = ["To'g'ri", "Chap", "O'ng", "Yuqori", "Quyi"]
MIN_PHOTOS = 5
REQUIRED_PHOTOS = 10


class VideoLabel(QLabel):
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


class RegisterWidget(QWidget):
    person_saved = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.current_frame = None
        self.person_id = None
        self.photo_count = 0
        self.current_angle_idx = 0
        self._build_ui()

    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setContentsMargins(16, 16, 16, 16)
        main.setSpacing(16)

        # ---------- CHAP: kamera (kengayuvchan) ----------
        left = QVBoxLayout()
        left.setSpacing(8)

        cam_title = QLabel("Kamera")
        cam_title.setFont(QFont("Arial", 12, QFont.Bold))
        left.addWidget(cam_title)

        self.cam_label = VideoLabel()
        left.addWidget(self.cam_label, stretch=1)

        self.angle_label = QLabel(f"Keyingi burchak: {ANGLES[0]}")
        self.angle_label.setStyleSheet(
            "color:#1D9E75;font-size:13px;font-weight:bold;"
        )
        left.addWidget(self.angle_label)

        prog_row = QHBoxLayout()
        prog_lbl = QLabel("Rasmlar:")
        prog_lbl.setFixedWidth(65)
        prog_lbl.setStyleSheet("font-size:13px;")
        self.progress = QProgressBar()
        self.progress.setRange(0, REQUIRED_PHOTOS)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat(f"0 / {REQUIRED_PHOTOS}")
        self.progress.setStyleSheet("""
            QProgressBar { border:1px solid #ccc;border-radius:5px;height:20px;font-size:12px; }
            QProgressBar::chunk { background:#1D9E75;border-radius:4px; }
        """)
        prog_row.addWidget(prog_lbl)
        prog_row.addWidget(self.progress)
        left.addLayout(prog_row)

        btn_row = QHBoxLayout()
        self.btn_cam = QPushButton("▶  Kamerani yoq")
        self.btn_cam.setFixedHeight(38)
        self.btn_cam.setStyleSheet(
            "background:#1D9E75;color:white;border-radius:6px;font-size:13px;"
        )
        self.btn_cam.clicked.connect(self._toggle_camera)

        self.btn_snap = QPushButton("📷  Rasm ol")
        self.btn_snap.setFixedHeight(38)
        self.btn_snap.setEnabled(False)
        self.btn_snap.setStyleSheet(
            "background:#378ADD;color:white;border-radius:6px;font-size:13px;"
        )
        self.btn_snap.clicked.connect(self._take_photo)

        btn_row.addWidget(self.btn_cam)
        btn_row.addWidget(self.btn_snap)
        left.addLayout(btn_row)

        # ---------- O'NG: forma (kengayuvchan, lekin cheklangan) ----------
        right = QVBoxLayout()
        right.setSpacing(10)

        form_title = QLabel("Shaxs ma'lumotlari")
        form_title.setFont(QFont("Arial", 12, QFont.Bold))
        right.addWidget(form_title)

        form = QGridLayout()
        form.setSpacing(8)
        form.setColumnStretch(1, 1)

        lbl_style = "font-size:13px;color:#444;"

        name_lbl = QLabel("Ism:")
        name_lbl.setStyleSheet(lbl_style)
        form.addWidget(name_lbl, 0, 0)
        self.inp_name = QLineEdit()
        self.inp_name.setPlaceholderText("Akbar")
        self.inp_name.setMinimumHeight(32)
        self.inp_name.setStyleSheet("font-size:13px;padding:4px 8px;border:1px solid #ccc;border-radius:5px;")
        form.addWidget(self.inp_name, 0, 1)

        sur_lbl = QLabel("Familiya:")
        sur_lbl.setStyleSheet(lbl_style)
        form.addWidget(sur_lbl, 1, 0)
        self.inp_sur = QLineEdit()
        self.inp_sur.setPlaceholderText("Jurayev")
        self.inp_sur.setMinimumHeight(32)
        self.inp_sur.setStyleSheet("font-size:13px;padding:4px 8px;border:1px solid #ccc;border-radius:5px;")
        form.addWidget(self.inp_sur, 1, 1)

        role_lbl = QLabel("Toifa:")
        role_lbl.setStyleSheet(lbl_style)
        form.addWidget(role_lbl, 2, 0)
        self.inp_role = QComboBox()
        self.inp_role.addItems(["Talaba", "O'qituvchi", "Xodim", "Mehmon"])
        self.inp_role.setMinimumHeight(32)
        self.inp_role.setStyleSheet("font-size:13px;padding:4px 8px;border:1px solid #ccc;border-radius:5px;")
        form.addWidget(self.inp_role, 2, 1)

        right.addLayout(form)

        self.btn_start = QPushButton("✚  Yangi shaxs boshlash")
        self.btn_start.setFixedHeight(40)
        self.btn_start.setStyleSheet(
            "background:#534AB7;color:white;border-radius:6px;font-size:13px;"
        )
        self.btn_start.clicked.connect(self._start_registration)
        right.addWidget(self.btn_start)

        thumb_title = QLabel("Olingan rasmlar:")
        thumb_title.setStyleSheet("font-size:12px;color:#666;margin-top:4px;")
        right.addWidget(thumb_title)

        self.thumb_row = QHBoxLayout()
        self.thumb_row.setSpacing(6)
        self.thumbs = []
        for angle in ANGLES:
            col = QVBoxLayout()
            lbl = QLabel()
            lbl.setFixedSize(60, 60)
            lbl.setStyleSheet(
                "background:#f5f5f5;border:1px dashed #ccc;border-radius:6px;"
            )
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setText("—")
            angle_lbl = QLabel(angle)
            angle_lbl.setAlignment(Qt.AlignCenter)
            angle_lbl.setStyleSheet("font-size:10px;color:#888;")
            col.addWidget(lbl)
            col.addWidget(angle_lbl)
            self.thumb_row.addLayout(col)
            self.thumbs.append(lbl)
        self.thumb_row.addStretch()
        right.addLayout(self.thumb_row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size:12px;color:#666;")
        right.addWidget(self.status_label)

        self.btn_save = QPushButton("💾  Bazaga saqlash")
        self.btn_save.setFixedHeight(42)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet("""
            QPushButton:enabled {
                background:#1D9E75;color:white;border-radius:6px;
                font-size:14px;font-weight:bold;
            }
            QPushButton:disabled {
                background:#ddd;color:#999;border-radius:6px;font-size:14px;
            }
        """)
        self.btn_save.clicked.connect(self._save_person)
        right.addWidget(self.btn_save)

        self.btn_reset = QPushButton("Tozalash")
        self.btn_reset.setFixedHeight(34)
        self.btn_reset.setStyleSheet(
            "background:transparent;border:1px solid #ccc;"
            "border-radius:6px;color:#666;font-size:13px;"
        )
        self.btn_reset.clicked.connect(self.reset)
        right.addWidget(self.btn_reset)
        right.addStretch()

        # Proporsiya: kamera 60%, forma 40%
        left_frame = QFrame()
        left_frame.setLayout(left)
        left_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        right_frame = QFrame()
        right_frame.setLayout(right)
        right_frame.setMinimumWidth(260)
        right_frame.setMaximumWidth(400)
        right_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        main.addWidget(left_frame, stretch=3)
        main.addWidget(right_frame, stretch=2)

    def _toggle_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "Xato", "Kamera topilmadi!")
                return
            self.timer.start(30)
            self.btn_cam.setText("⏹  Kamerani o'chir")
            self.btn_cam.setStyleSheet(
                "background:#c0392b;color:white;border-radius:6px;font-size:13px;"
            )
            self.btn_snap.setEnabled(self.person_id is not None)
        else:
            self._stop_camera()

    def _stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.cam_label.clear_frame()
        self.btn_cam.setText("▶  Kamerani yoq")
        self.btn_cam.setStyleSheet(
            "background:#1D9E75;color:white;border-radius:6px;font-size:13px;"
        )
        self.btn_snap.setEnabled(False)

    def _update_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_frame = frame.copy()
        faces = detect_faces(frame)
        display = frame.copy()
        if len(faces) > 0:
            display = draw_face_box(display, faces[0])
        self.cam_label.set_frame(display)

    def _start_registration(self):
        name = self.inp_name.text().strip()
        sur  = self.inp_sur.text().strip()
        if not name or not sur:
            QMessageBox.warning(self, "Xato", "Ism va familiyani kiriting!")
            return
        full_name = f"{name} {sur}"
        role = self.inp_role.currentText()
        self.person_id, code = add_person(full_name, role)
        self.status_label.setText(
            f"✓ {full_name} ({code}) ro'yxatga olindi.\n"
            f"Endi {REQUIRED_PHOTOS} ta rasm oling."
        )
        self.status_label.setStyleSheet("font-size:12px;color:#1D9E75;")
        self.btn_start.setEnabled(False)
        self.inp_name.setEnabled(False)
        self.inp_sur.setEnabled(False)
        self.inp_role.setEnabled(False)
        if self.cap and self.cap.isOpened():
            self.btn_snap.setEnabled(True)

    def _take_photo(self):
        if self.current_frame is None or self.person_id is None:
            return
        faces = detect_faces(self.current_frame)
        if len(faces) == 0:
            self.status_label.setText("⚠  Yuz topilmadi. Kameraga yaqinroq turing.")
            self.status_label.setStyleSheet("font-size:12px;color:#e67e22;")
            return
        angle = ANGLES[self.current_angle_idx % len(ANGLES)]
        path  = save_face_image(self.current_frame, self.person_id, angle, BASE_DIR)
        add_face_image(self.person_id, path, angle)

        idx = self.current_angle_idx % len(ANGLES)
        x, y, w, h = faces[0]
        thumb_img = self.current_frame[y:y+h, x:x+w]
        if thumb_img.size > 0:
            thumb_img = cv2.resize(thumb_img, (60, 60))
            rgb  = cv2.cvtColor(thumb_img, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, 60, 60, 3*60, QImage.Format_RGB888)
            self.thumbs[idx].setPixmap(QPixmap.fromImage(qimg))
            self.thumbs[idx].setStyleSheet("border:2px solid #1D9E75;border-radius:6px;")

        self.photo_count += 1
        self.current_angle_idx += 1
        self.progress.setValue(min(self.photo_count, REQUIRED_PHOTOS))
        self.progress.setFormat(f"{self.photo_count} / {REQUIRED_PHOTOS}")

        if self.current_angle_idx < len(ANGLES):
            self.angle_label.setText(f"Keyingi burchak: {ANGLES[self.current_angle_idx]}")
        else:
            self.angle_label.setText("Istalgan burchakdan davom eting")

        self.status_label.setText(
            f"✓ {self.photo_count} ta rasm olindi. "
            f"{'Saqlash mumkin!' if self.photo_count >= MIN_PHOTOS else str(MIN_PHOTOS - self.photo_count) + ' ta yana kerak.'}"
        )
        if self.photo_count >= MIN_PHOTOS:
            self.btn_save.setEnabled(True)
            self.status_label.setStyleSheet("font-size:12px;color:#1D9E75;font-weight:bold;")

    def _save_person(self):
        self.status_label.setText("⏳ Encoding hisoblanmoqda...")
        self.btn_save.setEnabled(False)

        from database.db_manager import get_connection
        conn = get_connection()
        rows = conn.execute(
            "SELECT image_path FROM face_images WHERE person_id=?",
            (self.person_id,)
        ).fetchall()
        conn.close()

        vectors = []
        for row in rows:
            img = cv2.imread(row['image_path'])
            if img is None:
                continue
            faces = detect_faces(img)
            if faces is not None and len(faces) > 0:
                try:
                    vec = get_face_encoding(img, faces[0])
                    if vec is not None:
                        vectors.append(vec)
                except Exception as e:
                    print(f"Encoding xatosi: {e}")

        if len(vectors) > 0:
            save_encoding(self.person_id, vectors)
            self.status_label.setText(
                f"✓ Muvaffaqiyatli saqlandi! {len(vectors)} ta rasmdan encoding olindi."
            )
            self.status_label.setStyleSheet("font-size:12px;color:#1D9E75;font-weight:bold;")
            self._stop_camera()
            self.person_saved.emit()
        else:
            self.status_label.setText("⚠ Xato: Rasmlardan yuz xususiyatlarini ajratib bo'lmadi.")
            self.status_label.setStyleSheet("font-size:12px;color:#c0392b;")
            self.btn_save.setEnabled(True)

    def reset(self):
        self._stop_camera()
        self.person_id = None
        self.photo_count = 0
        self.current_angle_idx = 0
        self.inp_name.setText("")
        self.inp_sur.setText("")
        self.inp_name.setEnabled(True)
        self.inp_sur.setEnabled(True)
        self.inp_role.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.progress.setValue(0)
        self.progress.setFormat(f"0 / {REQUIRED_PHOTOS}")
        self.angle_label.setText(f"Keyingi burchak: {ANGLES[0]}")
        self.status_label.setText("")
        for thumb in self.thumbs:
            thumb.clear()
            thumb.setText("—")
            thumb.setStyleSheet(
                "background:#f5f5f5;border:1px dashed #ccc;border-radius:6px;"
            )

    def closeEvent(self, event):
        self._stop_camera()
        super().closeEvent(event)