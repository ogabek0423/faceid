import cv2
import os
import sys
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPushButton,
    QFrame, QProgressBar, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from database.db_manager import add_person, add_face_image, save_encoding, get_image_count
from utils.face_processor import detect_faces, get_face_encoding, draw_face_box, save_face_image

ANGLES = ["To'g'ri", "Chap", "O'ng", "Yuqori", "Quyi"]
MIN_PHOTOS = 5
REQUIRED_PHOTOS = 10  # tavsiya etilgan


class RegisterWidget(QWidget):
    person_saved = pyqtSignal()  # asosiy oynaga signal

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

        # ---------- CHAP: kamera ----------
        left = QVBoxLayout()
        left.setSpacing(10)

        cam_title = QLabel("Kamera")
        cam_title.setFont(QFont("Arial", 11, QFont.Bold))
        left.addWidget(cam_title)

        self.cam_label = QLabel()
        self.cam_label.setFixedSize(480, 360)
        self.cam_label.setStyleSheet(
            "background:#111; border-radius:8px; border:1px solid #333;"
        )
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setText("Kamera o'chirilgan")
        left.addWidget(self.cam_label)

        # Holat satri
        self.angle_label = QLabel(f"Keyingi burchak: {ANGLES[0]}")
        self.angle_label.setStyleSheet(
            "color:#1D9E75; font-size:13px; font-weight:bold;"
        )
        left.addWidget(self.angle_label)

        # Progress bar
        prog_row = QHBoxLayout()
        prog_label = QLabel("Rasmlar:")
        prog_label.setFixedWidth(60)
        self.progress = QProgressBar()
        self.progress.setRange(0, REQUIRED_PHOTOS)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat(f"0 / {REQUIRED_PHOTOS}")
        self.progress.setStyleSheet("""
            QProgressBar { border:1px solid #ccc; border-radius:5px; height:18px; }
            QProgressBar::chunk { background:#1D9E75; border-radius:4px; }
        """)
        prog_row.addWidget(prog_label)
        prog_row.addWidget(self.progress)
        left.addLayout(prog_row)

        # Tugmalar
        btn_row = QHBoxLayout()
        self.btn_cam = QPushButton("▶  Kamerani yoq")
        self.btn_cam.setFixedHeight(36)
        self.btn_cam.setStyleSheet(
            "background:#1D9E75;color:white;border-radius:6px;font-size:13px;"
        )
        self.btn_cam.clicked.connect(self._toggle_camera)

        self.btn_snap = QPushButton("📷  Rasm ol")
        self.btn_snap.setFixedHeight(36)
        self.btn_snap.setEnabled(False)
        self.btn_snap.setStyleSheet(
            "background:#378ADD;color:white;border-radius:6px;font-size:13px;"
        )
        self.btn_snap.clicked.connect(self._take_photo)

        btn_row.addWidget(self.btn_cam)
        btn_row.addWidget(self.btn_snap)
        left.addLayout(btn_row)
        left.addStretch()
        main.addLayout(left)

        # ---------- O'NG: forma + natija ----------
        right = QVBoxLayout()
        right.setSpacing(10)

        form_title = QLabel("Shaxs ma'lumotlari")
        form_title.setFont(QFont("Arial", 11, QFont.Bold))
        right.addWidget(form_title)

        # Forma
        form = QGridLayout()
        form.setSpacing(8)

        form.addWidget(QLabel("Ism:"), 0, 0)
        self.inp_name = QLineEdit()
        self.inp_name.setPlaceholderText("Akbar")
        form.addWidget(self.inp_name, 0, 1)

        form.addWidget(QLabel("Familiya:"), 1, 0)
        self.inp_sur = QLineEdit()
        self.inp_sur.setPlaceholderText("Jurayev")
        form.addWidget(self.inp_sur, 1, 1)

        form.addWidget(QLabel("Toifa:"), 2, 0)
        self.inp_role = QComboBox()
        self.inp_role.addItems(["Talaba", "O'qituvchi", "Xodim", "Mehmon"])
        form.addWidget(self.inp_role, 2, 1)

        right.addLayout(form)

        # Boshlash tugmasi
        self.btn_start = QPushButton("✚  Yangi shaxs boshlash")
        self.btn_start.setFixedHeight(38)
        self.btn_start.setStyleSheet(
            "background:#534AB7;color:white;border-radius:6px;font-size:13px;"
        )
        self.btn_start.clicked.connect(self._start_registration)
        right.addWidget(self.btn_start)

        # Olingan rasmlar thumbnail
        thumb_title = QLabel("Olingan rasmlar:")
        thumb_title.setStyleSheet("font-size:12px;color:#666;margin-top:8px;")
        right.addWidget(thumb_title)

        self.thumb_row = QHBoxLayout()
        self.thumb_row.setSpacing(6)
        self.thumbs = []
        for i, angle in enumerate(ANGLES):
            col = QVBoxLayout()
            lbl = QLabel()
            lbl.setFixedSize(64, 64)
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

        right.addLayout(self.thumb_row)

        # Status xabar
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size:12px;color:#666;")
        right.addWidget(self.status_label)

        # Saqlash tugmasi
        self.btn_save = QPushButton("💾  Bazaga saqlash")
        self.btn_save.setFixedHeight(40)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet("""
            QPushButton:enabled {
                background:#1D9E75;color:white;border-radius:6px;font-size:14px;font-weight:bold;
            }
            QPushButton:disabled {
                background:#ccc;color:#999;border-radius:6px;font-size:14px;
            }
        """)
        self.btn_save.clicked.connect(self._save_person)
        right.addWidget(self.btn_save)

        self.btn_reset = QPushButton("Tozalash")
        self.btn_reset.setFixedHeight(32)
        self.btn_reset.setStyleSheet(
            "background:transparent;border:1px solid #ccc;border-radius:6px;color:#666;"
        )
        self.btn_reset.clicked.connect(self.reset)
        right.addWidget(self.btn_reset)
        right.addStretch()

        main.addLayout(right)

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
        self.cam_label.setText("Kamera o'chirilgan")
        self.cam_label.setStyleSheet(
            "background:#111;border-radius:8px;border:1px solid #333;"
            "color:white;font-size:14px;"
        )
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

        # Yuzni aniqlash va to'rtburchak chizish
        faces = detect_faces(frame)
        display = frame.copy()
        if len(faces) > 0:
            x, y, w, h = faces[0]
            display = draw_face_box(display, faces[0])

        # Qt ga convert
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            480, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.cam_label.setPixmap(pix)

    def _start_registration(self):
        name = self.inp_name.text().strip()
        sur = self.inp_sur.text().strip()
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
        path = save_face_image(self.current_frame, self.person_id, angle, BASE_DIR)
        add_face_image(self.person_id, path, angle)

        # Thumbnail
        idx = self.current_angle_idx % len(ANGLES)
        thumb_img = self.current_frame[
            faces[0][1]:faces[0][1]+faces[0][3],
            faces[0][0]:faces[0][0]+faces[0][2]
        ]
        if thumb_img.size > 0:
            thumb_img = cv2.resize(thumb_img, (64, 64))
            rgb = cv2.cvtColor(thumb_img, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, 64, 64, 3*64, QImage.Format_RGB888)
            self.thumbs[idx].setPixmap(QPixmap.fromImage(qimg))
            self.thumbs[idx].setStyleSheet(
                "border:2px solid #1D9E75;border-radius:6px;"
            )

        self.photo_count += 1
        self.current_angle_idx += 1
        self.progress.setValue(min(self.photo_count, REQUIRED_PHOTOS))
        self.progress.setFormat(f"{self.photo_count} / {REQUIRED_PHOTOS}")

        if self.current_angle_idx < len(ANGLES):
            next_angle = ANGLES[self.current_angle_idx]
            self.angle_label.setText(f"Keyingi burchak: {next_angle}")
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
        """Encodinglarni hisoblash va saqlash"""
        self.status_label.setText("⏳ Encoding hisoblanmoqda...")
        self.btn_save.setEnabled(False)

        # Barcha rasmlardan encoding hisoblash
        from database.db_manager import get_connection
        conn = get_connection()
        rows = conn.execute(
            "SELECT image_path FROM face_images WHERE person_id=?",
            (self.person_id,)
        ).fetchall()
        conn.close()

        vectors = []
        print(f"Bazada jami {len(rows)} ta rasm yo'li topildi. Qayta ishlanmoqda...")

        for row in rows:
            img = cv2.imread(row['image_path'])
            if img is None:
                print(f"Xatolik: Rasmni o'qib bo'lmadi -> {row['image_path']}")
                continue

            # Yuzni qayta aniqlaymiz
            faces = detect_faces(img)

            # XAVFSIZ TEKSHIRUV: Ro'yxat yoki massiv bo'sh emasligini aniqlash
            if faces is not None and len(faces) > 0:
                # detect_faces odatda [[x, y, w, h]] formatida qaytaradi, shuning uchun birinchisini olamiz
                face_box = faces[0]

                try:
                    # Yuzning raqamli modelini (128 o'lchamli vektor) olamiz
                    vec = get_face_encoding(img, face_box)

                    if vec is not None:
                        vectors.append(vec)
                        print(f"✓ {row['image_path']} dan encoding olindi.")
                    else:
                        print(f"⚠ {row['image_path']} dan encoding olib bo'lmadi (get_face_encoding None qaytardi).")
                except Exception as e:
                    print(f"❌ Xususiyat ajratishda ichki xato: {e}")
            else:
                print(f"⚠ {row['image_path']} rasm ichidan yuz qayta topilmadi!")

        # Agar kamida bitta rasmdan encoding ololsak, o'rtachasini hisoblaymiz
        if len(vectors) > 0:
            # Barcha vektorlarni saqlaymiz (o'rtacha emas)
            save_encoding(self.person_id, vectors)

            self.status_label.setText(
                f"✓ Muvaffaqiyatli saqlandi! {len(vectors)} ta rasmdan encoding olindi."
            )
            self.status_label.setStyleSheet("font-size:12px;color:#1D9E75;font-weight:bold;")
            self._stop_camera()
            self.person_saved.emit()
        else:
            self.status_label.setText("⚠ Xato: Olingan rasmlardan yuz xususiyatlarini (encoding) ajratib bo'lmadi.")
            self.status_label.setStyleSheet("font-size:12px;color:#c0392b;")
            self.btn_save.setEnabled(True)

    def reset(self):
        """Formani tozalash"""
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