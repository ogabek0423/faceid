import os
import sys
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFrame, QCheckBox, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_env():
    """
    .env fayldan ADMIN_USERNAME va ADMIN_PASSWORD ni o'qiydi.
    python-dotenv o'rnatilmagan bo'lsa o'zimiz parse qilamiz.
    """
    env_path = os.path.join(BASE_DIR, '.env')
    creds = {'ADMIN_USERNAME': 'admin', 'ADMIN_PASSWORD': 'Admin@1234'}

    if not os.path.exists(env_path):
        # .env yo'q — standart qiymatlar bilan yaratib qo'yamiz
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write("# FaceID tizimi - Admin ma'lumotlari\n")
            f.write("# Bu faylni hech qachon GitHub ga yuklamang!\n")
            f.write(f"ADMIN_USERNAME={creds['ADMIN_USERNAME']}\n")
            f.write(f"ADMIN_PASSWORD={creds['ADMIN_PASSWORD']}\n")
        return creds

    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, val = line.partition('=')
            key = key.strip()
            val = val.strip()
            if key in creds:
                creds[key] = val

    return creds


class LoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FaceID — Kirish")
        self.setFixedSize(400, 480)
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setStyleSheet("background:#f4f6f8;font-family:Arial;")

        self._creds = _load_env()
        self._attempts = 0
        self._locked = False

        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- Yuqori sarlavha ---
        header = QFrame()
        header.setFixedHeight(140)
        header.setStyleSheet("background:#1a1a2e;border-radius:0;")
        h_layout = QVBoxLayout(header)
        h_layout.setAlignment(Qt.AlignCenter)
        h_layout.setSpacing(6)

        icon_lbl = QLabel("👁")
        icon_lbl.setFont(QFont("Arial", 36))
        icon_lbl.setAlignment(Qt.AlignCenter)

        title_lbl = QLabel("FaceID")
        title_lbl.setFont(QFont("Arial", 20, QFont.Bold))
        title_lbl.setAlignment(Qt.AlignCenter)
        title_lbl.setStyleSheet("color:white;")

        sub_lbl = QLabel("Admin paneliga kirish")
        sub_lbl.setAlignment(Qt.AlignCenter)
        sub_lbl.setStyleSheet("color:#7f8c8d;font-size:12px;")

        h_layout.addWidget(icon_lbl)
        h_layout.addWidget(title_lbl)
        h_layout.addWidget(sub_lbl)
        root.addWidget(header)

        # --- Forma ---
        form_frame = QFrame()
        form_frame.setStyleSheet(
            "background:white;border-radius:0;"
            "border-top:3px solid #1D9E75;"
        )
        form_layout = QVBoxLayout(form_frame)
        form_layout.setContentsMargins(36, 32, 36, 32)
        form_layout.setSpacing(16)

        # Login
        user_lbl = QLabel("Foydalanuvchi nomi")
        user_lbl.setStyleSheet("font-size:12px;color:#666;font-weight:bold;")
        form_layout.addWidget(user_lbl)

        self.inp_user = QLineEdit()
        self.inp_user.setPlaceholderText("admin")
        self.inp_user.setMinimumHeight(42)
        self.inp_user.setStyleSheet(self._input_style())
        self.inp_user.returnPressed.connect(self._try_login)
        form_layout.addWidget(self.inp_user)

        # Parol
        pass_lbl = QLabel("Parol")
        pass_lbl.setStyleSheet("font-size:12px;color:#666;font-weight:bold;")
        form_layout.addWidget(pass_lbl)

        pass_row = QHBoxLayout()
        self.inp_pass = QLineEdit()
        self.inp_pass.setPlaceholderText("••••••••")
        self.inp_pass.setEchoMode(QLineEdit.Password)
        self.inp_pass.setMinimumHeight(42)
        self.inp_pass.setStyleSheet(self._input_style())
        self.inp_pass.returnPressed.connect(self._try_login)
        pass_row.addWidget(self.inp_pass)

        self.btn_eye = QPushButton("👁")
        self.btn_eye.setFixedSize(42, 42)
        self.btn_eye.setCheckable(True)
        self.btn_eye.setStyleSheet(
            "border:1.5px solid #ddd;border-radius:8px;"
            "background:white;font-size:16px;"
        )
        self.btn_eye.toggled.connect(self._toggle_pass_visibility)
        pass_row.addWidget(self.btn_eye)
        form_layout.addLayout(pass_row)

        # Xato xabari
        self.err_label = QLabel("")
        self.err_label.setAlignment(Qt.AlignCenter)
        self.err_label.setStyleSheet(
            "color:#c0392b;font-size:12px;"
            "background:#fdecea;border-radius:6px;padding:6px;"
        )
        self.err_label.hide()
        form_layout.addWidget(self.err_label)

        # Kirish tugmasi
        self.btn_login = QPushButton("Kirish")
        self.btn_login.setFixedHeight(46)
        self.btn_login.setStyleSheet("""
            QPushButton {
                background:#1D9E75;color:white;
                border-radius:8px;font-size:15px;font-weight:bold;
            }
            QPushButton:hover { background:#17835f; }
            QPushButton:disabled { background:#ccc;color:#999; }
        """)
        self.btn_login.clicked.connect(self._try_login)
        form_layout.addWidget(self.btn_login)

        # Izoh
        hint = QLabel(f"")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color:#aaa;font-size:11px;")
        form_layout.addWidget(hint)

        root.addWidget(form_frame)

        # --- Pastki ---
        footer = QFrame()
        footer.setStyleSheet("background:#f4f6f8;")
        f_layout = QVBoxLayout(footer)
        f_layout.setContentsMargins(20, 12, 20, 12)

        env_hint = QLabel("")
        env_hint.setAlignment(Qt.AlignCenter)
        env_hint.setStyleSheet("color:#999;font-size:11px;")
        f_layout.addWidget(env_hint)
        root.addWidget(footer)

    def _input_style(self):
        return (
            "border:1.5px solid #ddd;border-radius:8px;"
            "padding:0 12px;font-size:13px;color:#333;"
            "background:white;"
        )

    def _toggle_pass_visibility(self, checked):
        if checked:
            self.inp_pass.setEchoMode(QLineEdit.Normal)
        else:
            self.inp_pass.setEchoMode(QLineEdit.Password)

    def _try_login(self):
        if self._locked:
            return

        username = self.inp_user.text().strip()
        password = self.inp_pass.text()

        ok_user = username == self._creds['ADMIN_USERNAME']
        ok_pass = password == self._creds['ADMIN_PASSWORD']

        if ok_user and ok_pass:
            self.accept()
        else:
            self._attempts += 1
            remaining = 5 - self._attempts

            if self._attempts >= 5:
                self._locked = True
                self.btn_login.setEnabled(False)
                self.inp_user.setEnabled(False)
                self.inp_pass.setEnabled(False)
                self._show_error("❌ 5 ta urinish tugadi. Dasturni qayta ishga tushiring.")
                return

            if not ok_user:
                msg = f"Foydalanuvchi nomi noto'g'ri. ({remaining} urinish qoldi)"
            else:
                msg = f"Parol noto'g'ri. ({remaining} urinish qoldi)"

            self._show_error(msg)
            self.inp_pass.clear()
            self.inp_pass.setFocus()

            # Qizil border
            self.inp_pass.setStyleSheet(
                self._input_style().replace("#ddd", "#c0392b")
            )
            if not ok_user:
                self.inp_user.setStyleSheet(
                    self._input_style().replace("#ddd", "#c0392b")
                )
            QTimer.singleShot(1500, self._reset_input_style)

    def _show_error(self, msg):
        self.err_label.setText(msg)
        self.err_label.show()

    def _reset_input_style(self):
        self.inp_pass.setStyleSheet(self._input_style())
        self.inp_user.setStyleSheet(self._input_style())
