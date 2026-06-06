import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QStackedWidget, QFrame, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from database.db_manager import init_db
from ui.login_window import LoginDialog
from ui.register_window import RegisterWidget
from ui.recognize_window import RecognizeWidget
from ui.database_window import DatabaseWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Yuz Tanib Olish Tizimi")
        self.setMinimumSize(1100, 700)
        self.setStyleSheet("background:#f9f9f9;font-family:Arial;")

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- CHAP: Navigatsiya panel ----
        nav = QFrame()
        nav.setFixedWidth(200)
        nav.setStyleSheet("background:#1a1a2e;border-right:1px solid #0f0f23;")
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(0)

        # Logo
        logo_frame = QFrame()
        logo_frame.setStyleSheet("background:#16213e;padding:20px 0;")
        logo_layout = QVBoxLayout(logo_frame)
        logo_layout.setContentsMargins(16, 20, 16, 20)

        logo_icon = QLabel("👁")
        logo_icon.setFont(QFont("Arial", 28))
        logo_icon.setAlignment(Qt.AlignCenter)

        logo_text = QLabel("FaceID")
        logo_text.setFont(QFont("Arial", 16, QFont.Bold))
        logo_text.setAlignment(Qt.AlignCenter)
        logo_text.setStyleSheet("color:white;")

        logo_sub = QLabel("Yuz tanib olish")
        logo_sub.setAlignment(Qt.AlignCenter)
        logo_sub.setStyleSheet("color:#7f8c8d;font-size:11px;")

        logo_layout.addWidget(logo_icon)
        logo_layout.addWidget(logo_text)
        logo_layout.addWidget(logo_sub)
        nav_layout.addWidget(logo_frame)

        # Navigatsiya tugmalari
        self.nav_buttons = []

        # --- Tanib olish: hamma ko'ra oladi ---
        nav_layout.addSpacing(16)
        btn_rec = QPushButton("🔍  Tanib olish")
        btn_rec.setCheckable(True)
        btn_rec.setFixedHeight(44)
        btn_rec.setStyleSheet(self._nav_btn_style())
        btn_rec.clicked.connect(lambda: self._switch_tab(0))
        nav_layout.addWidget(btn_rec)
        self.nav_buttons.append(btn_rec)

        # --- Admin tugmalari: qulf belgisi bilan ---
        btn_reg = QPushButton("🔒  Ro'yxatga olish")
        btn_reg.setCheckable(True)
        btn_reg.setFixedHeight(44)
        btn_reg.setStyleSheet(self._nav_btn_style())
        btn_reg.clicked.connect(lambda: self._admin_switch(1))
        nav_layout.addWidget(btn_reg)
        self.nav_buttons.append(btn_reg)

        btn_db = QPushButton("🔒  Ma'lumotlar bazasi")
        btn_db.setCheckable(True)
        btn_db.setFixedHeight(44)
        btn_db.setStyleSheet(self._nav_btn_style())
        btn_db.clicked.connect(lambda: self._admin_switch(2))
        nav_layout.addWidget(btn_db)
        self.nav_buttons.append(btn_db)

        nav_layout.addStretch()

        # Admin holat ko'rsatgich
        self.admin_status = QLabel("👤  Mehmon rejimi")
        self.admin_status.setAlignment(Qt.AlignCenter)
        self.admin_status.setStyleSheet(
            "color:#7f8c8d;font-size:11px;padding:6px 0;"
        )
        nav_layout.addWidget(self.admin_status)

        # Chiqish tugmasi (admin kirganda ko'rinadi)
        self.btn_logout = QPushButton("🚪  Chiqish")
        self.btn_logout.setFixedHeight(36)
        self.btn_logout.setVisible(False)
        self.btn_logout.setStyleSheet("""
            QPushButton {
                background:#c0392b;color:white;border:none;
                font-size:12px;border-radius:0;
            }
            QPushButton:hover { background:#a93226; }
        """)
        self.btn_logout.clicked.connect(self._logout)
        nav_layout.addWidget(self.btn_logout)

        version_lbl = QLabel("v1.0  ·  Diplom loyiha")
        version_lbl.setAlignment(Qt.AlignCenter)
        version_lbl.setStyleSheet("color:#4a4a6a;font-size:10px;padding:10px 0;")
        nav_layout.addWidget(version_lbl)

        root.addWidget(nav)

        # ---- O'NG: Asosiy kontent ----
        content = QFrame()
        content.setStyleSheet("background:#f4f6f8;")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget()
        self.stack.setStyleSheet("background:#f4f6f8;")

        self.rec_widget = RecognizeWidget()
        self.reg_widget = RegisterWidget()
        self.db_widget  = DatabaseWidget()

        self.reg_widget.person_saved.connect(self._on_person_saved)

        self.stack.addWidget(self.rec_widget)   # 0
        self.stack.addWidget(self.reg_widget)   # 1
        self.stack.addWidget(self.db_widget)    # 2

        content_layout.addWidget(self.stack)
        root.addWidget(content)

        # Holat
        self._is_admin = False
        self._switch_tab(0)

    def _nav_btn_style(self):
        return """
            QPushButton {
                background:transparent;color:#bdc3c7;
                border:none;text-align:left;
                padding:0 20px;font-size:13px;border-radius:0;
            }
            QPushButton:hover { background:#16213e;color:white; }
            QPushButton:checked { background:#1D9E75;color:white;font-weight:bold; }
        """

    def _switch_tab(self, idx):
        self.stack.setCurrentIndex(idx)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == idx)
        if idx == 2:
            self.db_widget.load_data()

    def _admin_switch(self, idx):
        """Admin sahifaga o'tish — login so'raydi"""
        if self._is_admin:
            self._switch_tab(idx)
            return

        dlg = LoginDialog(self)
        if dlg.exec_() == LoginDialog.Accepted:
            self._is_admin = True
            # Qulf belgisini ochiq qildik
            self.nav_buttons[1].setText("➕  Ro'yxatga olish")
            self.nav_buttons[2].setText("🗃   Ma'lumotlar bazasi")
            self.admin_status.setText("🛡  Admin rejimi")
            self.admin_status.setStyleSheet(
                "color:#1D9E75;font-size:11px;padding:6px 0;font-weight:bold;"
            )
            self.btn_logout.setVisible(True)
            self._switch_tab(idx)

    def _logout(self):
        reply = QMessageBox.question(
            self, "Chiqish",
            "Admin rejimidan chiqmoqchimisiz?\nTanib olish sahifasiga qaytiladi.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self._is_admin = False
            self.nav_buttons[1].setText("🔒  Ro'yxatga olish")
            self.nav_buttons[2].setText("🔒  Ma'lumotlar bazasi")
            self.admin_status.setText("👤  Mehmon rejimi")
            self.admin_status.setStyleSheet(
                "color:#7f8c8d;font-size:11px;padding:6px 0;"
            )
            self.btn_logout.setVisible(False)
            self._switch_tab(0)

    def _on_person_saved(self):
        self.db_widget.load_data()
        self.rec_widget._reload_encodings()
        self._switch_tab(2)

    def closeEvent(self, event):
        self.rec_widget.closeEvent(event)
        self.reg_widget.closeEvent(event)
        super().closeEvent(event)


def main():
    init_db()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
