import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QStackedWidget, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon

from database.db_manager import init_db
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
        nav.setStyleSheet(
            "background:#1a1a2e;border-right:1px solid #0f0f23;"
        )
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
        nav_items = [
            ("🔍  Tanib olish", 0),
            ("➕  Ro'yxatga olish", 1),
            ("🗃   Ma'lumotlar bazasi", 2),
        ]

        nav_layout.addSpacing(16)
        for label, idx in nav_items:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(44)
            btn.setStyleSheet("""
                QPushButton {
                    background:transparent;
                    color:#bdc3c7;
                    border:none;
                    text-align:left;
                    padding:0 20px;
                    font-size:13px;
                    border-radius:0;
                }
                QPushButton:hover {
                    background:#16213e;
                    color:white;
                }
                QPushButton:checked {
                    background:#1D9E75;
                    color:white;
                    font-weight:bold;
                }
            """)
            btn.clicked.connect(lambda _, i=idx: self._switch_tab(i))
            nav_layout.addWidget(btn)
            self.nav_buttons.append(btn)

        nav_layout.addStretch()

        # Quyi: versiya
        version_lbl = QLabel("v1.0  ·  Diplom loyiha")
        version_lbl.setAlignment(Qt.AlignCenter)
        version_lbl.setStyleSheet("color:#4a4a6a;font-size:10px;padding:12px 0;")
        nav_layout.addWidget(version_lbl)

        root.addWidget(nav)

        # ---- O'NG: Asosiy kontent ----
        content = QFrame()
        content.setStyleSheet("background:#f4f6f8;")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget()
        self.stack.setStyleSheet("background:#f4f6f8;")

        # Sahifalar
        self.rec_widget = RecognizeWidget()
        self.reg_widget = RegisterWidget()
        self.db_widget = DatabaseWidget()

        # Ro'yxatga olish tugallanganda DB ni yangilash
        self.reg_widget.person_saved.connect(self._on_person_saved)

        self.stack.addWidget(self.rec_widget)   # index 0
        self.stack.addWidget(self.reg_widget)   # index 1
        self.stack.addWidget(self.db_widget)    # index 2

        content_layout.addWidget(self.stack)
        root.addWidget(content)

        # Birinchi tabni faollashtirish
        self._switch_tab(0)

    def _switch_tab(self, idx):
        self.stack.setCurrentIndex(idx)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == idx)
        # DB tabga o'tganda ma'lumotlarni yangilash
        if idx == 2:
            self.db_widget.load_data()

    def _on_person_saved(self):
        """Yangi odam qo'shilganda"""
        self.db_widget.load_data()
        self.rec_widget._reload_encodings()
        self._switch_tab(2)  # DB ga o'tish

    def closeEvent(self, event):
        self.rec_widget.closeEvent(event)
        self.reg_widget.closeEvent(event)
        super().closeEvent(event)


def main():
    # Ma'lumotlar bazasini ishga tushirish
    init_db()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()