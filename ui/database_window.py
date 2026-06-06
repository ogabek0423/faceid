import sys
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QAbstractItemView, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from database.db_manager import get_all_persons, delete_person


class DatabaseWidget(QWidget):
    refresh_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self.load_data()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # --- Sarlavha ---
        top = QHBoxLayout()
        title = QLabel("Bazadagi shaxslar")
        title.setFont(QFont("Arial", 13, QFont.Bold))
        top.addWidget(title)
        top.addStretch()

        self.count_label = QLabel("0 ta yozuv")
        self.count_label.setStyleSheet("color:#888;font-size:13px;")
        top.addWidget(self.count_label)

        btn_refresh = QPushButton("🔄  Yangilash")
        btn_refresh.setFixedHeight(32)
        btn_refresh.setStyleSheet(
            "border:1px solid #ccc;border-radius:5px;padding:0 14px;font-size:13px;"
        )
        btn_refresh.clicked.connect(self.load_data)
        top.addWidget(btn_refresh)
        root.addLayout(top)

        # --- Jadval ---
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["Ism familiya", "ID kod", "Toifa", "Rasmlar", "Holati", "Amal"]
        )

        hdr = self.table.horizontalHeader()
        # Ism — kengayadi
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        # Qolganlar — mazmumga qarab
        for col in range(1, 6):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeToContents)

        # Minimal ustun kengliklarini belgilash
        self.table.setColumnWidth(1, 80)
        self.table.setColumnWidth(2, 100)
        self.table.setColumnWidth(3, 80)
        self.table.setColumnWidth(4, 110)
        self.table.setColumnWidth(5, 90)

        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setDefaultSectionSize(36)
        self.table.verticalHeader().hide()
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                gridline-color: #f0f0f0;
                font-size: 13px;
                outline: none;
            }
            QTableWidget::item {
                padding: 4px 10px;
            }
            QTableWidget::item:selected {
                background: #E1F5EE;
                color: #085041;
            }
            QHeaderView::section {
                background: #f8f8f8;
                border: none;
                border-bottom: 2px solid #e0e0e0;
                padding: 8px 10px;
                font-weight: bold;
                font-size: 13px;
                color: #444;
            }
            QScrollBar:vertical {
                width: 8px;
                background: #f5f5f5;
            }
            QScrollBar::handle:vertical {
                background: #ccc;
                border-radius: 4px;
            }
        """)
        root.addWidget(self.table, stretch=1)

        # --- Statistika ---
        stats_row = QHBoxLayout()
        stats_row.setSpacing(10)
        self.stat_labels = {}
        for key, text, color in [
            ("total",   "Jami",       "#555"),
            ("ready",   "✓ Tayyor",   "#1D9E75"),
            ("pending", "⚠ Kutilmoqda", "#BA7517"),
        ]:
            lbl = QLabel(f"{text}: 0")
            lbl.setStyleSheet(
                f"background:#f5f5f5;padding:5px 14px;border-radius:6px;"
                f"font-size:13px;color:{color};font-weight:bold;"
            )
            self.stat_labels[key] = lbl
            stats_row.addWidget(lbl)
        stats_row.addStretch()
        root.addLayout(stats_row)

    def load_data(self):
        persons = get_all_persons()
        self.table.setRowCount(len(persons))
        self.count_label.setText(f"{len(persons)} ta yozuv")

        ready = 0
        for row_idx, p in enumerate(persons):
            # Ism familiya
            name_item = QTableWidgetItem(p['full_name'])
            name_item.setData(Qt.UserRole, p['id'])
            name_item.setFont(QFont("Arial", 13))
            self.table.setItem(row_idx, 0, name_item)

            # ID kod
            id_item = QTableWidgetItem(p['person_code'] or '—')
            id_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row_idx, 1, id_item)

            # Toifa
            role_item = QTableWidgetItem(p['role'] or '—')
            role_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row_idx, 2, role_item)

            # Rasmlar soni
            photo_item = QTableWidgetItem(str(p['photo_count']))
            photo_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row_idx, 3, photo_item)

            # Encoding holati
            has_enc = p['has_encoding']
            enc_text = "✓ Tayyor" if has_enc else "⚠ Yo'q"
            enc_item = QTableWidgetItem(enc_text)
            enc_item.setTextAlignment(Qt.AlignCenter)
            enc_item.setForeground(
                QColor("#1D9E75") if has_enc else QColor("#BA7517")
            )
            if has_enc:
                ready += 1
            self.table.setItem(row_idx, 4, enc_item)

            # O'chirish tugmasi
            btn = QPushButton("🗑  O'chirish")
            btn.setStyleSheet(
                "background:#fdecea;color:#c0392b;border:none;"
                "border-radius:5px;padding:4px 10px;font-size:12px;margin:2px;"
            )
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(
                lambda _, pid=p['id'], name=p['full_name']:
                    self._delete_person(pid, name)
            )
            self.table.setCellWidget(row_idx, 5, btn)

        self.stat_labels['total'].setText(f"Jami: {len(persons)}")
        self.stat_labels['ready'].setText(f"✓ Tayyor: {ready}")
        self.stat_labels['pending'].setText(f"⚠ Kutilmoqda: {len(persons) - ready}")

    def _delete_person(self, person_id, name):
        reply = QMessageBox.question(
            self, "O'chirish",
            f"'{name}' ni bazadan o'chirishni tasdiqlaysizmi?\nBarcha rasmlari ham o'chiriladi.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            delete_person(person_id)
            self.load_data()