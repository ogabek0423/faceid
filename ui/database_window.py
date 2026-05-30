import sys
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QAbstractItemView
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

        # Sarlavha
        top = QHBoxLayout()
        title = QLabel("Bazadagi shaxslar")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        top.addWidget(title)
        top.addStretch()

        self.count_label = QLabel("0 ta yozuv")
        self.count_label.setStyleSheet("color:#888;font-size:13px;")
        top.addWidget(self.count_label)

        btn_refresh = QPushButton("🔄  Yangilash")
        btn_refresh.setFixedHeight(30)
        btn_refresh.setStyleSheet(
            "border:1px solid #ccc;border-radius:5px;padding:0 12px;font-size:12px;"
        )
        btn_refresh.clicked.connect(self.load_data)
        top.addWidget(btn_refresh)

        root.addLayout(top)

        # Jadval
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["Ism familiya", "ID", "Toifa", "Rasmlar", "Encoding", "Amal"]
        )
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("""
            QTableWidget {
                border:1px solid #e0e0e0;
                border-radius:8px;
                gridline-color:#f0f0f0;
                font-size:13px;
            }
            QHeaderView::section {
                background:#f8f8f8;
                border:none;
                border-bottom:1px solid #e0e0e0;
                padding:6px 8px;
                font-weight:bold;
                color:#555;
            }
            QTableWidget::item { padding:4px 8px; }
            QTableWidget::item:selected { background:#E1F5EE; color:#085041; }
        """)
        root.addWidget(self.table)

        # Statistika qator
        stats_row = QHBoxLayout()
        self.stat_labels = {}
        for key, text in [("total", "Jami"), ("ready", "Tayyor"), ("pending", "Kutilmoqda")]:
            lbl = QLabel(f"{text}: 0")
            lbl.setStyleSheet(
                "background:#f5f5f5;padding:4px 12px;border-radius:5px;font-size:12px;color:#666;"
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
            # Ism
            name_item = QTableWidgetItem(p['full_name'])
            name_item.setData(Qt.UserRole, p['id'])
            self.table.setItem(row_idx, 0, name_item)

            # ID
            self.table.setItem(row_idx, 1, QTableWidgetItem(p['person_code'] or '—'))

            # Toifa
            self.table.setItem(row_idx, 2, QTableWidgetItem(p['role'] or '—'))

            # Rasmlar soni
            photo_item = QTableWidgetItem(str(p['photo_count']))
            photo_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row_idx, 3, photo_item)

            # Encoding holati
            has_enc = p['has_encoding']
            enc_item = QTableWidgetItem("✓ Tayyor" if has_enc else "⚠ Yo'q")
            enc_item.setTextAlignment(Qt.AlignCenter)
            if has_enc:
                enc_item.setForeground(QColor("#1D9E75"))
                ready += 1
            else:
                enc_item.setForeground(QColor("#BA7517"))
            self.table.setItem(row_idx, 4, enc_item)

            # O'chirish tugmasi
            btn = QPushButton("O'chirish")
            btn.setStyleSheet(
                "background:#fdecea;color:#c0392b;border:none;"
                "border-radius:4px;padding:3px 10px;font-size:11px;"
            )
            btn.clicked.connect(lambda _, pid=p['id'], name=p['full_name']:
                                self._delete_person(pid, name))
            self.table.setCellWidget(row_idx, 5, btn)

        # Statistika
        self.stat_labels['total'].setText(f"Jami: {len(persons)}")
        self.stat_labels['ready'].setText(f"Tayyor: {ready}")
        self.stat_labels['pending'].setText(f"Kutilmoqda: {len(persons) - ready}")

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