import sqlite3
import pickle
import os
from datetime import datetime


DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'face_system.db')


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Jadvallarni yaratadi (birinchi ishga tushganda)"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS persons (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name   TEXT NOT NULL,
            person_code TEXT UNIQUE,
            role        TEXT DEFAULT 'Talaba',
            created_at  TEXT
        );

        CREATE TABLE IF NOT EXISTS face_images (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id   INTEGER REFERENCES persons(id) ON DELETE CASCADE,
            image_path  TEXT NOT NULL,
            angle       TEXT
        );

        CREATE TABLE IF NOT EXISTS encodings (
            person_id       INTEGER UNIQUE REFERENCES persons(id) ON DELETE CASCADE,
            feature_vector  BLOB
        );
    """)
    conn.commit()
    conn.close()


def add_person(full_name, role, person_code=None):
    """Yangi shaxs qo'shadi, ID va kodni qaytaradi"""
    conn = get_connection()
    try:
        if not person_code:
            cur = conn.execute("SELECT COUNT(*) as cnt FROM persons")
            cnt = cur.fetchone()['cnt']
            person_code = f"#{cnt + 1:04d}"
        conn.execute(
            "INSERT INTO persons (full_name, role, person_code, created_at) VALUES (?,?,?,?)",
            (full_name, role, person_code, datetime.now().isoformat(timespec='seconds'))
        )
        conn.commit()
        person_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        return person_id, person_code
    finally:
        conn.close()


def add_face_image(person_id, image_path, angle="to'g'ri"):
    """Rasm yo'lini bazaga saqlaydi"""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO face_images (person_id, image_path, angle) VALUES (?,?,?)",
            (person_id, image_path, angle)
        )
        conn.commit()
    finally:
        conn.close()


def save_encoding(person_id, vectors):
    """
    Feature vektorlar ro'yxatini saqlaydi yoki yangilaydi.
    vectors: list of list (har bir rasm uchun alohida vektor)
    """
    conn = get_connection()
    try:
        # vectors — ro'yxat bo'lishi kerak, agar bitta vektor kelsa, list ichiga olamiz
        if isinstance(vectors, list) and len(vectors) > 0 and not isinstance(vectors[0], list):
            vectors = [vectors]
        blob = pickle.dumps(vectors)
        conn.execute(
            "INSERT OR REPLACE INTO encodings (person_id, feature_vector) VALUES (?,?)",
            (person_id, blob)
        )
        conn.commit()
    finally:
        conn.close()


def load_all_encodings():
    """
    Barcha encoding + shaxs ma'lumotlarini qaytaradi.
    Har bir shaxs uchun 'vectors' — barcha vektorlar ro'yxati.
    """
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT p.id, p.full_name, p.role, p.person_code, e.feature_vector
            FROM encodings e
            JOIN persons p ON p.id = e.person_id
        """).fetchall()
        result = []
        for row in rows:
            raw = pickle.loads(row['feature_vector'])
            # Eski format: bitta vektor (list of float)
            # Yangi format: list of list
            if isinstance(raw, list) and len(raw) > 0:
                if isinstance(raw[0], (int, float)):
                    # Eski format — bitta vektor
                    vectors = [raw]
                else:
                    # Yangi format — bir nechta vektor
                    vectors = raw
            else:
                vectors = [raw] if raw else []

            result.append({
                'id': row['id'],
                'full_name': row['full_name'],
                'role': row['role'],
                'person_code': row['person_code'],
                'vectors': vectors,
                # Backward compatibility uchun
                'vector': vectors[0] if vectors else None,
                'encoding': vectors[0] if vectors else None,
            })
        return result
    finally:
        conn.close()


def get_all_persons():
    """Barcha shaxslar ro'yxati"""
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT p.*, COUNT(fi.id) as photo_count,
                   CASE WHEN e.person_id IS NOT NULL THEN 1 ELSE 0 END as has_encoding
            FROM persons p
            LEFT JOIN face_images fi ON fi.person_id = p.id
            LEFT JOIN encodings e ON e.person_id = p.id
            GROUP BY p.id
            ORDER BY p.created_at DESC
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_person(person_id):
    """Shaxsni bazadan o'chiradi"""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT image_path FROM face_images WHERE person_id=?", (person_id,)
        ).fetchall()
        for row in rows:
            if os.path.exists(row['image_path']):
                os.remove(row['image_path'])
        conn.execute("DELETE FROM persons WHERE id=?", (person_id,))
        conn.commit()
    finally:
        conn.close()


def get_image_count(person_id):
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM face_images WHERE person_id=?", (person_id,)
        ).fetchone()
        return row['cnt']
    finally:
        conn.close()