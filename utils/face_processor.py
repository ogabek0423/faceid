import cv2
import numpy as np
import os
import sys
import json

try:
    import dlib
    DLIB_AVAILABLE = True
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PREDICTOR_PATH = os.path.join(BASE_DIR, '..', 'models', 'shape_predictor_68_face_landmarks.dat')
    FACE_REC_PATH = os.path.join(BASE_DIR, '..', 'models', 'dlib_face_recognition_resnet_model_v1.dat')

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH) if os.path.exists(PREDICTOR_PATH) else None
    face_rec_model = dlib.face_recognition_model_v1(FACE_REC_PATH) if os.path.exists(FACE_REC_PATH) else None

    if predictor is None:
        print(f"[OGOHLANTIRISH] shape_predictor_68_face_landmarks.dat topilmadi: {PREDICTOR_PATH}")
    if face_rec_model is None:
        print(f"[OGOHLANTIRISH] dlib_face_recognition_resnet_model_v1.dat topilmadi: {FACE_REC_PATH}")

except ImportError:
    DLIB_AVAILABLE = False
    detector = None
    predictor = None
    face_rec_model = None
    print("[OGOHLANTIRISH] dlib o'rnatilmagan. OpenCV fallback ishlatiladi.")


LANDMARK_REGIONS = {
    'jaw':        (0,  17,  (29, 200, 100)),
    'left_brow':  (17, 22,  (180, 100, 230)),
    'right_brow': (22, 27,  (180, 100, 230)),
    'nose':       (27, 36,  (232, 160, 36)),
    'left_eye':   (36, 42,  (80, 180, 255)),
    'right_eye':  (42, 48,  (80, 180, 255)),
    'mouth':      (48, 68,  (240, 100, 160)),
}


def detect_faces(frame):
    """Ramkadagi yuzlarni topadi — dlib yoki OpenCV"""
    if not DLIB_AVAILABLE or detector is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            return []
        return [tuple(f) for f in faces]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # upsampling=1 — kichik yuzlarni ham topish uchun
    dets = detector(gray, 1)
    result = []
    for d in dets:
        result.append((d.left(), d.top(), d.width(), d.height()))
    return result


def get_landmarks(frame, face_rect):
    """68 ta landmark nuqtasini qaytaradi"""
    if not DLIB_AVAILABLE or predictor is None:
        # dlib modeli yo'q — OpenCV LBF yoki geometrik taxmin
        return _geometry_landmarks(frame, face_rect)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x, y, w, h = face_rect
    # Chetlarni biroz kengaytirish — aniqroq landmark uchun
    pad = int(min(w, h) * 0.1)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + w + pad)
    y2 = min(frame.shape[0], y + h + pad)
    rect = dlib.rectangle(x1, y1, x2, y2)
    shape = predictor(gray, rect)
    points = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    return points


def _geometry_landmarks(frame, face_rect):
    """
    dlib modeli bo'lmasa — OpenCV FacemarkLBF yoki oddiy geometrik taxmin.
    68 ta nuqtani taxminiy joylashtiramiz.
    """
    x, y, w, h = face_rect
    pts = []

    # Jaw (0-16) — pastki qism
    for i in range(17):
        px = int(x + w * (i / 16.0))
        py = int(y + h * (0.75 + 0.25 * abs(i / 8.0 - 1)))
        pts.append((px, py))

    # Left brow (17-21)
    for i in range(5):
        pts.append((int(x + w * (0.15 + i * 0.07)), int(y + h * 0.25)))

    # Right brow (22-26)
    for i in range(5):
        pts.append((int(x + w * (0.55 + i * 0.07)), int(y + h * 0.25)))

    # Nose bridge (27-30)
    for i in range(4):
        pts.append((int(x + w * 0.5), int(y + h * (0.3 + i * 0.08))))

    # Nose tip (31-35)
    for i in range(5):
        pts.append((int(x + w * (0.38 + i * 0.06)), int(y + h * 0.6)))

    # Left eye (36-41)
    ex, ey = int(x + w * 0.25), int(y + h * 0.38)
    ew, eh = int(w * 0.18), int(h * 0.08)
    for ang in [0, 30, 150, 180, 210, 330]:
        rad = np.radians(ang)
        pts.append((int(ex + ew * np.cos(rad)), int(ey + eh * np.sin(rad))))

    # Right eye (42-47)
    ex2, ey2 = int(x + w * 0.75), int(y + h * 0.38)
    for ang in [0, 30, 150, 180, 210, 330]:
        rad = np.radians(ang)
        pts.append((int(ex2 + ew * np.cos(rad)), int(ey2 + eh * np.sin(rad))))

    # Outer mouth (48-59)
    mx, my = int(x + w * 0.5), int(y + h * 0.75)
    mw, mh = int(w * 0.22), int(h * 0.08)
    for i in range(12):
        ang = np.radians(i * 30)
        pts.append((int(mx + mw * np.cos(ang)), int(my + mh * np.sin(ang))))

    # Inner mouth (60-67)
    mw2, mh2 = int(w * 0.15), int(h * 0.05)
    for i in range(8):
        ang = np.radians(i * 45)
        pts.append((int(mx + mw2 * np.cos(ang)), int(my + mh2 * np.sin(ang))))

    return pts[:68]


def get_face_encoding(frame, face_rect):
    """128-o'lchamli feature vektor qaytaradi"""
    if not DLIB_AVAILABLE or predictor is None or face_rec_model is None:
        return _geometry_encoding(frame, face_rect)

    # Rasmni yuzlash uchun preprocessing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x, y, w, h = face_rect
    pad = int(min(w, h) * 0.1)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + w + pad)
    y2 = min(frame.shape[0], y + h + pad)
    rect = dlib.rectangle(x1, y1, x2, y2)
    shape = predictor(gray, rect)
    descriptor = face_rec_model.compute_face_descriptor(frame_rgb, shape, num_jitters=1)
    return list(descriptor)


def _geometry_encoding(frame, face_rect):
    """
    Yaxshilangan fallback encoding:
    HOG + pixel histogram + nisbatlar kombinatsiyasi
    """
    x, y, w, h = face_rect
    # Padding qo'shish
    pad = int(min(w, h) * 0.15)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + w + pad)
    y2 = min(frame.shape[0], y + h + pad)

    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return [0.0] * 128

    # 64x64 ga keltirish
    face_resized = cv2.resize(face, (64, 64))

    # CLAHE bilan normalizatsiya — yorug'lik muammosini kamaytirish
    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray_eq = clahe.apply(gray)

    # HOG descriptor (kichik versiya)
    hog = cv2.HOGDescriptor(
        (64, 64), (16, 16), (8, 8), (8, 8), 9
    )
    hog_feat = hog.compute(gray_eq).flatten()

    # Pixel xususiyatlari
    flat = gray_eq.flatten().astype(np.float32) / 255.0

    # Birinchi 128 ta xususiyat
    combined = np.concatenate([hog_feat[:64], flat[:64]])
    if combined.shape[0] < 128:
        combined = np.pad(combined, (0, 128 - combined.shape[0]))

    # L2 normalizatsiya
    norm = np.linalg.norm(combined[:128])
    if norm > 0:
        combined = combined[:128] / norm
    return combined[:128].tolist()


def draw_landmarks(frame, landmarks, show_connections=True):
    """Landmark nuqtalarni rasmga chizadi"""
    if landmarks is None or len(landmarks) < 68:
        return frame

    result = frame.copy()

    for region, (start, end, color) in LANDMARK_REGIONS.items():
        pts = landmarks[start:end]
        bgr = (color[2], color[1], color[0])  # RGB -> BGR

        if show_connections and len(pts) > 1:
            if region == 'jaw':
                for i in range(len(pts) - 1):
                    cv2.line(result, pts[i], pts[i+1], bgr, 1, cv2.LINE_AA)
            elif 'eye' in region:
                for i in range(len(pts)):
                    cv2.line(result, pts[i], pts[(i+1) % len(pts)], bgr, 1, cv2.LINE_AA)
            elif region == 'mouth':
                outer = pts[:12]
                for i in range(len(outer)):
                    cv2.line(result, outer[i], outer[(i+1) % len(outer)], bgr, 1, cv2.LINE_AA)
            else:
                for i in range(len(pts) - 1):
                    cv2.line(result, pts[i], pts[i+1], bgr, 1, cv2.LINE_AA)

        for idx_pt, pt in enumerate(pts):
            cv2.circle(result, pt, 2, bgr, -1, cv2.LINE_AA)

    return result


def draw_face_box(frame, face_rect, label=None, confidence=None, color=(29, 158, 117)):
    """Yuz atrofida burchakli ramka va nom chizadi"""
    result = frame.copy()
    x, y, w, h = face_rect
    bgr = (color[2], color[1], color[0])

    thickness = 2
    corner_len = min(w, h) // 5

    corners = [
        ((x, y),     (x + corner_len, y),     (x, y + corner_len)),
        ((x+w, y),   (x+w - corner_len, y),   (x+w, y + corner_len)),
        ((x, y+h),   (x + corner_len, y+h),   (x, y+h - corner_len)),
        ((x+w, y+h), (x+w - corner_len, y+h), (x+w, y+h - corner_len)),
    ]
    for corner, pt1, pt2 in corners:
        cv2.line(result, corner, pt1, bgr, thickness)
        cv2.line(result, corner, pt2, bgr, thickness)

    if label:
        text = label
        if confidence is not None:
            text += f"  {confidence:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(result, (x, y - th - 10), (x + tw + 8, y), (0, 0, 0), -1)
        cv2.putText(result, text, (x + 4, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 1, cv2.LINE_AA)

    return result


def compare_encodings(known_encodings, unknown_vector, threshold=0.55):
    """
    Noma'lum vektorni barcha ma'lum vektorlar bilan solishtiradi.
    Eng yaqin shaxsni qaytaradi yoki None.

    dlib uchun tavsiya etilgan threshold: 0.6 (lekin 0.55 aniqroq)
    Fallback uchun cosine similarity ishlatiladi.
    """
    if not known_encodings or unknown_vector is None:
        return None, 0.0

    unknown = np.array(unknown_vector, dtype=np.float64)
    best_match = None
    best_dist = float('inf')
    best_score = 0.0

    for person in known_encodings:
        # 'encoding' yoki 'vector' kalitidan olish
        db_enc = person.get('encoding') or person.get('vector')

        if db_enc is None:
            continue

        if isinstance(db_enc, str):
            try:
                db_enc = json.loads(db_enc)
            except Exception:
                continue

        known = np.array(db_enc, dtype=np.float64)

        if known.shape != unknown.shape:
            continue

        # Evklid masofasi
        dist = float(np.linalg.norm(known - unknown))

        if dist < best_dist:
            best_dist = dist
            best_match = person

    if best_dist > threshold:
        return None, 0.0

    # Aniqroq confidence formula:
    # dist=0 -> 100%, dist=threshold -> 0%
    # Lekin threshold dan ancha kichik bo'lsa 95%+ chiqarish
    ratio = best_dist / threshold  # 0..1
    confidence = (1.0 - ratio) * 100.0
    confidence = max(0.0, min(100.0, confidence))

    # person_code normalizatsiya
    if best_match and 'person_code' not in best_match:
        best_match['person_code'] = best_match.get('code', 'ID: ' + str(best_match.get('id', '???')))

    return best_match, confidence


def save_face_image(frame, person_id, angle, base_dir):
    """Rasmni diskka saqlaydi, yo'lni qaytaradi"""
    folder = os.path.join(base_dir, 'faces', str(person_id))
    os.makedirs(folder, exist_ok=True)
    filename = f"{angle.replace(' ', '_')}_{len(os.listdir(folder))}.jpg"
    path = os.path.join(folder, filename)
    cv2.imwrite(path, frame)
    return path


def compare_encodings_multi(known_encodings, unknown_vector, threshold=0.55):
    """
    Yangi format: har bir shaxs uchun bir nechta vektor.
    Eng yaqin masofani topadi.
    """
    if not known_encodings or unknown_vector is None:
        return None, 0.0

    import numpy as np
    unknown = np.array(unknown_vector, dtype=np.float64)
    best_match = None
    best_dist = float('inf')

    for person in known_encodings:
        # Yangi format: 'vectors' — ro'yxat
        vectors = person.get('vectors') or []
        if not vectors:
            # Eski format fallback
            v = person.get('encoding') or person.get('vector')
            vectors = [v] if v is not None else []

        for db_vec in vectors:
            if db_vec is None:
                continue
            if isinstance(db_vec, str):
                try:
                    import json
                    db_vec = json.loads(db_vec)
                except Exception:
                    continue

            known = np.array(db_vec, dtype=np.float64)
            if known.shape != unknown.shape:
                continue

            dist = float(np.linalg.norm(known - unknown))
            if dist < best_dist:
                best_dist = dist
                best_match = person

    if best_dist > threshold:
        return None, 0.0

    confidence = max(0.0, min(100.0, (1.0 - best_dist / threshold) * 100.0))

    if best_match and 'person_code' not in best_match:
        best_match['person_code'] = best_match.get('code', 'ID: ' + str(best_match.get('id', '???')))

    return best_match, confidence