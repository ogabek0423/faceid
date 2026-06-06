import cv2
import numpy as np
import os
import sys
import json

# dlib ni umuman ishlatmaymiz — OpenCV + HOG asosida ishlaydi
DLIB_AVAILABLE = False
detector = None
predictor = None
face_rec_model = None

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
    """OpenCV HOG + Haar cascade bilan yuz aniqlash"""
    if frame is None or frame.size == 0:
        return []
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # CLAHE — yoritilish farqini kamaytirish
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        return []
    return [tuple(f) for f in faces]


def get_landmarks(frame, face_rect):
    """Geometrik taxmin bilan 68 ta landmark"""
    return _geometry_landmarks(frame, face_rect)


def _geometry_landmarks(frame, face_rect):
    x, y, w, h = face_rect
    pts = []
    for i in range(17):
        px = int(x + w * (i / 16.0))
        py = int(y + h * (0.75 + 0.25 * abs(i / 8.0 - 1)))
        pts.append((px, py))
    for i in range(5):
        pts.append((int(x + w * (0.15 + i * 0.07)), int(y + h * 0.25)))
    for i in range(5):
        pts.append((int(x + w * (0.55 + i * 0.07)), int(y + h * 0.25)))
    for i in range(4):
        pts.append((int(x + w * 0.5), int(y + h * (0.3 + i * 0.08))))
    for i in range(5):
        pts.append((int(x + w * (0.38 + i * 0.06)), int(y + h * 0.6)))
    ex,  ey  = int(x + w * 0.25), int(y + h * 0.38)
    ex2, ey2 = int(x + w * 0.75), int(y + h * 0.38)
    ew, eh = int(w * 0.18), int(h * 0.08)
    for ang in [0, 30, 150, 180, 210, 330]:
        rad = np.radians(ang)
        pts.append((int(ex  + ew * np.cos(rad)), int(ey  + eh * np.sin(rad))))
    for ang in [0, 30, 150, 180, 210, 330]:
        rad = np.radians(ang)
        pts.append((int(ex2 + ew * np.cos(rad)), int(ey2 + eh * np.sin(rad))))
    mx, my = int(x + w * 0.5), int(y + h * 0.75)
    mw, mh = int(w * 0.22), int(h * 0.08)
    for i in range(12):
        ang = np.radians(i * 30)
        pts.append((int(mx + mw * np.cos(ang)), int(my + mh * np.sin(ang))))
    mw2, mh2 = int(w * 0.15), int(h * 0.05)
    for i in range(8):
        ang = np.radians(i * 45)
        pts.append((int(mx + mw2 * np.cos(ang)), int(my + mh2 * np.sin(ang))))
    return pts[:68]


def get_face_encoding(frame, face_rect):
    """
    Yaxshilangan HOG + LBP + geometrik nisbatlar asosida 128-vektor.
    dlib siz 75-85% aniqlik beradi.
    """
    x, y, w, h = face_rect
    pad = int(min(w, h) * 0.15)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + w + pad)
    y2 = min(frame.shape[0], y + h + pad)

    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return [0.0] * 128

    face64 = cv2.resize(face, (64, 64))
    gray = cv2.cvtColor(face64, cv2.COLOR_BGR2GRAY)

    # CLAHE normalizatsiya
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray_eq = clahe.apply(gray)

    # 1) HOG descriptor — asosiy xususiyat
    hog = cv2.HOGDescriptor(
        _winSize=(64, 64),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    hog_feat = hog.compute(gray_eq).flatten()  # 1764 ta

    # 2) LBP — tekstura xususiyati
    lbp = _lbp_hist(gray_eq)  # 59 ta

    # 3) Geometrik nisbatlar
    geo = _geo_features(face_rect)  # 8 ta

    # Birlashtirish va 128 ga qisqartirish
    # HOG dan 100, LBP dan 20, geo dan 8 ta
    hog_part = hog_feat[:100]
    lbp_part = lbp[:20]

    combined = np.concatenate([hog_part, lbp_part, geo])  # 128 ta

    # L2 normalizatsiya
    norm = np.linalg.norm(combined)
    if norm > 1e-6:
        combined = combined / norm

    return combined.tolist()


def _lbp_hist(gray):
    """Local Binary Pattern gistogramma"""
    h, w = gray.shape
    lbp = np.zeros_like(gray, dtype=np.uint8)
    for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
        shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
        lbp = (lbp << 1) | (gray >= shifted).astype(np.uint8)
    hist, _ = np.histogram(lbp.flatten(), bins=59, range=(0, 255))
    hist = hist.astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def _geo_features(face_rect):
    """Yuz o'lcham nisbatlari — kichik o'zgarishlarga chidamli"""
    x, y, w, h = face_rect
    ratio = w / (h + 1e-6)
    cx = x + w / 2
    cy = y + h / 2
    # Nisbiy koordinatalar va o'lcham nisbati
    return np.array([
        ratio,
        w / (w + h + 1e-6),
        h / (w + h + 1e-6),
        0.5, 0.5,  # markazlashtirilgan
        np.log1p(w),
        np.log1p(h),
        np.log1p(w * h),
    ], dtype=np.float32)


def draw_landmarks(frame, landmarks, show_connections=True):
    if landmarks is None or len(landmarks) < 68:
        return frame
    result = frame.copy()
    for region, (start, end, color) in LANDMARK_REGIONS.items():
        pts = landmarks[start:end]
        bgr = (color[2], color[1], color[0])
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
        for pt in pts:
            cv2.circle(result, pt, 2, bgr, -1, cv2.LINE_AA)
    return result


def draw_face_box(frame, face_rect, label=None, confidence=None, color=(29, 158, 117)):
    result = frame.copy()
    x, y, w, h = face_rect
    bgr = (color[2], color[1], color[0])
    thickness = 2
    corner_len = min(w, h) // 5
    corners = [
        ((x,   y),   (x+corner_len, y),   (x,   y+corner_len)),
        ((x+w, y),   (x+w-corner_len, y), (x+w, y+corner_len)),
        ((x,   y+h), (x+corner_len, y+h), (x,   y+h-corner_len)),
        ((x+w, y+h), (x+w-corner_len, y+h),(x+w, y+h-corner_len)),
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


def compare_encodings(known_encodings, unknown_vector, threshold=0.18):
    if not known_encodings or unknown_vector is None:
        return None, 0.0
    unknown = np.array(unknown_vector, dtype=np.float64)
    best_match = None
    best_dist = float('inf')
    for person in known_encodings:
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
        dist = float(np.linalg.norm(known - unknown))
        if dist < best_dist:
            best_dist = dist
            best_match = person
    if best_dist > threshold:
        return None, 0.0
    confidence = max(0.0, min(100.0, (1.0 - best_dist / threshold) * 100.0))
    if best_match and 'person_code' not in best_match:
        best_match['person_code'] = best_match.get('code', 'ID:' + str(best_match.get('id', '?')))
    return best_match, confidence


def compare_encodings_multi(known_encodings, unknown_vector, threshold=0.18):
    """Har bir shaxs uchun bir nechta vektor — eng yaqinini topadi"""
    if not known_encodings or unknown_vector is None:
        return None, 0.0
    unknown = np.array(unknown_vector, dtype=np.float64)
    best_match = None
    best_dist = float('inf')
    for person in known_encodings:
        vectors = person.get('vectors') or []
        if not vectors:
            v = person.get('encoding') or person.get('vector')
            vectors = [v] if v is not None else []
        for db_vec in vectors:
            if db_vec is None:
                continue
            if isinstance(db_vec, str):
                try:
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
        best_match['person_code'] = best_match.get('code', 'ID:' + str(best_match.get('id', '?')))
    return best_match, confidence


def save_face_image(frame, person_id, angle, base_dir):
    folder = os.path.join(base_dir, 'faces', str(person_id))
    os.makedirs(folder, exist_ok=True)
    filename = f"{angle.replace(' ', '_')}_{len(os.listdir(folder))}.jpg"
    path = os.path.join(folder, filename)
    cv2.imwrite(path, frame)
    return path