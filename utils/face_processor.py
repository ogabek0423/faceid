import cv2
import numpy as np
import os
import sys

try:
    import dlib
    DLIB_AVAILABLE = True
    # dlib modellarini yuklash
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PREDICTOR_PATH = os.path.join(BASE_DIR, '..', 'models', 'shape_predictor_68_face_landmarks.dat')
    FACE_REC_PATH = os.path.join(BASE_DIR, '..', 'models', 'dlib_face_recognition_resnet_model_v1.dat')

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH) if os.path.exists(PREDICTOR_PATH) else None
    face_rec_model = dlib.face_recognition_model_v1(FACE_REC_PATH) if os.path.exists(FACE_REC_PATH) else None
except ImportError:
    DLIB_AVAILABLE = False
    detector = None
    predictor = None
    face_rec_model = None


# Har bir mintaqa uchun nuqta indekslari
LANDMARK_REGIONS = {
    'jaw':       (0,  17,  (29, 158, 117)),   # yashil
    'left_brow': (17, 22,  (127, 119, 221)),  # binafsha
    'right_brow':(22, 27,  (127, 119, 221)),
    'nose':      (27, 36,  (232, 93,  36)),   # to'q sariq
    'left_eye':  (36, 42,  (55, 139, 211)),   # ko'k
    'right_eye': (42, 48,  (55, 139, 211)),
    'mouth':     (48, 68,  (212, 83,  126)),  # pushti
}


def detect_faces(frame):
    """Ramkadagi yuzlarni topadi, to'rtburchaklar ro'yxati qaytaradi"""
    if not DLIB_AVAILABLE or detector is None:
        # OpenCV fallback
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        return faces  # (x, y, w, h) formatda

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    result = []
    for d in dets:
        result.append((d.left(), d.top(), d.width(), d.height()))
    return result


def get_landmarks(frame, face_rect):
    """68 ta landmark nuqtasini qaytaradi"""
    if not DLIB_AVAILABLE or predictor is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x, y, w, h = face_rect
    rect = dlib.rectangle(x, y, x + w, y + h)
    shape = predictor(gray, rect)
    points = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    return points


def get_face_encoding(frame, face_rect):
    """128-o'lchamli feature vektor qaytaradi"""
    if not DLIB_AVAILABLE or predictor is None or face_rec_model is None:
        # Fallback: geometrik xususiyatlar
        return _geometry_encoding(frame, face_rect)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x, y, w, h = face_rect
    rect = dlib.rectangle(x, y, x + w, y + h)
    shape = predictor(gray, rect)
    descriptor = face_rec_model.compute_face_descriptor(frame, shape)
    return list(descriptor)


def _geometry_encoding(frame, face_rect):
    """dlib bo'lmasa geometrik xususiyatlar (demo uchun)"""
    x, y, w, h = face_rect
    face = frame[y:y+h, x:x+w]
    resized = cv2.resize(face, (64, 64))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    flat = gray.flatten().astype(np.float32)
    norm = flat / 255.0
    return norm[:128].tolist()


def draw_landmarks(frame, landmarks, show_connections=True):
    """Landmark nuqtalarni rasmga chizadi"""
    if landmarks is None:
        return frame

    result = frame.copy()

    for region, (start, end, color) in LANDMARK_REGIONS.items():
        pts = landmarks[start:end]
        bgr = (color[2], color[1], color[0])  # RGB -> BGR

        # Chiziqlar bilan ulash
        if show_connections and len(pts) > 1:
            if region == 'jaw':
                for i in range(len(pts) - 1):
                    cv2.line(result, pts[i], pts[i+1], bgr, 1, cv2.LINE_AA)
            elif 'eye' in region:
                for i in range(len(pts)):
                    cv2.line(result, pts[i], pts[(i+1) % len(pts)], bgr, 1, cv2.LINE_AA)
            elif region in ('mouth',):
                outer = pts[:12]
                for i in range(len(outer)):
                    cv2.line(result, outer[i], outer[(i+1) % len(outer)], bgr, 1, cv2.LINE_AA)
            else:
                for i in range(len(pts) - 1):
                    cv2.line(result, pts[i], pts[i+1], bgr, 1, cv2.LINE_AA)

        # Nuqtalarni chizish
        for pt in pts:
            cv2.circle(result, pt, 2, bgr, -1, cv2.LINE_AA)

    return result


def draw_face_box(frame, face_rect, label=None, confidence=None, color=(29, 158, 117)):
    """Yuz atrofida to'rtburchak va nom chizadi"""
    result = frame.copy()
    x, y, w, h = face_rect
    bgr = (color[2], color[1], color[0])

    # Burchak chizig'i (to'rtburchak o'rniga zamonaviy ko'rinish)
    thickness = 2
    corner_len = min(w, h) // 5

    corners = [
        ((x, y), (x + corner_len, y), (x, y + corner_len)),
        ((x+w, y), (x+w - corner_len, y), (x+w, y + corner_len)),
        ((x, y+h), (x + corner_len, y+h), (x, y+h - corner_len)),
        ((x+w, y+h), (x+w - corner_len, y+h), (x+w, y+h - corner_len)),
    ]
    for corner, pt1, pt2 in corners:
        cv2.line(result, corner, pt1, bgr, thickness)
        cv2.line(result, corner, pt2, bgr, thickness)

    # Nom va moslik
    if label:
        text = label
        if confidence is not None:
            text += f"  {confidence:.0f}%"
        # Qora fon
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(result, (x, y - th - 10), (x + tw + 8, y), (0, 0, 0), -1)
        cv2.putText(result, text, (x + 4, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 1, cv2.LINE_AA)

    return result


import json  # Faylning eng yuqorisiga qo'shib qo'ying (agar bo'lmasa)


def compare_encodings(known_encodings, unknown_vector, threshold=0.6):
    """
    Noma'lum vektorni barcha ma'lum vektorlar bilan solishtiradi.
    Eng yaqin shaxsni qaytaradi yoki None.
    """
    if not known_encodings or unknown_vector is None:
        return None, 0.0

    unknown = np.array(unknown_vector)
    best_match = None
    best_dist = float('inf')

    for person in known_encodings:
        # 'encoding' yoki 'vector' kalitlaridan qaysi biri bo'lsa ham xavfsiz olish
        db_enc = person.get('encoding') if person.get('encoding') is not None else person.get('vector')

        if db_enc is None:
            continue

        # Agar bazadan kelgan ma'lumot matn (String) bo'lsa, uni listga o'giramiz
        if isinstance(db_enc, str):
            try:
                db_enc = json.loads(db_enc)
            except Exception as e:
                print(f"Matnni decoding qilishda xato: {e}")
                continue

        known = np.array(db_enc)

        if known.shape != unknown.shape:
            continue

        # Evklid masofasini hisoblash (Euclidean distance)
        dist = np.linalg.norm(known - unknown)
        if dist < best_dist:
            best_dist = dist
            best_match = person

    # Agar eng yaqin masofa belgilangan chegaradan katta bo'lsa - tanimaydi
    if best_dist > threshold:
        return None, 0.0

    # Masofani foizga (Confidence) o'girish mantiqi (Chiroyli ko'rinish uchun)
    # Masofa 0 ga yaqin bo'lsa foiz 100% ga yaqinlashadi
    confidence = (1.0 - (best_dist / (threshold * 1.5))) * 100
    confidence = max(0, min(100, confidence))

    # Qo'shimcha tekshiruv: agar tanilgan bo'lsa, asosiy oynadagi 'person_code' formatini to'g'rilash
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