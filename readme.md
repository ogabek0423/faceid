# Yuz Tanib Olish Tizimi
## Diplom ishi loyihasi — PyQt5 + OpenCV + dlib

---

## Fayl strukturasi

```
face_recognition/
├── main.py                    ← Ishga tushirish nuqtasi
├── requirements.txt           ← Kerakli kutubxonalar
├── database/
│   └── db_manager.py          ← SQLite boshqaruvi
├── utils/
│   └── face_processor.py      ← Yuz aniqlash va landmark
├── ui/
│   ├── register_window.py     ← Ro'yxatga olish oynasi
│   ├── recognize_window.py    ← Tanib olish oynasi (3 panel)
│   └── database_window.py     ← Bazadagi shaxslar
├── models/                    ← dlib modellari (yuklab qo'yish kerak)
├── faces/                     ← Rasmlar saqlanadigan joy
└── data/
    └── face_system.db         ← SQLite fayl (avtomatik yaratiladi)
```

---

## O'rnatish

### 1. Kutubxonalarni o'rnatish
```bash
pip install -r requirements.txt
```

### 2. dlib modellarini yuklab olish (MUHIM!)
Bu ikkita faylni `models/` papkasiga qo'ying:

**shape_predictor_68_face_landmarks.dat**
Yuklab olish: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

**dlib_face_recognition_resnet_model_v1.dat**
Yuklab olish: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

Arxivni ochish:
```bash
bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
bzip2 -dk dlib_face_recognition_resnet_model_v1.dat.bz2
```

### 3. Ishga tushirish
```bash
python main.py
```

---

## Foydalanish

### Yangi shaxs qo'shish:
1. "Ro'yxatga olish" bo'limiga o'ting
2. Ism, familiya, toifani kiriting
3. "Yangi shaxs boshlash" tugmasini bosing
4. Kamerani yoqing
5. 5 ta burchakdan (to'g'ri, chap, o'ng, yuqori, quyi) rasm oling
6. "Bazaga saqlash" tugmasini bosing

### Tanib olish:
1. "Tanib olish" bo'limiga o'ting
2. Kamerani yoqing
3. Kameraga qarang — natija pastda ko'rinadi

---

## Texnik ma'lumotlar

- **Yuz aniqlash:** dlib HOG detector (fallback: OpenCV Haar)
- **Landmark:** dlib 68 nuqta modeli
- **Encoding:** dlib ResNet 128-o'lchamli vektor
- **Klassifikatsiya:** Evklid masofa (threshold: 0.45)
- **Ma'lumotlar bazasi:** SQLite3
- **Interfeys:** PyQt5