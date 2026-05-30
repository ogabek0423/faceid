import os
import bz2
import urllib.request

# Fayllar nomi va ularning yuklab olish havolalari
MODELS_TO_DOWNLOAD = {
    "shape_predictor_68_face_landmarks.dat": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
    "dlib_face_recognition_resnet_model_v1.dat": "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
}


def download_and_extract_models(target_dir="models"):
    # Target papkani yaratish (agar mavjud bo'lmasa)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"'{target_dir}' papkasi yaratildi.")

    for file_name, url in MODELS_TO_DOWNLOAD.items():
        destination_path = os.path.join(target_dir, file_name)

        # Agar fayl allaqachon bor bo'lsa, qayta yuklab o'tirmaydi
        if os.path.exists(destination_path):
            print(f"[{file_name}] allaqachon mavjud. O'tkazib yuborildi.")
            continue

        archive_name = file_name + ".bz2"
        archive_path = os.path.join(target_dir, archive_name)

        try:
            print(f"\n{file_name} yuklab olinmoqda...")
            # Faylni yuklab olish
            urllib.request.urlretrieve(url, archive_path)
            print(f"Yuklab olindi. Arxivdan chiqarilmoqda...")

            # bz2 arxivni ochish va .dat faylga yozish
            with bz2.BZ2File(archive_path, 'rb') as source, open(destination_path, 'wb') as dest:
                dest.write(source.read())

            # Vaqtincha yuklab olingan .bz2 arxivni o'chirib tashlash
            os.remove(archive_path)
            print(f"Muvaffaqiyatli bajarildi: {destination_path}")

        except Exception as e:
            print(f"Xatolik yuz berdi ({file_name}): {e}")
            if os.path.exists(archive_path):
                os.remove(archive_path)


if __name__ == "__main__":
    print("Dlib modellari yuklanishni boshlamoqda...")
    download_and_extract_models()
    print("\nJarayon yakunlandi!")