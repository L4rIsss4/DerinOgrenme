import cv2
from deepface import DeepFace
import os
from datetime import datetime

# Kamera başlat
cap = cv2.VideoCapture(0)

# Yüz tespiti için sınıflandırıcıyı yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# DeepFace modelini yükle
model = DeepFace.build_model("DeepFace")

# Depolama dizini
storage_directory = "C://Users//okans//Desktop//derin_ogrenme//gorseller"
new_records_directory = "C://Users//okans//Desktop//derin_ogrenme//gorseller"

# Dizinlerin olup olmadığını kontrol et ve oluştur 
if not os.path.exists(storage_directory):
    os.makedirs(storage_directory)
if not os.path.exists(new_records_directory):
    os.makedirs(new_records_directory)

new_person_photos = []  # Yeni tanınmayan kişilerin fotoğraflarını saklamak için liste

while True:
    # Kameradan bir kare oku
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü gri tonlamaya dönüştür
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüz tespiti yap
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Yüzü kırp
        face_img = frame[y:y + h, x:x + w]

        # Yüzü DeepFace modeline gönder
        result = DeepFace.analyze(face_img, actions=['emotion', 'age', 'gender', 'race'], enforce_detection=False)

        # Tanıma yap
        recognized_person = False
        # Tanınan kişilerin listesini tutmak için bir liste oluştur
        recognized_persons = []

        for person_name in os.listdir(storage_directory):
            # Kayıtlı kişilerin fotoğraflarını yükle
            person_image_path = os.path.join(storage_directory, person_name)
            registered_person_img = cv2.imread(person_image_path)
            if registered_person_img is None:
                print(f"Hata: {person_name} kişisinin fotoğrafı yüklenemedi.")
                continue

            registered_person_gray = cv2.cvtColor(registered_person_img, cv2.COLOR_BGR2GRAY)

            # Kayıtlı kişilerin yüzlerini tespit et
            registered_person_faces = face_cascade.detectMultiScale(registered_person_gray, 1.3, 5)
            for (rx, ry, rw, rh) in registered_person_faces:
                registered_face_img = registered_person_img[ry:ry + rh, rx:rx + rw]

                # Yüzleri karşılaştır
                result_similarity = DeepFace.verify(face_img, registered_face_img, enforce_detection=False)
                if result_similarity["verified"]:
                    recognized_persons.append(person_name)
                    break

        if recognized_persons:
            print(f"Tanınan kişiler: {recognized_persons}")
            recognized_person = True

        if not recognized_person:
            # Yeni tanınmayan kişilerin fotoğrafını kaydet
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            photo_filename = f"unknown_person_{timestamp}.jpg"
            cv2.imwrite(os.path.join(new_records_directory, photo_filename), frame)
            print(f"Tanınmayan kişi fotoğrafı kaydedildi: {photo_filename}")
            new_person_photos.append(photo_filename)  # Yeni tanınmayan kişi fotoğrafını listeye ekle

    # Eğer yeni tanınmayan kişi tekrar tespit edilirse, yeni kayıt altında bulunan fotoğraflardan al
    if new_person_photos and not recognized_person:
        # En son eklenen fotoğrafı al
        new_person_photo = new_person_photos[-1]
        # Yeni kayıt altındaki fotoğrafı oku
        new_person_img = cv2.imread(os.path.join(new_records_directory, new_person_photo))
        if new_person_img is not None:
            # Yüz tespiti yap
            new_person_gray = cv2.cvtColor(new_person_img, cv2.COLOR_BGR2GRAY)
            new_person_faces = face_cascade.detectMultiScale(new_person_gray, 1.3, 5)
            for (nx, ny, nw, nh) in new_person_faces:
                new_person_face_img = new_person_img[ny:ny + nh, nx:nx + nw]
                # Tanıma yap
                result = DeepFace.analyze(new_person_face_img, actions=['emotion', 'age', 'gender', 'race'], enforce_detection=False)
                print("Yeni tanınan kişi:", result)
                recognized_person = True

    # Sonuçları görselleştir
    cv2.imshow('Face Recognition', frame)

    # Çıkış için 'q' tuşuna basılmasını bekleyin
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırakın ve pencereleri kapatın
cap.release()
cv2.destroyAllWindows()
