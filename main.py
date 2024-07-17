import threading
import tkinter as tk
from tkinter import messagebox
import cv2
import dlib
import numpy as np
import time
import requests
import tempfile
import os
import dropbox
import photo_manager

known_face_descriptors = None
known_face_names = None
dbx = None
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
TELEGRAM_BOT_TOKEN = 'your token'
TELEGRAM_CHAT_ID = 'your token'
DROPBOX_ACCESS_TOKEN = 'your token'

def main():
    global dbx
    authenticate_dropbox()
    
    root = tk.Tk()
    root.title("Главное меню проекта")
    root.geometry("300x200")
    
    delete_photo_button = tk.Button(root, text="Удалить фото", command=lambda: photo_manager.delete_photo(dbx))
    delete_photo_button.pack(pady=10)
    
    add_photo_button = tk.Button(root, text="Добавить фото", command=lambda: photo_manager.add_photo(dbx))
    add_photo_button.pack(pady=10)
    
    root.mainloop()

def authenticate_dropbox():
    global dbx
    dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

def list_files_in_dropbox_folder(folder_path=""):
    global dbx
    try:
        files = dbx.files_list_folder(folder_path).entries
        file_paths = [file.path_lower for file in files if isinstance(file, dropbox.files.FileMetadata)]
        return file_paths
    except dropbox.exceptions.AuthError as e:
        print(f"Ошибка аутентификации Dropbox: {e}")
        return []
    except Exception as e:
        print(f"Ошибка при получении списка файлов из Dropbox: {e}")
        return []

def download_file_from_dropbox(dropbox_path, local_path):
    global dbx
    try:
        with open(local_path, "wb") as f:
            metadata, res = dbx.files_download(path=dropbox_path)
            f.write(res.content)
        print(f"Файл {dropbox_path} успешно загружен из Dropbox в {local_path}")
    except Exception as e:
        print(f"Ошибка загрузки файла из Dropbox: {e}")

def extract_face_descriptors(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_image)

    descriptors = []
    for face in faces:
        shape = shape_predictor(rgb_image, face)
        descriptor = face_rec_model.compute_face_descriptor(rgb_image, shape)
        descriptors.append(np.array(descriptor))
    
    return descriptors

def update_known_faces():
    global known_face_descriptors, known_face_names
    while True:
        file_paths = list_files_in_dropbox_folder()
        all_descriptors = []
        all_names = []

        for file_path in file_paths:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_file_path = temp_file.name
                download_file_from_dropbox(file_path, temp_file_path)

                descriptors = extract_face_descriptors(temp_file_path)
                name = os.path.splitext(os.path.basename(file_path))[0]

                all_descriptors.extend(descriptors)
                all_names.extend([name] * len(descriptors))

            os.remove(temp_file_path)

        if all_descriptors and all_names:
            # Удаление старых npy файлов
            if known_face_descriptors is not None:
                os.remove("known_face_descriptors.npy")
            if known_face_names is not None:
                os.remove("known_face_names.npy")

            known_face_descriptors = np.array(all_descriptors)
            known_face_names = np.array(all_names)

            # Сохранение новых npy файлов
            np.save("known_face_descriptors.npy", known_face_descriptors)
            np.save("known_face_names.npy", known_face_names)

        # Удаление старых фото из папки (необходимо реализовать)
        try:
            files_in_folder = os.listdir()
            for item in files_in_folder:
                if item.endswith(".jpg"):
                    os.remove(item)
        except Exception as e:
            print(f"Ошибка при удалении старых фотографий: {e}")

        time.sleep(30)


def load_known_faces():
    global known_face_descriptors, known_face_names
    try:
        known_face_descriptors = np.load("known_face_descriptors.npy")
        known_face_names = np.load("known_face_names.npy")
        if known_face_descriptors is None or known_face_names is None:
            raise ValueError("Загруженные данные пусты")
    except (EOFError, ValueError, FileNotFoundError) as e:
        print(f"Ошибка загрузки файлов лиц: {e}")
        known_face_descriptors = None
        known_face_names = None

def process_camera_frame():
    cap = cv2.VideoCapture(0)
    last_recognition_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка при захвате кадра")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        current_time = time.time()
        
        if current_time - last_recognition_time >= 3:
            recognize_faces_and_access(img_rgb, frame)
            last_recognition_time = current_time

        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def recognize_faces_and_access(img_rgb, frame):
    global known_face_descriptors, known_face_names
    faces = detector(img_rgb)
    
    for face in faces:
        shape = shape_predictor(img_rgb, face)
        face_descriptor = face_rec_model.compute_face_descriptor(img_rgb, shape)
        face_descriptor = np.array(face_descriptor)
        
        if known_face_descriptors is not None and len(known_face_descriptors) > 0:
            distances = np.linalg.norm(known_face_descriptors - face_descriptor, axis=1)
            min_distance_idx = np.argmin(distances)
            min_distance = distances[min_distance_idx]
            
            threshold = 0.6
            
            if min_distance <= threshold:
                recognized_name = known_face_names[min_distance_idx]
                print(f"Распознано лицо: {recognized_name} с расстоянием {min_distance}")
            else:
                print("Неизвестное лицо обнаружено. Отправка уведомления...")
                unknown_face_img = extract_unknown_face(frame, face)
                if unknown_face_img is not None and unknown_face_img.size > 0:
                    send_telegram_notification(unknown_face_img)
                else:
                    print("Ошибка: Извлечение неизвестного лица не удалось или изображение пустое")
        else:
            print("Известные лица не загружены или пусты.")

def send_telegram_notification(image):
    if image is None or image.size == 0:
        print("Ошибка: Изображение для отправки в Telegram пустое или невалидно")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        temp_image_path = temp_file.name
        cv2.imwrite(temp_image_path, image)
    
    try:
        with open(temp_image_path, 'rb') as file:
            files = {'photo': file}
            data = {
                'chat_id': TELEGRAM_CHAT_ID,
                'caption': 'Обнаружено неизвестное лицо! Отправляем уведомление.'
            }
            response = requests.post(url, files=files, data=data)
            if response.status_code != 200:
                print(f"Ошибка отправки изображения в Telegram: {response.status_code}, {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при отправке изображения в Telegram: {e}")
    finally:
        delete_temp_image(temp_image_path)

def delete_temp_image(temp_image_path):
    try:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            print("Временный файл удален успешно")
    except Exception as e:
        print(f"Ошибка при удалении временного файла: {e}")


def extract_unknown_face(frame, face):
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    if w > 0 and h > 0:
        face_img = frame[y:y+h, x:x+w]
        return face_img
    else:
        print("Ошибка: Невозможно извлечь лицо, размеры равны нулю")
        return None


if __name__ == "__main__":
    authenticate_dropbox()
    load_known_faces()
    
    gui_thread = threading.Thread(target=main)
    camera_thread = threading.Thread(target=process_camera_frame)
    update_thread = threading.Thread(target=update_known_faces)
    
    gui_thread.start()
    camera_thread.start()
    update_thread.start()
    
    gui_thread.join()
    camera_thread.join()
    update_thread.join()
