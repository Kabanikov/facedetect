import threading
import tkinter as tk
from tkinter import messagebox
import photo_manager
import cv2
import dlib
import numpy as np
import time
import requests
import tempfile
import os
import dropbox

known_face_descriptors = None
known_face_names = None
dbx = None
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
TELEGRAM_BOT_TOKEN = "token here"
TELEGRAM_CHAT_ID = "token here"
DROPBOX_ACCESS_TOKEN = 'token here'

def main():
    root = tk.Tk()
    root.title("Главное меню проекта")
    root.geometry("300x200")
    
    delete_photo_button = tk.Button(root, text="Удалить фото", command=photo_manager.delete_photo)
    delete_photo_button.pack(pady=10)
    
    add_photo_button = tk.Button(root, text="Добавить фото", command=photo_manager.add_photo)
    add_photo_button.pack(pady=10)
    
    root.mainloop()

def authenticate_dropbox():
    global dbx
    dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

def download_file_from_dropbox(dropbox_path, local_path):
    global dbx
    try:
        with open(local_path, "wb") as f:
            metadata, res = dbx.files_download(path=dropbox_path)
            f.write(res.content)
        print(f"Файл {dropbox_path} успешно загружен из Dropbox в {local_path}")
    except Exception as e:
        print(f"Ошибка загрузки файла из Dropbox: {e}")


def update_known_faces():
    while True:
        download_file_from_dropbox("/known_face_descriptors.npy", "known_face_descriptors.npy")
        download_file_from_dropbox("/known_face_names.npy", "known_face_names.npy")
        load_known_faces()
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
        print(f"Формат изображения: {img_rgb.shape}, тип данных: {img_rgb.dtype}")

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
            if unknown_face_img is not None:
                send_telegram_notification(unknown_face_img)
                delete_temp_image(unknown_face_img)
            else:
                print("Ошибка: Извлечение неизвестного лица не удалось")

def extract_unknown_face(frame, face):
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    if w > 0 and h > 0:
        face_img = frame[y:y+h, x:x+w]
        return face_img
    else:
        print("Ошибка: Невозможно извлечь лицо, размеры равны нулю")
        return None

def send_telegram_notification(image):
    if image is None:
        print("Ошибка: Изображение для отправки в Telegram пустое или невалидно")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    
    temp_image_path = tempfile.mktemp(suffix='.jpg')
    cv2.imwrite(temp_image_path, image)
    
    files = {'photo': open(temp_image_path, 'rb')}
    data = {
        'chat_id': TELEGRAM_CHAT_ID,
        'caption': 'Обнаружено неизвестное лицо! Отправляем уведомление.'
    }
    
    try:
        response = requests.post(url, files=files, data=data)
        if response.status_code != 200:
            print(f"Ошибка отправки изображения в Telegram: {response.status_code}, {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при отправке изображения в Telegram: {e}")

def delete_temp_image(temp_image_path):
    try:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            print("Временный файл удален успешно")
    except Exception as e:
        print(f"Ошибка при удалении временного файла: {e}")

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
