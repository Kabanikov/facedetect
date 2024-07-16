import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
import dropbox
import os

DROPBOX_ACCESS_TOKEN = 'your token'
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

def list_files_in_dropbox(dbx):
    try:
        files = dbx.files_list_folder('').entries
        file_paths = [file.path_lower for file in files if isinstance(file, dropbox.files.FileMetadata)]
        return file_paths
    except dropbox.exceptions.AuthError as e:
        print(f"Ошибка аутентификации Dropbox: {e}")
        return []
    except Exception as e:
        print(f"Ошибка при получении списка файлов из Dropbox: {e}")
        return []

def add_photo(dbx):
    try:
        source_file_path = filedialog.askopenfilename(
            title="Select a photo to add",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")]
        )
        
        if source_file_path:
            dest_file_name = os.path.basename(source_file_path)
            
            # Путь на Dropbox, куда загружаем файл
            dest_dropbox_path = '/' + dest_file_name
            
            with open(source_file_path, 'rb') as f:
                # Загружаем файл на Dropbox
                dbx.files_upload(f.read(), dest_dropbox_path)
            
            messagebox.showinfo("Success", f"File {dest_file_name} successfully added to Dropbox")
        else:
            messagebox.showwarning("Warning", "No file selected")
    
    except Exception as e:
        messagebox.showerror("Error", f"Could not add file to Dropbox. Error: {e}")

def delete_photo(dbx):
    try:
        file_paths = list_files_in_dropbox(dbx)
        if not file_paths:
            messagebox.showinfo("Info", "No files found in Dropbox.")
            return
        
        # Показываем пользователю пронумерованный список файлов
        message = "Select a file number to delete:\n"
        for idx, file_path in enumerate(file_paths):
            message += f"{idx + 1}. {file_path}\n"
        
        selected_index = simpledialog.askinteger("Select File", message)
        if selected_index is None or selected_index < 1 or selected_index > len(file_paths):
            messagebox.showwarning("Warning", "Invalid selection.")
            return
        
        # Выбранный файл для удаления
        file_path_to_delete = file_paths[selected_index - 1]
        
        # Удаляем файл с Dropbox
        dbx.files_delete_v2(file_path_to_delete)
        
        messagebox.showinfo("Success", f"File {file_path_to_delete} successfully deleted from Dropbox")
    
    except Exception as e:
        messagebox.showerror("Error", f"Could not delete file from Dropbox. Error: {e}")

def main():
    root = tk.Tk()
    root.title("Главное меню проекта")
    root.geometry("300x200")
    
    add_photo_button = tk.Button(root, text="Добавить фото", command=lambda: add_photo(dbx))
    add_photo_button.pack(pady=10)
    
    delete_photo_button = tk.Button(root, text="Удалить фото", command=lambda: delete_photo(dbx))
    delete_photo_button.pack(pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    main()
