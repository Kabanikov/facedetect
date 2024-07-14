import os
import tkinter as tk
from tkinter import filedialog, messagebox

def delete_photo():
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select a photo to delete",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")]
    )
    
    if file_path:
        try:
            os.remove(file_path)
            messagebox.showinfo("Success", f"File {file_path} successfully deleted")
        except Exception as e:
            messagebox.showerror("Error", f"Could not delete file {file_path}. Error: {e}")
    else:
        messagebox.showwarning("Warning", "No file selected")
    
    root.destroy()

def add_photo():
    root = tk.Tk()
    root.withdraw()
    
    source_file_path = filedialog.askopenfilename(
        title="Select a photo to add",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")]
    )
    
    if source_file_path:
        dest_dir = filedialog.askdirectory(
            title="Select a folder to save the photo"
        )
        
        if dest_dir:
            dest_file_path = os.path.join(dest_dir, os.path.basename(source_file_path))
            try:
                with open(source_file_path, 'rb') as src_file:
                    with open(dest_file_path, 'wb') as dest_file:
                        dest_file.write(src_file.read())
                messagebox.showinfo("Success", f"File {os.path.basename(source_file_path)} successfully added to {dest_dir}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not add file {os.path.basename(source_file_path)}. Error: {e}")
        else:
            messagebox.showwarning("Warning", "No folder selected")
    else:
        messagebox.showwarning("Warning", "No file selected")
    
    root.destroy()
