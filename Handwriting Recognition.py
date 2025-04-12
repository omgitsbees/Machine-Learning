import cv2
import pytesseract
from pytesseract import Output
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
import re
from difflib import get_close_matches

# Configure Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path if needed

# Custom Vocabulary for Error Correction
custom_vocabulary = ["machine", "learning", "Python", "TensorFlow", "OpenCV", "recognition", "handwriting"]

# Step 1: Preprocess Image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

# Step 2: Perform OCR
def perform_ocr(image, language="eng"):
    try:
        text = pytesseract.image_to_string(image, lang=language)
        return text
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during OCR: {e}")
        return None

# Step 3: Error Correction
def correct_errors(text):
    words = text.split()
    corrected_words = []
    for word in words:
        # Remove non-alphanumeric characters
        clean_word = re.sub(r'\W+', '', word)
        # Find closest match in custom vocabulary
        if clean_word and clean_word.lower() not in custom_vocabulary:
            matches = get_close_matches(clean_word.lower(), custom_vocabulary, n=1, cutoff=0.8)
            corrected_words.append(matches[0] if matches else word)
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)

# Step 4: Real-Time Recognition
def real_time_recognition(language="eng"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to access the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh_frame = cv2.threshold(gray_frame, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Perform OCR on the frame
        text = pytesseract.image_to_string(thresh_frame, lang=language)
        corrected_text = correct_errors(text)

        # Display the recognized text on the frame
        cv2.putText(frame, corrected_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Real-Time Handwriting Recognition", frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Step 5: Save Recognized Text
def save_text_to_file(text):
    save_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                             filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
    if save_path:
        with open(save_path, "w", encoding="utf-8") as file:
            file.write(text)
        messagebox.showinfo("Success", f"Text saved to {save_path}")

# Step 6: Load Image and Recognize Handwriting
def load_and_recognize():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg"), ("All Files", "*.*")])
    if not file_path:
        return

    language = language_dropdown.get()
    preprocessed_image = preprocess_image(file_path)
    recognized_text = perform_ocr(preprocessed_image, language)
    if recognized_text:
        corrected_text = correct_errors(recognized_text)
        text_output.delete(1.0, tk.END)
        text_output.insert(tk.END, corrected_text)

# Step 7: Build User Interface
root = tk.Tk()
root.title("Handwriting Recognition")

# Language Selection
tk.Label(root, text="Select Language:").grid(row=0, column=0, padx=10, pady=10)
language_dropdown = ttk.Combobox(root, values=["eng", "spa", "fra", "deu", "ita"])
language_dropdown.grid(row=0, column=1, padx=10, pady=10)
language_dropdown.set("eng")

# Buttons
load_button = tk.Button(root, text="Load Image", command=load_and_recognize)
load_button.grid(row=1, column=0, padx=10, pady=10)

real_time_button = tk.Button(root, text="Real-Time Recognition", command=lambda: real_time_recognition(language_dropdown.get()))
real_time_button.grid(row=1, column=1, padx=10, pady=10)

save_button = tk.Button(root, text="Save Text", command=lambda: save_text_to_file(text_output.get(1.0, tk.END)))
save_button.grid(row=2, column=0, columnspan=2, pady=10)

# Text Output
text_output = tk.Text(root, wrap=tk.WORD, width=50, height=15)
text_output.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()