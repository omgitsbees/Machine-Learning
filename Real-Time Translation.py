import speech_recognition as sr
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS
import os
from playsound import playsound
import tkinter as tk
from tkinter import ttk, messagebox
from langdetect import detect
import json

# Step 1: Initialize Translation Model
def load_translation_model(src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Step 2: Speech-to-Text
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            print(f"Recognized Speech: {text}")
            return text
        except sr.UnknownValueError:
            messagebox.showerror("Error", "Could not understand the audio.")
        except sr.RequestError:
            messagebox.showerror("Error", "Speech recognition service is unavailable.")
        return None

# Step 3: Language Detection
def detect_language(text):
    try:
        detected_lang = detect(text)
        print(f"Detected Language: {detected_lang}")
        return detected_lang
    except Exception as e:
        messagebox.showerror("Error", f"Language detection failed: {e}")
        return None

# Step 4: Translate Text
def translate_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    print(f"Translated Text: {translated_text}")
    return translated_text

# Step 5: Text-to-Speech with Speed Control
def text_to_speech(text, lang, speed):
    tts = gTTS(text=text, lang=lang, slow=(speed == "Slow"))
    tts.save("translated_audio.mp3")
    playsound("translated_audio.mp3")
    os.remove("translated_audio.mp3")

# Step 6: Save Translation History
def save_translation_history(src_text, translated_text, src_lang, tgt_lang):
    history = {
        "Source Text": src_text,
        "Translated Text": translated_text,
        "Source Language": src_lang,
        "Target Language": tgt_lang
    }
    with open("translation_history.json", "a") as file:
        file.write(json.dumps(history) + "\n")
    print("Translation history saved.")

# Step 7: Real-Time Translation Workflow
def real_time_translation():
    try:
        # Get user input for source and target languages
        src_lang = src_lang_dropdown.get()
        tgt_lang = tgt_lang_dropdown.get()
        speed = speed_dropdown.get()
        if not tgt_lang:
            messagebox.showerror("Error", "Please select a target language.")
            return

        # Perform speech-to-text
        text = speech_to_text()
        if not text:
            return

        # Detect source language if not provided
        if src_lang == "Auto":
            src_lang = detect_language(text)
            if not src_lang:
                return

        # Load translation model
        tokenizer, model = load_translation_model(src_lang, tgt_lang)

        # Perform translation
        translated_text = translate_text(text, tokenizer, model)

        # Perform text-to-speech
        text_to_speech(translated_text, tgt_lang, speed)

        # Save translation history
        save_translation_history(text, translated_text, src_lang, tgt_lang)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Step 8: Build User Interface
root = tk.Tk()
root.title("Real-Time Translation")

# Language Dropdowns
tk.Label(root, text="Source Language:").grid(row=0, column=0, padx=10, pady=10)
src_lang_dropdown = ttk.Combobox(root, values=["Auto", "en", "es", "fr", "de", "it"])
src_lang_dropdown.grid(row=0, column=1, padx=10, pady=10)
src_lang_dropdown.set("Auto")

tk.Label(root, text="Target Language:").grid(row=1, column=0, padx=10, pady=10)
tgt_lang_dropdown = ttk.Combobox(root, values=["en", "es", "fr", "de", "it"])
tgt_lang_dropdown.grid(row=1, column=1, padx=10, pady=10)
tgt_lang_dropdown.set("es")

# Speed Dropdown
tk.Label(root, text="Speech Speed:").grid(row=2, column=0, padx=10, pady=10)
speed_dropdown = ttk.Combobox(root, values=["Normal", "Slow"])
speed_dropdown.grid(row=2, column=1, padx=10, pady=10)
speed_dropdown.set("Normal")

# Translate Button
translate_button = tk.Button(root, text="Translate", command=real_time_translation)
translate_button.grid(row=3, column=0, columnspan=2, pady=10)

root.mainloop()