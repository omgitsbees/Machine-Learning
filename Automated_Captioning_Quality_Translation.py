import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import jiwer
from textstat import flesch_reading_ease
import language_tool_python
import logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)

# --- ASR: Automatic Speech Recognition ---
def transcribe_audio(audio_path, language="en"):
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=0 if torch.cuda.is_available() else -1)
    result = asr(audio_path, return_timestamps=False)
    transcript = result['text']
    confidence = result.get('score', 1.0)
    logging.info(f"ASR Transcript: {transcript}")
    return transcript, confidence

# --- Language Detection (Optional) ---
def detect_language(text):
    from langdetect import detect
    lang = detect(text)
    logging.info(f"Detected language: {lang}")
    return lang

# --- Grammar & Spelling Check ---
def grammar_check(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    num_errors = len(matches)
    logging.info(f"Grammar/Spelling errors: {num_errors}")
    return num_errors, matches

# --- Quality Assessment ---
def assess_caption_quality(transcript, reference=None, confidence=1.0):
    readability = flesch_reading_ease(transcript)
    wer = jiwer.wer(reference, transcript) if reference else None
    num_errors, _ = grammar_check(transcript)
    quality_score = 0.4 * (readability / 100) + 0.4 * confidence + 0.2 * (1 - num_errors / max(len(transcript.split()), 1))
    return {
        "readability": readability,
        "wer": wer,
        "confidence": confidence,
        "grammar_errors": num_errors,
        "quality_score": round(quality_score, 3)
    }

# --- NMT: Neural Machine Translation ---
def translate_caption(transcript, target_langs=["fr"]):
    translations = {}
    for lang in target_langs:
        model_name = f"Helsinki-NLP/opus-mt-en-{lang}"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        inputs = tokenizer(transcript, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs)
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translations[lang] = translated
        logging.info(f"Translation ({lang}): {translated}")
    return translations

# --- BLEU Score for Translation Quality ---
def compute_bleu(reference, hypothesis):
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.translate.bleu_score import sentence_bleu
    ref_tokens = nltk.word_tokenize(reference)
    hyp_tokens = nltk.word_tokenize(hypothesis)
    bleu = sentence_bleu([ref_tokens], hyp_tokens)
    return bleu

# --- Example Usage ---
if __name__ == "__main__":
    audio_path = "sample_audio.wav"  # Path to your audio file

    # 1. ASR: Transcribe audio
    transcript, confidence = transcribe_audio(audio_path)

    # 2. Language Detection (optional)
    detected_lang = detect_language(transcript)

    # 3. Quality Assessment (optional: provide reference for WER)
    reference_caption = "This is the reference caption for quality assessment."
    quality = assess_caption_quality(transcript, reference=reference_caption, confidence=confidence)
    print("Quality Assessment:", quality)

    # 4. NMT: Translate to multiple languages
    translations = translate_caption(transcript, target_langs=["fr", "es", "de"])
    print("Translations:", translations)

    # 5. BLEU Score for Translation (if reference available)
    if "fr" in translations:
        reference_fr = "Ceci est la légende de référence pour l'évaluation de la qualité."
        bleu_fr = compute_bleu(reference_fr, translations["fr"])
        print(f"French BLEU Score: {bleu_fr:.2f}")