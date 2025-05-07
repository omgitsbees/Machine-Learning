import os
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSpeechSeq2Seq, AutoProcessor
from PIL import Image
import pytesseract # Requires Tesseract OCR to be installed
from sklearn.cluster import KMeans
import cv2 # OpenCV
import logging # Added for logging
# For PDF processing: pip install PyPDF2
# For Speech-to-Text: pip install transformers[audio] librosa soundfile torch torchaudio
# Ensure ffmpeg is installed on your system for broader audio format support with Speech-to-Text: sudo apt-get install ffmpeg (or equivalent for your OS)
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
    logging.warning("PyPDF2 not installed. PDF processing will not be available. Install with: pip install PyPDF2")

try:
    import librosa # For speech_to_text audio loading
except ImportError:
    librosa = None
    logging.warning("librosa not installed. Advanced audio loading for speech_to_text might be limited. Install with: pip install librosa")


# --- Configuration ---
MODEL_CONFIG = {
    "conversational": "microsoft/DialoGPT-medium",
    "object_detection": "facebook/detr-resnet-50",
    "translation_en_fr": "Helsinki-NLP/opus-mt-en-fr",
    "summarization": "facebook/bart-large-cnn",
    "sentiment_analysis": "distilbert-base-uncased-finetuned-sst-2-english", # Default pipeline model often good
    "speech_to_text": "openai/whisper-tiny" # Or "openai/whisper-base", "openai/whisper-small", etc.
}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Core Functions ---

# Step 1: Conversational AI Agents
def conversational_ai_agent(user_input, model_name=MODEL_CONFIG["conversational"]):
    """
    Generates a response using a conversational AI model.
    """
    logging.info(f"Initializing conversational AI with model: {model_name}")
    try:
        # Load a pre-trained conversational model
        chatbot = pipeline("conversational", model=model_name)
        
        # Generate a response (Hugging Face conversation pipeline expects a Conversation object or list of inputs)
        # For simplicity, we'll handle a single turn. For multi-turn, you'd manage conversation history.
        from transformers import Conversation
        conversation = Conversation(user_input)
        response = chatbot(conversation)
        # The response object might be a Conversation object itself
        return response.generated_responses[-1]
    except Exception as e:
        logging.error(f"Error in conversational AI agent: {e}")
        return f"Error generating response: {e}"

# Step 2: Processing Documents
def process_documents(file_path):
    """
    Extracts text from image-based documents (PNG, JPG) using OCR
    and from PDF documents.
    """
    logging.info(f"Processing document: {file_path}")
    if not os.path.exists(file_path):
        logging.error(f"Document file not found: {file_path}")
        return "Error: Document file not found."

    _, file_extension = os.path.splitext(file_path.lower())
    text = ""

    try:
        if file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            logging.info(f"Extracting text from image: {file_path} using Tesseract OCR.")
            text = image_to_string(Image.open(file_path))
        elif file_extension == '.pdf':
            if PyPDF2:
                logging.info(f"Extracting text from PDF: {file_path} using PyPDF2.")
                with open(file_path, 'rb') as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    for page_num in range(len(reader.pages)):
                        text += reader.pages[page_num].extract_text() + "\n"
            else:
                logging.warning("PyPDF2 is not installed. Cannot process PDF file.")
                return "Error: PDF processing library (PyPDF2) not available."
        elif file_extension == '.txt':
            logging.info(f"Reading text from TXT file: {file_path}.")
            with open(file_path, 'r', encoding='utf-8') as txt_file:
                text = txt_file.read()
        else:
            return f"Error: Unsupported document format '{file_extension}'."
        
        logging.info(f"Successfully extracted text from {file_path}.")
        return text.strip()
    except Exception as e:
        logging.error(f"Error processing document {file_path}: {e}")
        return f"Error processing document: {e}"

# Step 3: Analyzing Images and Videos
def analyze_images(image_path, model_name=MODEL_CONFIG["object_detection"]):
    """
    Performs object detection on an image using a pre-trained model.
    """
    logging.info(f"Analyzing image: {image_path} with model: {model_name}")
    if not os.path.exists(image_path):
        logging.error(f"Image file not found: {image_path}")
        return "Error: Image file not found."
        
    try:
        object_detection = pipeline("object-detection", model=model_name)
        image = Image.open(image_path)
        results = object_detection(image)
        logging.info(f"Image analysis successful for {image_path}.")
        return results
    except Exception as e:
        logging.error(f"Error analyzing image {image_path}: {e}")
        return f"Error analyzing image: {e}"

def analyze_videos(video_path, num_clusters=5):
    """
    Extracts frames from a video and performs K-Means clustering for basic scene detection.
    Returns cluster labels for frames.
    """
    logging.info(f"Analyzing video: {video_path} for scene clustering.")
    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        return "Error: Video file not found."

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Could not open video file: {video_path}")
            return "Error: Could not open video file."

        frames = []
        frame_count = 0
        max_frames = 100 # Limit number of frames to process for performance
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= max_frames:
                break
            # Resize frame for faster processing
            resized_frame = cv2.resize(frame, (128, 72)) # Example resize
            frames.append(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY))
            frame_count += 1
        cap.release()

        if not frames:
            logging.warning(f"No frames extracted from video: {video_path}")
            return "Error: No frames extracted from video."

        logging.info(f"Extracted {len(frames)} frames. Performing clustering...")
        # Perform clustering on frames (flattened pixel values)
        frames_flat = [frame.flatten() for frame in frames]
        
        # Ensure n_clusters is not more than the number of samples
        actual_num_clusters = min(num_clusters, len(frames_flat))
        if actual_num_clusters < 1:
             logging.warning(f"Not enough frames to cluster for video: {video_path}")
             return "Error: Not enough frames to cluster."
        if actual_num_clusters < num_clusters:
            logging.warning(f"Number of clusters reduced to {actual_num_clusters} due to limited frames.")


        kmeans = KMeans(n_clusters=actual_num_clusters, random_state=42, n_init='auto').fit(frames_flat)
        logging.info(f"Video scene clustering successful for {video_path}.")
        return kmeans.labels_
    except Exception as e:
        logging.error(f"Error analyzing video {video_path}: {e}")
        return f"Error analyzing video: {e}"

# Step 4: Deploying Pre-Trained / Open-Source Models
def deploy_pretrained_model(model_name, task, input_data, tokenizer_name=None):
    """
    Uses HuggingFace pipelines for various tasks with specified models.
    Can also load tokenizer and model explicitly if tokenizer_name is provided.
    """
    logging.info(f"Deploying pre-trained model: {model_name} for task: {task}")
    try:
        if tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            # Determine the correct AutoModel class based on the task
            # This is a simplified example; a more robust solution would map tasks to model classes
            if "Seq2SeqLM" in type(AutoModelForSeq2SeqLM.from_pretrained(model_name)).__name__ or task == "translation":
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            else: # Add more conditions for other model types if needed
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name) # Fallback, adjust as needed
            model_pipeline = pipeline(task, model=model, tokenizer=tokenizer)
        else:
            model_pipeline = pipeline(task, model=model_name)
        
        result = model_pipeline(input_data)
        logging.info(f"Pre-trained model execution successful for task: {task}.")
        return result
    except Exception as e:
        logging.error(f"Error deploying pre-trained model {model_name} for task {task}: {e}")
        return f"Error deploying model: {e}"

def sentiment_analysis(text, model_name=MODEL_CONFIG["sentiment_analysis"]):
    """
    Performs sentiment analysis on text using a pre-trained model.
    """
    logging.info(f"Performing sentiment analysis with model: {model_name}")
    try:
        sentiment_model = pipeline("sentiment-analysis", model=model_name)
        result = sentiment_model(text)
        logging.info("Sentiment analysis successful.")
        return result
    except Exception as e:
        logging.error(f"Error in sentiment analysis: {e}")
        return f"Error performing sentiment analysis: {e}"

def translate_text(text, target_language="fr", model_name=MODEL_CONFIG["translation_en_fr"]):
    """
    Translates text to a target language using a pre-trained model.
    Note: The default model translates English to French. Change model for other pairs.
    """
    logging.info(f"Translating text to {target_language} with model: {model_name}")
    try:
        # The pipeline task for translation is "translation_xx_to_yy"
        # For dynamic target languages with a multi-lingual model, this setup would need adjustment.
        # The Helsinki-NLP models are typically bilingual.
        # Example: model_name = "Helsinki-NLP/opus-mt-en-de" for English to German
        
        # If model_name is fixed like "Helsinki-NLP/opus-mt-en-fr", target_language in pipeline() can be used
        # but it's often implicitly defined by the model choice.
        # For robust multi-language support, you might need a map of language pairs to models.
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        translator = pipeline("translation", model=model, tokenizer=tokenizer) # task can be more specific e.g. "translation_en_to_fr"

        # The pipeline's translate method might take `tgt_lang` or `locale`
        # Check documentation for the specific model/pipeline usage
        # For Helsinki-NLP, often the model name itself dictates source/target.
        # The pipeline call `translator(text)` is usually sufficient if model is bilingual.
        # If the model supports multiple targets, you might use `translator(text, tgt_lang=target_language)`
        
        # The following tries to be generic, but specific model behavior varies
        try:
            translation = translator(text, target_lang=target_language) # For models supporting target_lang
        except TypeError: # Some models might not take target_lang directly in pipeline call
            translation = translator(text)


        logging.info("Text translation successful.")
        return translation[0]["translation_text"]
    except Exception as e:
        logging.error(f"Error translating text: {e}")
        return f"Error translating text: {e}"

def summarize_text(text, model_name=MODEL_CONFIG["summarization"], max_length=150, min_length=30):
    """
    Summarizes a given text using a pre-trained model.
    """
    logging.info(f"Summarizing text with model: {model_name}")
    try:
        summarizer = pipeline("summarization", model=model_name)
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        logging.info("Text summarization successful.")
        return summary[0]["summary_text"]
    except Exception as e:
        logging.error(f"Error summarizing text: {e}")
        return f"Error summarizing text: {e}"

def speech_to_text(audio_path, model_name=MODEL_CONFIG["speech_to_text"]):
    """
    Transcribes speech from an audio file to text using a pre-trained model.
    Requires librosa and soundfile, and potentially ffmpeg.
    """
    logging.info(f"Performing speech-to-text on: {audio_path} with model: {model_name}")
    if not os.path.exists(audio_path):
        logging.error(f"Audio file not found: {audio_path}")
        return "Error: Audio file not found."
    if not librosa:
        logging.error("Librosa is not installed. Cannot perform speech-to-text.")
        return "Error: Librosa library not available for audio processing."

    try:
        # Load the audio file using librosa for broader format support
        # The ASR pipeline can often handle paths directly, but this ensures consistent loading
        speech_array, sampling_rate = librosa.load(audio_path, sr=16000) # Resample to 16kHz if needed by model

        # Using specific AutoProcessor and AutoModelForSpeechSeq2Seq for more control with Whisper
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)

        # Process the audio
        inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt")
        
        # Generate transcription
        predicted_ids = model.generate(inputs["input_features"])
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        # For pipeline approach (simpler, but might have less control):
        # asr_pipeline = pipeline("automatic-speech-recognition", model=model_name)
        # transcription_result = asr_pipeline(audio_path) # or pass the loaded audio_array
        # transcription = transcription_result["text"]
        
        logging.info(f"Speech-to-text successful for {audio_path}.")
        return transcription[0] if isinstance(transcription, list) else transcription # Whisper might return a list with one item
    except Exception as e:
        logging.error(f"Error in speech-to-text for {audio_path}: {e}")
        return f"Error performing speech-to-text: {e}"

# --- Example Usage ---
if __name__ == "__main__":
    # Create dummy files for testing if they don't exist
    # Note: For OCR (image_to_string), you need Tesseract installed and configured.
    # For video analysis, you need a valid video file.
    # For speech-to-text, you need a valid audio file.

    if not os.path.exists("example_document.png"):
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new('RGB', (400, 100), color = (255, 255, 255))
            d = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
            d.text((10,10), "Hello World from Pillow image.", fill=(0,0,0), font=font)
            img.save("example_document.png")
            logging.info("Created dummy example_document.png")
        except Exception as e:
            logging.warning(f"Could not create dummy PNG: {e}. Please provide your own 'example_document.png'.")

    if not os.path.exists("example_image.jpg"):
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (200, 150), color = (73, 109, 137))
            d = ImageDraw.Draw(img)
            d.rectangle([(20,20), (80,80)], fill=(255,0,0)) # Add a red square
            d.ellipse([(100,50), (180,130)], fill=(0,255,0)) # Add a green circle
            img.save("example_image.jpg")
            logging.info("Created dummy example_image.jpg")
        except Exception as e:
            logging.warning(f"Could not create dummy JPG: {e}. Please provide your own 'example_image.jpg'.")

    if not os.path.exists("example_document.pdf") and PyPDF2:
        # Creating a simple PDF is complex without external libraries like reportlab
        # For now, we'll skip automatic creation, assuming user provides one if PyPDF2 is available.
        logging.info("Please provide 'example_document.pdf' for PDF processing testing.")
    
    if not os.path.exists("example_text_doc.txt"):
        with open("example_text_doc.txt", "w") as f:
            f.write("This is a sample text document for testing purposes. It contains several sentences.\n")
            f.write("Cloud AI services offer powerful machine learning capabilities.")
        logging.info("Created dummy example_text_doc.txt")

    # Note: Creating dummy video and audio files programmatically is non-trivial.
    # Please provide your own 'example_video.mp4' and 'example_audio.wav' (or other supported format).
    logging.info("Please provide 'example_video.mp4' for video analysis.")
    logging.info("Please provide 'example_audio.wav' (or other supported audio format) for speech-to-text.")


    # 1. Conversational AI
    user_input_convo = "Hello, how can you help me today?"
    logging.info(f"\n--- Testing Conversational AI ---")
    print("User Input:", user_input_convo)
    print("Chatbot Response:", conversational_ai_agent(user_input_convo))
    
    # 2. Document Processing (Image)
    document_path_png = "example_document.png"
    logging.info(f"\n--- Testing Document Processing (Image: {document_path_png}) ---")
    if os.path.exists(document_path_png):
        print("\nExtracted Text from Image Document:", process_documents(document_path_png))
    else:
        print(f"\nSkipping image document processing: {document_path_png} not found.")
        
    # Document Processing (PDF) - only if PyPDF2 is installed and file exists
    document_path_pdf = "example_document.pdf" # Replace with your PDF path
    logging.info(f"\n--- Testing Document Processing (PDF: {document_path_pdf}) ---")
    if PyPDF2 and os.path.exists(document_path_pdf):
        print("\nExtracted Text from PDF Document:", process_documents(document_path_pdf))
    elif not PyPDF2:
        print("\nSkipping PDF document processing: PyPDF2 library not installed.")
    else:
        print(f"\nSkipping PDF document processing: {document_path_pdf} not found.")

    # Document Processing (TXT)
    document_path_txt = "example_text_doc.txt"
    logging.info(f"\n--- Testing Document Processing (TXT: {document_path_txt}) ---")
    if os.path.exists(document_path_txt):
        extracted_text_from_txt = process_documents(document_path_txt)
        print("\nExtracted Text from TXT Document:", extracted_text_from_txt)
        
        # 5. Text Summarization (using text from the TXT document)
        logging.info(f"\n--- Testing Text Summarization ---")
        if extracted_text_from_txt and not extracted_text_from_txt.startswith("Error"):
            print("\nOriginal Text for Summarization:\n", extracted_text_from_txt)
            print("\nSummary:", summarize_text(extracted_text_from_txt))
        else:
            print("\nSkipping summarization due to issues in text extraction or empty text.")
    else:
        print(f"\nSkipping TXT document processing and subsequent summarization: {document_path_txt} not found.")

    # 3. Image Analysis
    image_path = "example_image.jpg" # Replace with your image path
    logging.info(f"\n--- Testing Image Analysis ({image_path}) ---")
    if os.path.exists(image_path):
        print("\nImage Analysis Results:", analyze_images(image_path))
    else:
        print(f"\nSkipping image analysis: {image_path} not found.")
        
    # 4. Video Analysis
    video_path = "example_video.mp4" # Replace with your video path
    logging.info(f"\n--- Testing Video Analysis ({video_path}) ---")
    if os.path.exists(video_path):
        print("\nVideo Scene Clustering Labels:", analyze_videos(video_path))
    else:
        print(f"\nSkipping video analysis: {video_path} not found. Please provide a sample video.")
        
    # 6. Deploy Pre-Trained Model (Example: Translation, again, but via generic function)
    input_text_deploy = "This is a test of the generic model deployment function."
    logging.info(f"\n--- Testing Generic Pre-Trained Model Deployment (Translation) ---")
    # Using the translation model from config
    print("\nTranslation Result (via deploy_pretrained_model):",
          deploy_pretrained_model(MODEL_CONFIG["translation_en_fr"], "translation", input_text_deploy))
          
    # 7. Sentiment Analysis
    sentiment_text = "Cloud AI services are incredibly powerful and versatile!"
    logging.info(f"\n--- Testing Sentiment Analysis ---")
    print("\nSentiment Analysis Input:", sentiment_text)
    print("Sentiment Analysis Result:", sentiment_analysis(sentiment_text))
    
    # 8. Text Translation (using the dedicated function)
    text_to_translate = "Hello, world! How are you doing today?"
    logging.info(f"\n--- Testing Dedicated Text Translation ---")
    print("\nText to Translate:", text_to_translate)
    print("Translated Text (to French):", translate_text(text_to_translate, target_language="fr"))

    # 9. Speech-to-Text
    audio_file_path = "example_audio.wav"  # Replace with your audio file path (e.g., WAV, MP3, FLAC)
    logging.info(f"\n--- Testing Speech-to-Text ({audio_file_path}) ---")
    if os.path.exists(audio_file_path):
        print("\nSpeech-to-Text Transcription:", speech_to_text(audio_file_path))
    else:
        print(f"\nSkipping speech-to-text: {audio_file_path} not found. Please provide a sample audio file.")

    logging.info("\n--- All Tests Concluded ---")