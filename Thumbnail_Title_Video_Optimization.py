import pandas as pd
import numpy as np
import re # For regular expressions (e.g., finding all caps words)
import requests # For downloading thumbnails
from PIL import Image, ImageStat # For basic image stats like brightness
import io # To handle image data in memory
import os # For creating directories
import time # To add delays if needed

# --- Configuration ---
THUMBNAIL_DIR = "thumbnails" # Directory to save downloaded thumbnails
# Create the directory if it doesn't exist
if not os.path.exists(THUMBNAIL_DIR):
    os.makedirs(THUMBNAIL_DIR)

# --- 1. Data Loading ---
def load_video_data(csv_path="youtube_video_data.csv"):
    """Loads video data from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {csv_path} with {len(df)} rows.")
        # Ensure essential columns are present
        required_cols = ['video_id', 'title', 'thumbnail_url', 'impressions', 'clicks']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: CSV is missing one or more required columns: {missing_cols}")
            # You might want to return None or raise an error if critical columns are missing
            # For now, we'll proceed but CTR calculation might fail.

        # Calculate CTR if 'impressions' and 'clicks' columns exist
        if 'impressions' in df.columns and 'clicks' in df.columns:
            # Ensure impressions are not zero to avoid division by zero error
            # Replace 0 impressions with NaN temporarily for division, then fillna for CTR
            df['ctr'] = (df['clicks'] / df['impressions'].replace(0, np.nan)).fillna(0)
        elif 'ctr' not in df.columns: # If CTR is not provided and cannot be calculated
            print("Warning: 'impressions' or 'clicks' column missing, and 'ctr' not present. CTR will be NaN.")
            df['ctr'] = np.nan
        return df
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

# --- 2. Title Feature Extraction ---
def get_title_length_chars(title):
    """Calculates the number of characters in a title."""
    return len(str(title)) if pd.notna(title) else 0

def get_title_length_words(title):
    """Calculates the number of words in a title."""
    return len(str(title).split()) if pd.notna(title) else 0

def has_number(text):
    """Checks if the text contains any number."""
    return any(char.isdigit() for char in str(text)) if pd.notna(text) else False

def count_all_caps_words(title):
    """Counts the number of words in ALL CAPS (more than 1 letter long)."""
    if pd.isna(title):
        return 0
    words = str(title).split()
    return sum(1 for word in words if word.isupper() and len(word) > 1)

def has_question_mark(title):
    """Checks if the title contains a question mark."""
    return "?" in str(title) if pd.notna(title) else False

def has_exclamation_mark(title):
    """Checks if the title contains an exclamation mark."""
    return "!" in str(title) if pd.notna(title) else False

def extract_title_features(df):
    """Applies all title feature extraction functions to the DataFrame."""
    if 'title' not in df.columns:
        print("Error: 'title' column not found in DataFrame. Skipping title feature extraction.")
        return df

    print("Extracting title features...")
    df['title_length_chars'] = df['title'].apply(get_title_length_chars)
    df['title_length_words'] = df['title'].apply(get_title_length_words)
    df['title_has_number'] = df['title'].apply(has_number)
    df['title_all_caps_words_count'] = df['title'].apply(count_all_caps_words)
    df['title_has_question_mark'] = df['title'].apply(has_question_mark)
    df['title_has_exclamation_mark'] = df['title'].apply(has_exclamation_mark)
    print("Title features extracted.")
    return df

# --- 3. Thumbnail Feature Extraction (Initial Setup & Download) ---

def download_thumbnail(video_id, url, save_dir=THUMBNAIL_DIR):
    """Downloads a thumbnail image from a URL and saves it."""
    if not url or pd.isna(url) or str(url).strip() == "":
        return None
    try:
        # Ensure video_id is a string suitable for filenames
        video_id_str = str(video_id).replace('/', '_').replace('\\', '_')

        # Basic file extension guessing from URL, default to .jpg
        # Split by '?' to remove query parameters before getting extension
        path_part = url.split('?')[0]
        file_extension = os.path.splitext(path_part)[-1].lower()
        if not file_extension or file_extension not in ['.jpg', '.jpeg', '.png', '.webp']:
            file_extension = '.jpg' # Default extension

        filename = os.path.join(save_dir, f"{video_id_str}{file_extension}")

        if os.path.exists(filename):
            return filename

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, stream=True, headers=headers, timeout=15) # Increased timeout
        response.raise_for_status()

        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        time.sleep(0.1) # Small delay to be polite to servers
        return filename
    except requests.exceptions.MissingSchema:
        print(f"Error downloading for video_id {video_id}: Invalid URL (Missing schema) '{url}'")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error downloading thumbnail for video_id {video_id} from {url}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while downloading thumbnail for video_id {video_id} ({url}): {e}")
        return None

def get_image_brightness(image_path):
    """Calculates the average brightness of an image."""
    if image_path is None or not os.path.exists(image_path):
        return np.nan
    try:
        img = Image.open(image_path).convert('L') # Convert to grayscale
        stat = ImageStat.Stat(img)
        return stat.mean[0] # Mean pixel value (0-255)
    except Exception as e:
        print(f"Error calculating brightness for {image_path}: {e}")
        return np.nan

def get_image_saturation(image_path):
    """Calculates the average saturation of an image using HSV color space."""
    if image_path is None or not os.path.exists(image_path):
        return np.nan
    try:
        img = Image.open(image_path).convert('HSV')
        stat = ImageStat.Stat(img)
        # Saturation is the second channel (index 1) in HSV mode for Pillow
        return stat.mean[1] # Mean saturation value (0-255 for Pillow's HSV)
    except Exception as e:
        print(f"Error calculating saturation for {image_path}: {e}")
        return np.nan

# --- Placeholder for more advanced thumbnail features ---
def has_face_in_thumbnail(image_path):
    """
    Placeholder: Detects faces in a thumbnail.
    Requires a library like OpenCV and a pre-trained model.
    """
    if image_path is None or not os.path.exists(image_path):
        return np.nan # Using np.nan for not-yet-implemented or failed features
    # TODO: Implement face detection using opencv-python or face_recognition
    # Example (requires 'pip install opencv-python'):
    # import cv2
    # try:
    #     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    #     if not os.path.exists(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'):
    #         print("Error: Haar cascade file not found. Face detection will not work.")
    #         return np.nan
    #     img_cv = cv2.imread(image_path)
    #     if img_cv is None: return np.nan
    #     gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    #     return len(faces) > 0
    # except Exception as e:
    #     print(f"Error in face detection for {image_path}: {e}")
    #     return np.nan
    return np.nan # Placeholder until fully implemented

def has_text_in_thumbnail_ocr(image_path):
    """
    Placeholder: Detects text in a thumbnail using OCR.
    Requires a library like pytesseract and Tesseract OCR engine.
    """
    if image_path is None or not os.path.exists(image_path):
        return np.nan
    # TODO: Implement OCR using pytesseract
    # Example (requires 'pip install pytesseract' and Tesseract OCR engine installed on system):
    # import pytesseract
    # try:
    #   text = pytesseract.image_to_string(Image.open(image_path), timeout=5) # add timeout
    #   return len(text.strip()) > 0
    # except Exception as e:
    #   print(f"Error during OCR for {image_path}: {e}")
    #   return False # Or np.nan if you prefer
    return np.nan # Placeholder until fully implemented

def extract_thumbnail_features_for_row(df_row):
    """
    Downloads thumbnail (if not already) and extracts features for a single row.
    Returns a pd.Series of thumbnail features.
    """
    video_id = df_row.get('video_id', 'unknown_id') # Use .get for safety
    thumbnail_url = df_row.get('thumbnail_url')

    local_thumbnail_path = download_thumbnail(video_id, thumbnail_url)

    features = {
        'local_thumbnail_path': local_thumbnail_path,
        'thumbnail_avg_brightness': np.nan,
        'thumbnail_avg_saturation': np.nan,
        'thumbnail_has_face': np.nan,
        'thumbnail_text_detected_ocr': np.nan
    }

    if local_thumbnail_path and os.path.exists(local_thumbnail_path):
        features['thumbnail_avg_brightness'] = get_image_brightness(local_thumbnail_path)
        features['thumbnail_avg_saturation'] = get_image_saturation(local_thumbnail_path)
        features['thumbnail_has_face'] = has_face_in_thumbnail(local_thumbnail_path) # Uncomment when implemented
        features['thumbnail_text_detected_ocr'] = has_text_in_thumbnail_ocr(local_thumbnail_path) # Uncomment when implemented
    else:
        # print(f"Skipping feature extraction for video_id {video_id} as thumbnail path is invalid.")
        pass

    return pd.Series(features)


# --- 4. Main Processing Pipeline ---
def process_data(csv_path="youtube_video_data.csv"):
    """Main function to load, preprocess, and extract features."""
    df = load_video_data(csv_path)
    if df is None:
        print("Data loading failed. Exiting.")
        return None

    # Extract title features
    df = extract_title_features(df)

    # Extract thumbnail features
    if 'thumbnail_url' not in df.columns or 'video_id' not in df.columns:
        print("Error: 'thumbnail_url' or 'video_id' column not found. Skipping thumbnail feature extraction.")
    else:
        print("\nStarting thumbnail feature extraction (this may take a while for many videos)...")
        # Apply the thumbnail extraction function row by row
        # This creates new columns based on the keys in the dictionary returned by extract_thumbnail_features_for_row
        # Using .progress_apply from tqdm would be nice for large datasets: from tqdm.auto import tqdm; tqdm.pandas()
        thumbnail_features_df = df.apply(extract_thumbnail_features_for_row, axis=1)
        
        # Join the new thumbnail features back to the original DataFrame
        df = pd.concat([df, thumbnail_features_df], axis=1)
        print("Thumbnail features extracted (or attempted).")

    print("\n--- Processed DataFrame Head (first 5 rows) ---")
    print(df.head())
    print("\n--- Processed DataFrame Info ---")
    df.info()
    
    # Show summary of NaNs for new features
    new_feature_cols = [
        'title_length_chars', 'title_length_words', 'title_has_number',
        'title_all_caps_words_count', 'title_has_question_mark', 'title_has_exclamation_mark',
        'local_thumbnail_path', 'thumbnail_avg_brightness', 'thumbnail_avg_saturation',
        'thumbnail_has_face', 'thumbnail_text_detected_ocr'
    ]
    print("\n--- NaN Counts for Extracted Features ---")
    for col in new_feature_cols:
        if col in df.columns:
            print(f"{col}: {df[col].isna().sum()} NaNs")
        else:
            print(f"{col}: Column not found")

    return df

# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy CSV for testing if you don't have 'youtube_video_data.csv'
    # Replace this with your actual CSV path.
    INPUT_CSV_PATH = "youtube_video_data.csv"

    if not os.path.exists(INPUT_CSV_PATH):
        print(f"'{INPUT_CSV_PATH}' not found. Creating a dummy CSV for demonstration.")
        dummy_data = {
            'video_id': ['vid001_rick', 'vid002_invalid', 'vid003_google', 'vid004_no_url', 'vid005_short'],
            'title': [
                'My AMAZING Test Video! (Numbers 123)',
                'Quick Guide to Python Programming?',
                'TOP 5 Secrets REVEALED',
                'A simple vlog',
                'Cool!'
            ],
            'thumbnail_url': [
                'https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg', # Rick Astley
                'https://i.ytimg.com/this_is_not_a_real_video_id/hqdefault.jpg', # Invalid, should fail download
                'https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png', # Google Logo
                None, # Missing URL
                'https://i.ytimg.com/vi/short/hqdefault.jpg' # Example for a short video ID
            ],
            'impressions': [1000, 2000, 500, 100, 3000],
            'clicks': [100, 150, 50, 2, 300],
            'category': ['Test', 'Education', 'Entertainment','Vlog', 'Test'],
            'publish_date': ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-10', '2023-03-01'],
            'transcript_available': [True, False, True, False, True]
        }
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_csv(INPUT_CSV_PATH, index=False)
        print(f"Dummy '{INPUT_CSV_PATH}' created. Please replace it with your actual data or ensure your data is in this file.")

    # Process the data
    processed_df = process_data(csv_path=INPUT_CSV_PATH)

    if processed_df is not None:
        # Example: Save the processed DataFrame
        OUTPUT_CSV_PATH = "processed_youtube_data.csv"
        try:
            processed_df.to_csv(OUTPUT_CSV_PATH, index=False)
            print(f"\nProcessed data saved to '{OUTPUT_CSV_PATH}'")
        except Exception as e:
            print(f"\nError saving processed data to '{OUTPUT_CSV_PATH}': {e}")