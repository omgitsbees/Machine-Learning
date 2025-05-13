from transformers import pipeline
import pytube
import whisper
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
from datetime import datetime, timedelta
import re

# Download necessary NLTK data if you haven't already
nltk.download('punkt', quiet=True)

class VideoSummarizerChapterer:
    def __init__(self, summarization_model="facebook/bart-large-cnn", whisper_model_name="base"):
        """
        Initializes the VideoSummarizerChapterer with specified models.

        Args:
            summarization_model (str): Name of the transformer model for summarization.
            whisper_model_name (str): Name of the Whisper ASR model.
        """
        self.summarizer = pipeline("summarization", model=summarization_model)
        self.whisper_model = whisper.load_model(whisper_model_name)
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    def download_audio(self, youtube_url, output_path="."):
        """
        Downloads the audio from a YouTube video.

        Args:
            youtube_url (str): The URL of the YouTube video.
            output_path (str): The directory to save the audio file.

        Returns:
            str: The path to the downloaded audio file, or None if download fails.
        """
        try:
            yt = pytube.YouTube(youtube_url)
            audio_stream = yt.streams.filter(only_audio=True).first()
            if audio_stream:
                output_file = os.path.join(output_path, f"{yt.video_id}.mp4")
                audio_stream.download(output_path=output_path, filename=yt.video_id)
                return output_file
            else:
                print("No audio stream found for this video.")
                return None
        except pytube.exceptions.RegexMatchError:
            print("Invalid YouTube URL.")
            return None
        except pytube.exceptions.AgeRestrictedError:
            print("Video is age-restricted and cannot be accessed.")
            return None
        except Exception as e:
            print(f"An error occurred during download: {e}")
            return None

    def transcribe_audio(self, audio_path):
        """
        Transcribes the audio file using Whisper.

        Args:
            audio_path (str): The path to the audio file.

        Returns:
            dict: A dictionary containing the transcription with timestamps.
        """
        if audio_path and os.path.exists(audio_path):
            try:
                transcript = self.whisper_model.transcribe(audio_path)
                return transcript
            except Exception as e:
                print(f"An error occurred during transcription: {e}")
                return None
        else:
            print(f"Audio file not found: {audio_path}")
            return None

    def get_segments_with_timestamps(self, transcript):
        """
        Extracts segments with start and end timestamps from the transcript.

        Args:
            transcript (dict): The transcript dictionary from Whisper.

        Returns:
            list: A list of dictionaries, where each dictionary contains 'text', 'start', and 'end'.
        """
        segments_with_timestamps = []
        if transcript and 'segments' in transcript:
            for segment in transcript['segments']:
                segments_with_timestamps.append({
                    'text': segment['text'].strip(),
                    'start': segment['start'],
                    'end': segment['end']
                })
        return segments_with_timestamps

    def summarize_transcript(self, full_text, max_length=150, min_length=30):
        """
        Summarizes the full transcript.

        Args:
            full_text (str): The complete text of the transcript.
            max_length (int): Maximum length of the summary.
            min_length (int): Minimum length of the summary.

        Returns:
            str: The summarized text.
        """
        if not full_text:
            return "No text to summarize."
        try:
            summary = self.summarizer(full_text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            return summary
        except Exception as e:
            print(f"An error occurred during summarization: {e}")
            return "Could not generate summary."

    def segment_video_by_topic(self, segments, min_segment_length=30):
        """
        Segments the video based on topic changes using text similarity.

        Args:
            segments (list): A list of transcript segments with timestamps.
            min_segment_length (int): Minimum number of words for a segment to be considered.

        Returns:
            list: A list of dictionaries, where each dictionary represents a chapter
                  with 'title', 'start_time', and 'end_time'.
        """
        if not segments or len(segments) < 2:
            return []

        segment_texts = [segment['text'] for segment in segments if len(word_tokenize(segment['text'])) >= min_segment_length]
        valid_segments_with_timestamps = [segment for segment in segments if len(word_tokenize(segment['text'])) >= min_segment_length]

        if not segment_texts:
            return []

        tfidf_matrix = self.tfidf_vectorizer.fit_transform(segment_texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Identify potential chapter boundaries based on lower similarity with the previous segment
        chapter_boundaries = [0]
        threshold = 0.15  # Adjust this threshold as needed
        for i in range(1, len(similarity_matrix)):
            if similarity_matrix[i, i - 1] < threshold:
                chapter_boundaries.append(i)
        chapter_boundaries.append(len(valid_segments_with_timestamps))

        chapters = []
        for i in range(len(chapter_boundaries) - 1):
            start_index = chapter_boundaries[i]
            end_index = chapter_boundaries[i + 1]
            if start_index < len(valid_segments_with_timestamps) and end_index <= len(valid_segments_with_timestamps) and start_index < end_index:
                start_time = valid_segments_with_timestamps[start_index]['start']
                end_time = valid_segments_with_timestamps[end_index - 1]['end']
                chapter_text = " ".join([valid_segments_with_timestamps[j]['text'] for j in range(start_index, end_index)])

                # Generate a concise title for the chapter (you might need a more sophisticated approach)
                title_sentences = sent_tokenize(chapter_text)
                chapter_title = title_sentences[0] if title_sentences else f"Chapter {len(chapters) + 1}"
                chapter_title = chapter_title[:50] + "..." if len(chapter_title) > 50 else chapter_title

                chapters.append({
                    'title': chapter_title,
                    'start_time': self.format_timestamp(start_time),
                    'end_time': self.format_timestamp(end_time)
                })

        return chapters

    def format_timestamp(self, seconds):
        """
        Formats seconds into HH:MM:SS or MM:SS format.

        Args:
            seconds (float): The time in seconds.

        Returns:
            str: The formatted timestamp.
        """
        if seconds >= 3600:
            return str(timedelta(seconds=seconds))
        else:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"

    def generate_summary_and_chapters(self, youtube_url):
        """
        Downloads, transcribes, summarizes, and chapters a YouTube video.

        Args:
            youtube_url (str): The URL of the YouTube video.

        Returns:
            tuple: A tuple containing the summary (str) and a list of chapters (list of dicts),
                   or (None, None) if an error occurs.
        """
        print("Downloading audio...")
        audio_path = self.download_audio(youtube_url)
        if not audio_path:
            return None, None

        print("Transcribing audio...")
        transcript_data = self.transcribe_audio(audio_path)
        if not transcript_data or 'text' not in transcript_data:
            return None, None

        full_transcript_text = transcript_data['text']
        segments_with_timestamps = self.get_segments_with_timestamps(transcript_data)

        print("Summarizing transcript...")
        summary = self.summarize_transcript(full_transcript_text)

        print("Segmenting video into chapters...")
        chapters = self.segment_video_by_topic(segments_with_timestamps)

        # Clean up the downloaded audio file
        try:
            os.remove(audio_path)
            print(f"Deleted temporary audio file: {audio_path}")
        except OSError as e:
            print(f"Error deleting temporary audio file {audio_path}: {e}")

        return summary, chapters

if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with your YouTube video URL
    summarizer_chapterer = VideoSummarizerChapterer()
    summary, chapters = summarizer_chapterer.generate_summary_and_chapters(video_url)

    if summary:
        print("\n--- Summary ---")
        print(summary)

    if chapters:
        print("\n--- Chapters ---")
        for chapter in chapters:
            print(f"{chapter['start_time']} - {chapter['end_time']}: {chapter['title']}")