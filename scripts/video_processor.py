import numpy as np
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# --- Step 1: Extract Audio from Video ---
def extract_audio_from_video(video_path):
    """Extracts audio from a video file using moviepy and saves it as WAV."""
    video_clip = VideoFileClip(video_path)
    audio_path = video_path.replace(".mp4", ".wav")
    
    # Extract audio and save it as a WAV file
    video_clip.audio.write_audiofile(audio_path)
    video_clip.close()
    
    return audio_path

# --- Step 2: Generate Transcription using CMU Sphinx (Offline) ---
def generate_transcription(audio_path):
    """Generates transcription from the extracted audio using the SpeechRecognition library with CMU Sphinx."""
    recognizer = sr.Recognizer()
    
    # Load the audio file
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    
    try:
        # Perform the recognition using Sphinx (offline)
        transcription = recognizer.recognize_sphinx(audio_data)
        return transcription
    except sr.UnknownValueError:
        print("Speech Recognition could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Sphinx; {e}")
        return None

# --- Step 3: Store Transcription in a Vector Database (FAISS) ---
def store_transcription_in_vector_db(transcription, model):
    """Converts transcription to embeddings using Sentence Transformers and stores them in a FAISS vector database."""
    sentences = [sentence.strip() for sentence in transcription.split(".") if sentence.strip()]
    embeddings = model.encode(sentences)  # Generate embeddings for each sentence
    
    # Initialize FAISS vector database
    vector_db = faiss.IndexFlatL2(embeddings.shape[1])
    vector_db.add(embeddings)  # Add embeddings to the database
    
    return vector_db, sentences, embeddings

# --- Step 4: Decode and Retrieve All Sentences ---
def decode_all_embeddings(vector_db, embeddings, sentences):
    """Reconstructs all stored sentences using FAISS embeddings."""
    decoded_sentences = []
    for i in range(len(embeddings)):
        # Retrieve the nearest embedding (itself)
        _, indices = vector_db.search(np.expand_dims(embeddings[i], axis=0), k=1)
        decoded_sentences.append(sentences[indices[0][0]])  # Decode using the original sentence list
    
    return decoded_sentences

# --- Step 5: Generate Contextual Summary from Decoded Sentences ---
def generate_contextual_summary_from_decoded(decoded_sentences):
    """Generates a contextual summary from all decoded sentences."""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    full_text = " ".join(decoded_sentences)  # Combine all decoded sentences into a single text
    summary = summarizer(full_text, max_length=500, min_length=40, do_sample=False)
    return summary[0]["summary_text"]

# --- Step 6: AI-Based Dynamic Performance Analysis ---
def analyze_performance_dynamically_from_decoded(decoded_sentences):
    """Analyzes the candidate's performance dynamically based on all decoded sentences."""
    # Load pre-trained sentiment and emotion analysis pipelines
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    emotion_analyzer = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)
    
    full_text = " ".join(decoded_sentences)  # Combine all decoded sentences for analysis

    # Analyze sentiment for overall communication tone
    sentiment = sentiment_analyzer(full_text[:512])  # Limiting input to avoid exceeding token limits
    dominant_sentiment = sentiment[0]["label"]
    sentiment_score = sentiment[0]["score"]

    # Analyze emotional tones from the full text
    emotions = emotion_analyzer(full_text[:512])  # Take the top chunk of the text for processing
    emotions_scores = {emotion["label"]: emotion["score"] for emotion in emotions[0]}
    dominant_emotion = max(emotions_scores, key=emotions_scores.get)

    # Build performance traits based on analysis
    performance_analysis = {
        "Communication Style": f"Sentiment indicates a {dominant_sentiment} tone with confidence of {sentiment_score:.2f}.",
        "Active Listening": f"Emotion analysis suggests the dominant emotion is '{dominant_emotion}' (score: {emotions_scores[dominant_emotion]:.2f}).",
        "Engagement with the Interviewer": f"Analyzed responses show {dominant_emotion}, potentially reflecting the candidate's enthusiasm or mood."
    }

    return performance_analysis

# --- Main Process ---
def process_video_for_analysis(video_path):
    """Complete process for extracting and analyzing video transcription."""
    # Step 1: Extract audio from video
    audio_path = extract_audio_from_video(video_path)
    
    # Step 2: Generate transcription
    transcription = generate_transcription(audio_path)
    if not transcription:
        return {"error": "Failed to generate transcription."}

    # Step 3: Store transcription in vector database
    model = SentenceTransformer("all-mpnet-base-v2")  # Load a Sentence Transformer model
    vector_db, sentences, embeddings = store_transcription_in_vector_db(transcription, model)

    # Step 4: Decode all embeddings to retrieve all sentences
    decoded_sentences = decode_all_embeddings(vector_db, embeddings, sentences)

    # Step 5: Generate contextual summary from decoded sentences
    summary = generate_contextual_summary_from_decoded(decoded_sentences)

    # Step 6: Analyze candidate's performance dynamically
    performance_analysis = analyze_performance_dynamically_from_decoded(decoded_sentences)

    return {
        "transcription": transcription,
        "decoded_sentences": decoded_sentences,
        "summary": summary,
        "performance_analysis": performance_analysis,
        "File_Path":video_path
    }

# # --- Example Usage ---
# video_path = "v4.mp4"  # Path to your video file
# results = process_video_for_analysis(video_path)

# # Output results
# if "error" in results:
#     print(results["error"])
# else:
#     print("\n--- Transcription ---")
#     print(results["transcription"])
#     print("\n--- Decoded Sentences ---")
#     for sentence in results["decoded_sentences"]:
#         print(f"- {sentence}")
#     print("\n--- Contextual Summary ---")
#     print(results["summary"])
#     print("\n--- Performance Analysis ---")
#     for trait, analysis in results["performance_analysis"].items():
#         print(f"{trait}: {analysis}")
