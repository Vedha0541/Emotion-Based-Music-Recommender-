import streamlit as st
import cv2
from deepface import DeepFace
import webbrowser
import numpy as np

# Emotion mapping to YouTube search query
def get_youtube_search_query(emotion):
    emotion_to_song = {
        'happy': 'happy music playlist',
        'sad': 'sad music playlist',
        'angry': 'angry music playlist',
        'fear': 'calm music playlist',
        'surprise': 'surprise music playlist',
        'disgust': 'chill music playlist',
        'neutral': 'relaxing music playlist'
    }
    return emotion_to_song.get(emotion, 'relaxing music playlist')

# Function to process webcam input and detect emotion using DeepFace
def detect_emotion_from_webcam():
    cap = cv2.VideoCapture(0)  # Start the webcam feed
    emotion = None

    # Capture a single frame
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame.")
        cap.release()
        return None

    # Use DeepFace for emotion detection
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        detected_emotion = analysis[0]['dominant_emotion']
        emotion = detected_emotion

        # Draw bounding box and display emotion on the frame
        cv2.putText(frame, detected_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (50, 50), (400, 200), (255, 0, 0), 2)
    except Exception as e:
        print(f"Error: {e}")
    
    # Display the frame in Streamlit
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    # Release the webcam feed
    cap.release()
    return emotion

# Streamlit App UI
st.title('Emotion-Based Music Recommender')
st.write("Click the button to start emotion detection and recommend a song!")

# Button to start emotion detection and recommend a song
if st.button('Detect Emotion and Recommend Song'):
    # Run emotion detection from webcam (only capture one frame)
    detected_emotion = detect_emotion_from_webcam()

    if detected_emotion:
        st.write(f"Detected Emotion: {detected_emotion}")

        # Get the YouTube search query for the detected emotion
        search_query = get_youtube_search_query(detected_emotion)
        st.write(f"Recommended song for emotion '{detected_emotion}': {search_query}")

        # Redirect to YouTube search results for the recommended song
        webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")
    else:
        st.write("No emotion detected.")
