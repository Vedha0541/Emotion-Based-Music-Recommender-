# Emotion-Based-Music-Recommender-
An AI-based music recommendation system that uses facial emotion detection to suggest songs dynamically.

🎶 Overview

The Emotion-Based Music Recommender (EmoMusic) is an AI-powered system that detects facial emotions in real time and suggests personalized music based on the user's mood. By integrating machine learning, computer vision, and YouTube API, the system enhances user experience by recommending music dynamically based on detected emotions.

🧠 Technologies Used

Python (for backend processing)

Streamlit (for user interface)

OpenCV & DeepFace (for facial emotion recognition)

TensorFlow (for training and deploying models)

YouTube API (for fetching music recommendations)

🚀 Features

✅ Real-time emotion detection via webcam
✅ Personalized music recommendations based on mood
✅ Seamless YouTube integration for song playback
✅ Lightweight UI using Streamlit
✅ Uses CNN models for emotion classification

⚙️ Installation & Setup

Clone the repository:

git clone https://github.com/your-username/Emotion-Based-Music-Recommender.git
cd Emotion-Based-Music-Recommender

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

🎭 How It Works

Face Detection: The system captures facial expressions using the webcam.

Emotion Recognition: The model classifies emotions (Happy, Sad, Angry, Neutral, etc.).

Music Recommendation: The detected emotion is mapped to a music category.

YouTube Integration: The system fetches and plays a relevant song from YouTube.

🔮 Future Enhancements

Improve emotion recognition accuracy with more datasets

Add support for Spotify & Apple Music integration

Implement mobile version for wider accessibility


