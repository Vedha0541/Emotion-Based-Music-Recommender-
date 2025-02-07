# youtube_search.py
from youtubesearchpython import VideosSearch

class YouTubeSearcher:
    def search_music(self, emotion):
        emotion_to_music = {
            'happy': 'Upbeat music',
            'sad': 'Sad songs',
            'angry': 'Rock music',
            'surprised': 'Electronic music',
            'neutral': 'Ambient music',
        }
        genre = emotion_to_music.get(emotion, 'Pop music')
        search = VideosSearch(genre, limit=5)
        return [result['link'] for result in search.result()['result']]
