from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

api = FastAPI(title='Spotify Song Popularity Prediction')

# Loading Models
print("Loading Models and Scaler...")
scaler = joblib.load('../models/spotify_2.0_minmax_scaler.joblib')
regressor = joblib.load('../models/spotify_2.0_XGBregressor_model.joblib')
classifier = joblib.load('../models/spotify_2.0_XGBclassifier_model.joblib')
print("Models Loaded!")

GENRE_MEANS = {
    "acoustic": 45.0, "afrobeat": 24.8, "alt-rock": 52.9, "alternative": 47.5, 
    "ambient": 47.6, "anime": 48.9, "black-metal": 22.1, "bluegrass": 25.9, 
    "blues": 48.6, "brazil": 46.4, "breakbeat": 19.5, "british": 49.4, 
    "cantopop": 35.3, "chicago-house": 11.9, "children": 38.0, "chill": 56.1, 
    "classical": 22.7, "club": 32.8, "comedy": 24.7, "country": 43.1, 
    "dance": 53.2, "dancehall": 39.4, "death-metal": 32.9, "deep-house": 49.2, 
    "detroit-techno": 11.2, "disco": 41.6, "disney": 27.5, "drum-and-bass": 26.6, 
    "dub": 42.1, "dubstep": 39.8, "edm": 57.3, "electro": 58.7, "electronic": 45.8, 
    "emo": 50.4, "folk": 46.1, "forro": 41.8, "french": 42.0, "funk": 45.6, 
    "garage": 41.0, "german": 40.9, "gospel": 39.8, "goth": 28.8, "grindcore": 14.5, 
    "groove": 37.4, "grunge": 50.4, "guitar": 29.5, "happy": 22.6, "hard-rock": 48.7, 
    "hardcore": 40.9, "hardstyle": 29.3, "heavy-metal": 26.4, "hip-hop": 57.1, 
    "honky-tonk": 15.7, "house": 57.9, "idm": 15.5, "indian": 50.2, "indie": 48.1, 
    "indie-pop": 56.4, "industrial": 32.2, "iranian": 6.1, "j-dance": 25.6, 
    "j-idol": 25.1, "j-pop": 44.2, "j-rock": 36.9, "jazz": 41.9, "k-pop": 58.4, 
    "kids": 15.0, "latin": 26.7, "latino": 56.0, "malay": 30.4, "mandopop": 45.4, 
    "metal": 57.0, "metalcore": 43.1, "minimal-techno": 34.3, "mpb": 41.8, 
    "new-age": 29.7, "opera": 26.8, "pagode": 45.7, "party": 27.7, "piano": 51.3, 
    "pop": 61.4, "pop-film": 59.1, "power-pop": 26.9, "progressive-house": 49.9, 
    "psych-rock": 43.8, "punk": 44.0, "punk-rock": 42.0, "r-n-b": 46.2, 
    "reggae": 38.6, "reggaeton": 46.2, "rock": 33.9, "rock-n-roll": 36.4, 
    "rockabilly": 30.2, "romance": 9.1, "sad": 52.0, "salsa": 31.2, "samba": 37.4, 
    "sertanejo": 48.0, "show-tunes": 31.7, "singer-songwriter": 52.1, "ska": 35.9, 
    "sleep": 43.5, "soul": 54.9, "spanish": 42.6, "study": 27.2, "swedish": 41.8, 
    "synth-pop": 38.2, "tango": 20.6, "techno": 40.6, "trance": 38.7, "trip-hop": 33.7, 
    "turkish": 41.5, "world-music": 41.5
}

class SongFeatures(BaseModel):
    genre: str
    duration_ms: int
    explicit: int
    danceability: float
    energy: float
    loudness: float
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float

@api.post("/predict_custom")
def predict_custom(features: SongFeatures):
    
    live_data = features.dict()
    
    #Extracting the genre and converting to the numeric mean score
    selected_genre = live_data.pop('genre') 
    live_data['genre_mean_popularity'] = GENRE_MEANS.get(selected_genre)
    
    #Average values for some missing features
    live_data['key'] = 5 
    live_data['mode'] = 1 
    live_data['time_signature'] = 4 
    
    live_data['banger_index'] = live_data['danceability'] * live_data['energy'] * live_data['loudness']
    live_data['chill_index'] = live_data['valence'] * live_data['acousticness']
    
    input_data = pd.DataFrame([live_data])
    expected_cols = [
        'banger_index', 'chill_index', 'key', 'genre_mean_popularity', 
        'duration_ms', 'explicit', 'danceability', 'energy', 'loudness', 
        'mode', 'speechiness', 'acousticness', 'instrumentalness', 
        'liveness', 'valence', 'tempo', 'time_signature'
    ]
    input_data = input_data[expected_cols]
    
    #Scaling and Predictions
    scaled_features = scaler.transform(input_data)
    predicted_score = float(regressor.predict(scaled_features)[0])
    is_hit = int(classifier.predict(scaled_features)[0])
    
    return {
        "predicted_popularity_score": round(predicted_score, 2),
        "classification": "Hit" if is_hit == 1 else "Flop"
    }