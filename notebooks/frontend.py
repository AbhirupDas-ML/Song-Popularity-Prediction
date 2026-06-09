import gradio as gr
import requests

def predict_hit(genre, dance, energy, loud, speech, acoustic, instrument, live, valence, tempo, duration, explicit):
    api_url = "http://127.0.0.1:8000/predict_custom"
    
    payload = {
        "genre": genre,
        "danceability": dance,
        "energy": energy,
        "loudness": loud,
        "speechiness": speech,
        "acousticness": acoustic,
        "instrumentalness": instrument,
        "liveness": live,
        "valence": valence,
        "tempo": tempo,
        "duration_ms": int(duration * 1000), 
        "explicit": 1 if explicit else 0
    }
    
    try:
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            data = response.json()
            score = data['predicted_popularity_score']
            status = data['classification']
            
            score_text = f"{score} / 100"
            status_text = f"{status}" if status == "Hit" else f"{status}"
            return score_text, status_text
        else:
            return "Error", f"Backend Error: {response.status_code}"
    except:
        return "Error", "Could not connect to FastAPI server."

# Build the Interactive Dashboard
with gr.Blocks() as demo:
    gr.Markdown("# The AI Hit Maker")
    gr.Markdown("Dial in the acoustic physics of your track below, select a genre, and see if the XGBoost algorithm thinks it will be a global hit!")
    
    with gr.Row():
        genre_dropdown = gr.Dropdown(
            choices=[
                "acoustic", "afrobeat", "alt-rock", "alternative", "ambient", "anime", 
                "black-metal", "bluegrass", "blues", "brazil", "breakbeat", "british", 
                "cantopop", "chicago-house", "children", "chill", "classical", "club", 
                "comedy", "country", "dance", "dancehall", "death-metal", "deep-house", 
                "detroit-techno", "disco", "disney", "drum-and-bass", "dub", "dubstep", 
                "edm", "electro", "electronic", "emo", "folk", "forro", "french", "funk", 
                "garage", "german", "gospel", "goth", "grindcore", "groove", "grunge", 
                "guitar", "happy", "hard-rock", "hardcore", "hardstyle", "heavy-metal", 
                "hip-hop", "honky-tonk", "house", "idm", "indian", "indie", "indie-pop", 
                "industrial", "iranian", "j-dance", "j-idol", "j-pop", "j-rock", "jazz", 
                "k-pop", "kids", "latin", "latino", "malay", "mandopop", "metal", 
                "metalcore", "minimal-techno", "mpb", "new-age", "opera", "pagode", 
                "party", "piano", "pop", "pop-film", "power-pop", "progressive-house", 
                "psych-rock", "punk", "punk-rock", "r-n-b", "reggae", "reggaeton", "rock", 
                "rock-n-roll", "rockabilly", "romance", "sad", "salsa", "samba", 
                "sertanejo", "show-tunes", "singer-songwriter", "ska", "sleep", "soul", 
                "spanish", "study", "swedish", "synth-pop", "tango", "techno", "trance", 
                "trip-hop", "turkish", "world-music"
            ],
            value="pop",
            label="Select Target Genre",
            scale=1
        )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Track Physics")
            dance = gr.Slider(0, 1, value=0.5, label="Danceability")
            energy = gr.Slider(0, 1, value=0.5, label="Energy")
            valence = gr.Slider(0, 1, value=0.5, label="Valence (Happiness)")
            acoustic = gr.Slider(0, 1, value=0.2, label="Acousticness")
            
        with gr.Column():
            gr.Markdown("### Mix & Master")
            loud = gr.Slider(-60, 0, value=-5.0, label="Loudness (dB)")
            tempo = gr.Slider(50, 200, value=120, label="Tempo (BPM)")
            speech = gr.Slider(0, 1, value=0.05, label="Speechiness")
            instrument = gr.Slider(0, 1, value=0.0, label="Instrumentalness")
            
        with gr.Column():
            gr.Markdown("### Track Info")
            duration = gr.Slider(60, 300, value=180, label="Duration (Seconds)")
            live = gr.Slider(0, 1, value=0.1, label="Liveness (Audience Noise)")
            explicit = gr.Checkbox(label="Explicit Content (E)")
            
            predict_btn = gr.Button("Predict AI Popularity", variant="primary")
            
    with gr.Row():
        score_output = gr.Textbox(label="Predicted Popularity Score", interactive=False)
        status_output = gr.Textbox(label="AI Verdict", interactive=False)
        
    predict_btn.click(
        fn=predict_hit, 
        inputs=[genre_dropdown, dance, energy, loud, speech, acoustic, instrument, live, valence, tempo, duration, explicit], 
        outputs=[score_output, status_output]
    )

if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Monochrome())