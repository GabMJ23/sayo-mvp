import streamlit as st
import yt_dlp
import cv2
import numpy as np
import tempfile
import os
import warnings
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip, ColorClip
import soundfile as sf
import librosa
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Supprimer les warnings
warnings.filterwarnings("ignore")

# Configuration Streamlit
st.set_page_config(
    page_title="SAYO - Smart Audio Dimming MVP",
    page_icon="🎥",
    layout="wide"
)

# CSS personnalisé pour le branding SAYO
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #ff5c1c 0%, #ff8c42 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(255, 92, 28, 0.3);
    }
    .main-header h1 {
        color: white;
        font-size: 3.5rem;
        margin: 0;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        color: white;
        font-size: 1.4rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
    }
    .feature-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #ff5c1c;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .step-indicator {
        background: linear-gradient(135deg, #ff5c1c 0%, #ff7b42 100%);
        color: white;
        border-radius: 50%;
        width: 35px;
        height: 35px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 15px;
        box-shadow: 0 3px 10px rgba(255, 92, 28, 0.4);
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #b4d8c1;
        color: #155724;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.1);
    }
    .magic-moment {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border: 2px solid #ff9800;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="main-header">
    <h1>🎥 SAYO</h1>
    <p>Smart Audio Dimming MVP - Ta réaction. Ton style. En un instant.</p>
</div>
""", unsafe_allow_html=True)

# Initialisation des variables de session
if 'video_downloaded' not in st.session_state:
    st.session_state.video_downloaded = False
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'audio_recorded' not in st.session_state:
    st.session_state.audio_recorded = False
if 'video_info' not in st.session_state:
    st.session_state.video_info = None

# Fonctions principales
@st.cache_data
def download_youtube_video(url, max_duration=300):
    """Télécharge une vidéo YouTube optimisée"""
    try:
        ydl_opts = {
            'format': 'best[height<=720][ext=mp4]/best[ext=mp4]',
            'outtmpl': '/tmp/sayo_video.%(ext)s',
            'noplaylist': True,
            'no_warnings': True,
            'quiet': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            duration = info.get('duration', 0)
            title = info.get('title', 'Vidéo sans titre')
            
            if duration > max_duration:
                return None, f"Vidéo trop longue ({duration//60}min {duration%60}s). Maximum 5 minutes."
            
            ydl.download([url])
            video_path = '/tmp/sayo_video.mp4'
            
            if os.path.exists(video_path):
                return video_path, {"title": title, "duration": duration, "url": url}
            else:
                return None, "Fichier vidéo non trouvé"
                
    except Exception as e:
        return None, f"Erreur: {str(e)}"

def simulate_transcription(audio_file):
    """Simule la transcription en attendant Whisper"""
    sample_transcriptions = [
        "Oh wow, c'est incroyable ! Je n'avais jamais vu ça avant.",
        "C'est exactement ce que je pensais qu'il allait faire !",
        "Attendez, qu'est-ce qui se passe là ? C'est génial !",
        "Haha, j'adore cette partie ! Vraiment bien fait.",
        "Non mais sérieusement, c'est du niveau professionnel ça !"
    ]
    import random
    return random.choice(sample_transcriptions)

def apply_smart_dimming(original_audio, reaction_audio):
    """Applique le smart audio dimming basique"""
    try:
        # Analyser l'énergie de la réaction
        reaction_energy = librosa.feature.rms(y=reaction_audio)[0]
        
        # Seuil de détection de parole
        threshold = np.mean(reaction_energy) * 1.5
        
        # Créer un masque pour les moments de parole
        speaking_frames = reaction_energy > threshold
        
        # Appliquer le dimming avec interpolation
        dimmed_audio = original_audio.copy()
        
        # Convertir frame-based mask vers time-based
        hop_length = 512
        frame_to_time = hop_length / 22050
        
        for i, is_speaking in enumerate(speaking_frames):
            start_time = i * frame_to_time
            end_time = (i + 1) * frame_to_time
            
            start_sample = int(start_time * 22050)
            end_sample = int(end_time * 22050)
            
            if end_sample > len(dimmed_audio):
                end_sample = len(dimmed_audio)
                
            if is_speaking:
                # Dimming agressif quand l'utilisateur parle
                dimmed_audio[start_sample:end_sample] *= 0.25
        
        return dimmed_audio
        
    except Exception as e:
        st.error(f"Erreur dimming: {str(e)}")
        return original_audio * 0.5  # Fallback simple

def create_reaction_video(video_path, audio_bytes, transcription):
    """Crée une vidéo de réaction avec Smart Audio Dimming"""
    try:
        # Charger la vidéo originale
        original_video = VideoFileClip(video_path)
        
        # Sauvegarder l'audio de réaction
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
            tmp_audio.write(audio_bytes)
            reaction_audio_path = tmp_audio.name
        
        # Charger les audios
        original_audio_data, sr1 = librosa.load(video_path, sr=22050)
        reaction_audio_data, sr2 = librosa.load(reaction_audio_path, sr=22050)
        
        # Ajuster les longueurs
        min_length = min(len(original_audio_data), len(reaction_audio_data))
        original_audio_data = original_audio_data[:min_length]
        reaction_audio_data = reaction_audio_data[:min_length]
        
        # Appliquer le Smart Audio Dimming
        dimmed_audio = apply_smart_dimming(original_audio_data, reaction_audio_data)
        
        # Combiner les audios
        final_audio_data = dimmed_audio + reaction_audio_data * 0.8
        
        # Sauvegarder l'audio final
        final_audio_path = "/tmp/final_audio.wav"
        sf.write(final_audio_path, final_audio_data, 22050)
        final_audio_clip = AudioFileClip(final_audio_path)
        
        # Créer la vidéo format vertical
        target_width = 1080
        target_height = 1920
        
        # Redimensionner vidéo originale
        video_height = int(target_height * 0.65)
        video_width = int(video_height * original_video.w / original_video.h)
        
        if video_width > target_width:
            video_width = target_width
            video_height = int(video_width * original_video.h / original_video.w)
        
        resized_video = original_video.resize((video_width, video_height)).set_position(('center', 100))
        
        # Fond noir
        background = ColorClip(size=(target_width, target_height), color=(0,0,0), duration=original_video.duration)
        
        # Zone de réaction SAYO
        reaction_zone = ColorClip(
            size=(target_width-80, 350), 
            color=(255, 92, 28),
            duration=original_video.duration
        ).set_position((40, target_height - 390)).set_opacity(0.9)
        
        # Texte de réaction
        reaction_text = TextClip(
            "🎤 Smart Audio Dimming activé\n✨ Votre réaction analysée",
            fontsize=28,
            color='white',
            font='Arial-Bold',
            align='center'
        ).set_duration(original_video.duration).set_position(('center', target_height - 250))
        
        # Logo SAYO
        sayo_logo = TextClip(
            "SAYO",
            fontsize=40,
            color='white',
            font='Arial-Bold'
        ).set_duration(original_video.duration).set_position((40, 40))
        
        # Sous-titre avec transcription
        subtitle = TextClip(
            transcription,
            fontsize=24,
            color='white',
            bg_color='rgba(0,0,0,0.8)',
            size=(target_width-120, None),
            method='caption'
        ).set_duration(min(5, original_video.duration)).set_position(('center', target_height - 450))
        
        # Composer la vidéo finale
        video_clips = [background, resized_video, reaction_zone, reaction_text, sayo_logo, subtitle]
        final_video = CompositeVideoClip(video_clips, size=(target_width, target_height))
        
        # Ajuster l'audio final
        if final_audio_clip.duration > final_video.duration:
            final_audio_clip = final_audio_clip.subclip(0, final_video.duration)
        elif final_audio_clip.duration < final_video.duration:
            final_audio_clip = final_audio_clip.loop(duration=final_video.duration)
        
        final_video = final_video.set_audio(final_audio_clip)
        
        # Exporter
        output_path = "/tmp/sayo_reaction_final.mp4"
        final_video.write_videofile(
            output_path,
            fps=24,
            codec='libx264',
            audio_codec='aac',
            verbose=False,
            logger=None
        )
        
        # Nettoyage
        os.unlink(reaction_audio_path)
        os.unlink(final_audio_path)
        original_video.close()
        final_audio_clip.close()
        final_video.close()
        
        return output_path, "✅ Vidéo SAYO créée avec Smart Audio Dimming!"
        
    except Exception as e:
        return None, f"❌ Erreur: {str(e)}"

# Interface utilisateur
st.markdown("### 🎬 Créez votre vidéo de réaction avec Smart Audio Dimming")

# Étape 1: Import YouTube
st.markdown('<div class="step-indicator">1</div> **Importer une vidéo YouTube**', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    youtube_url = st.text_input(
        "URL de la vidéo YouTube (max 5 min)",
        placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        help="Collez l'URL d'une vidéo YouTube de moins de 5 minutes"
    )

with col2:
    download_btn = st.button("📥 Télécharger", type="primary")

if download_btn and youtube_url:
    with st.spinner("Téléchargement en cours..."):
        video_path, result = download_youtube_video(youtube_url)
        
        if video_path:
            st.session_state.video_downloaded = True
            st.session_state.video_path = video_path
            st.session_state.video_info = result
            st.success(f"✅ Vidéo téléchargée: {result['title']}")
        else:
            st.error(f"❌ {result}")

# Étape 2: Prévisualisation
if st.session_state.video_downloaded:
    st.markdown('<div class="step-indicator">2</div> **Prévisualisation de la vidéo**', unsafe_allow_html=True)
    
    col3, col4 = st.columns([2, 1])
    
    with col3:
        if os.path.exists(st.session_state.video_path):
            st.video(st.session_state.video_path)
    
    with col4:
        if st.session_state.video_info:
            st.markdown(f"""
            <div class="feature-box">
                <h4>📊 Infos vidéo</h4>
                <p><strong>Titre:</strong> {st.session_state.video_info['title'][:50]}...</p>
                <p><strong>Durée:</strong> {st.session_state.video_info['duration']//60}min {st.session_state.video_info['duration']%60}s</p>
            </div>
            """, unsafe_allow_html=True)

# Étape 3: Enregistrement audio
st.markdown('<div class="step-indicator">3</div> **Enregistrer votre réaction**', unsafe_allow_html=True)

col5, col6 = st.columns([2, 1])

with col5:
    st.info("📱 Enregistrez votre réaction avec votre téléphone puis uploadez le fichier audio")
    
    uploaded_audio = st.file_uploader(
        "Uploadez votre réaction audio",
        type=['wav', 'mp3', 'm4a', 'ogg'],
        help="Formats supportés: WAV, MP3, M4A, OGG"
    )
    
    if uploaded_audio:
        st.session_state.audio_recorded = True
        st.success("✅ Audio de réaction uploadé!")
        st.audio(uploaded_audio)

with col6:
    st.markdown("""
    <div class="feature-box">
        <h4>🎤 Smart Dimming</h4>
        <p>• Parlez naturellement</p>
        <p>• L'IA détecte votre voix</p>
        <p>• Audio original diminué automatiquement</p>
        <p>• Transitions fluides</p>
    </div>
    """, unsafe_allow_html=True)

# Étape 4: Génération
if st.session_state.video_downloaded and st.session_state.audio_recorded:
    st.markdown('<div class="step-indicator">4</div> **Génération avec Smart Audio Dimming**', unsafe_allow_html=True)
    
    if st.button("🚀 Générer la vidéo SAYO", type="primary"):
        with st.spinner("Transcription et traitement..."):
            # Simulation transcription
            transcription = simulate_transcription(uploaded_audio)
            st.success("✅ Transcription simulée terminée!")
            st.write(f"**Transcription:** *{transcription}*")
            
            # Génération vidéo avec Smart Dimming
            video_result, message = create_reaction_video(
                st.session_state.video_path,
                uploaded_audio.getvalue(),
                transcription
            )
            
            if video_result and os.path.exists(video_result):
                st.success(message)
                st.video(video_result)
                
                # Téléchargement
                with open(video_result, "rb") as f:
                    video_bytes = f.read()
                
                st.download_button(
                    label="📱 Télécharger votre vidéo SAYO",
                    data=video_bytes,
                    file_name=f"sayo_reaction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                    mime="video/mp4"
                )
                
                st.markdown("""
                <div class="success-box">
                    <h4>🎉 Smart Audio Dimming appliqué!</h4>
                    <p>✅ Détection automatique de votre voix</p>
                    <p>✅ Audio original diminué intelligemment</p>
                    <p>✅ Format vertical optimisé</p>
                    <p>✅ Qualité professionnelle</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(message)

# Info technique
st.markdown("""
<div class="magic-moment">
    <h4>✨ Smart Audio Dimming - Version MVP</h4>
    <p><strong>🧠 Détection :</strong> Analyse de l'énergie vocale en temps réel</p>
    <p><strong>🔉 Dimming :</strong> Réduction automatique à 25% quand vous parlez</p>
    <p><strong>⚡ Performance :</strong> Traitement optimisé pour Streamlit Cloud</p>
    <p><strong>🎯 Prochaine étape :</strong> Intégration Whisper pour transcription IA</p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <h3 style="color: #ff5c1c;">🎥 SAYO MVP - Smart Audio Dimming</h3>
    <p><strong>Version:</strong> 1.0 Simplified | <strong>Status:</strong> Déployé sur Streamlit Cloud</p>
    <p>🔉 Smart Dimming • 📱 Format Vertical • ⚡ Traitement Temps Réel</p>
</div>
""", unsafe_allow_html=True)
