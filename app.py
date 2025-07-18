import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

# --- Styling dark mode ---
st.markdown("""
    <style>
    .main, .block-container {
        background-color: #121212;
        color: #FFFFFF;
    }
    h1, h2, h3, h4 {
        color: #1DB954;
    }
    button[kind="primary"] {
        background-color: #1DB954;
        color: #FFFFFF;
        border-radius: 20px;
        border: none;
        padding: 8px 20px;
    }
    button[kind="primary"]:hover {
        background-color: #1ed760;
    }
    .stTextInput>div>div>input {
        background-color: #222222;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 8px;
    }
    .music-card {
        background-color: #282828;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
        cursor: pointer;
        transition: background-color 0.2s ease;
    }
    .music-card:hover {
        background-color: #333333;
    }
    .music-cover {
        width: 50px;
        height: 50px;
        color: #1DB954;
        font-size: 30px;
        display: flex;
        justify-content: center;
        align-items: center;
        border-radius: 6px;
        flex-shrink: 0;
    }
    .music-info {
        flex-grow: 1;
    }
    .music-title {
        font-weight: 600;
        font-size: 16px;
        margin: 0;
    }
    .music-artist {
        color: #b3b3b3;
        margin: 0;
        font-size: 14px;
    }
    .popularity {
        color: #1DB954;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Data from ZIP ---
@st.cache_data

def load_data():
    zip_path = "spotify_songs.csv.zip"
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_filename = z.namelist()[0]
        with z.open(csv_filename) as f:
            df = pd.read_csv(f)

    fitur_wajib = ["track_popularity", "playlist_genre", "playlist_subgenre", "tempo", "duration_ms", "energy", "danceability"]
    df_clean = df.dropna(subset=fitur_wajib)

    low_thresh = df_clean['track_popularity'].quantile(0.33)
    high_thresh = df_clean['track_popularity'].quantile(0.66)

    def categorize_popularity(pop):
        if pop <= low_thresh:
            return 'Rendah'
        elif pop > high_thresh:
            return 'Tinggi'
        else:
            return np.nan

    df_clean['pop_category'] = df_clean['track_popularity'].apply(categorize_popularity)
    df_clean = df_clean.dropna(subset=['pop_category'])

    label_enc = LabelEncoder()
    df_clean['pop_encoded'] = label_enc.fit_transform(df_clean['pop_category'])
    return df, df_clean, label_enc

# --- Main App ---
df_raw, df_filtered, label_encoder = load_data()

st.title("🎵 Rekomendasi Musik & Prediksi Popularitas")

# --- Eksplorasi Data ---
with st.expander("📈 Lihat Statistik Data Lengkap"):
    st.dataframe(df_raw.head(20))
    st.markdown("### Statistik Deskriptif")
    st.write(df_raw.describe())
    st.markdown("### Distribusi Genre Playlist")
    st.bar_chart(df_raw['playlist_genre'].value_counts())

# --- Sidebar ---
st.sidebar.header("🔎 Filter Musik")
judul_pilihan = st.sidebar.selectbox("Pilih Judul Lagu", options=df_filtered['track_name'].unique())

# --- Fitur numerik dari seluruh fitur yang tersedia ---
fitur_numerik = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
if 'pop_encoded' in fitur_numerik:
    fitur_numerik.remove('pop_encoded')

# --- Rekomendasi berdasarkan kemiripan fitur ---
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df_filtered[fitur_numerik])
sim_matrix = cosine_similarity(X_scaled)

idx_lagu = df_filtered[df_filtered['track_name'] == judul_pilihan].index[0]
sim_scores = list(enumerate(sim_matrix[idx_lagu]))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
similar_indexes = [i[0] for i in sim_scores[1:6]]

st.subheader("🎧 Lagu Mirip Berdasarkan Seluruh Fitur Numerik")
for i in similar_indexes:
    lagu = df_filtered.iloc[i]
    st.markdown(f"""
    <div class="music-card">
        <div class="music-cover">🎵</div>
        <div class="music-info">
            <p class="music-title">{lagu['track_name']}</p>
            <p class="music-artist">{lagu['track_artist']} - {lagu['track_album_name']}</p>
        </div>
        <div class="popularity">🔥 {lagu['track_popularity']}</div>
    </div>
    """, unsafe_allow_html=True)

# --- Prediksi popularitas ---
X = df_filtered[fitur_numerik]
y = df_filtered['pop_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

st.subheader("📊 Prediksi Popularitas Musik Berdasarkan Fitur Numerik")
fitur_input = []
st.markdown("Masukkan nilai untuk semua fitur numerik:")
for fitur in fitur_numerik:
    min_val = float(df_filtered[fitur].min())
    max_val = float(df_filtered[fitur].max())
    mean_val = float(df_filtered[fitur].mean())
    if "_ms" in fitur or "duration" in fitur:
        val = st.slider(fitur, int(min_val), int(max_val), int(mean_val))
    else:
        val = st.slider(fitur, float(min_val), float(max_val), float(mean_val))
    fitur_input.append(val)

fitur_input = np.array([fitur_input])
prediksi = model.predict(fitur_input)[0]
kategori = label_encoder.inverse_transform([prediksi])[0]
st.success(f"Prediksi popularitas lagu: {kategori}")
