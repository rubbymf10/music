import streamlit as st
import pandas as pd
import numpy as np
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Styling
# -------------------------------
st.markdown("""
    <style>
    .main, .block-container {
        background-color: #121212;
        color: #FFFFFF;
    }
    h1, h2, h3 {
        color: #1DB954;
    }
    .music-card {
        background-color: #282828;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
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

# -------------------------------
# Fungsi Load Data
# -------------------------------
@st.cache_data
def load_data(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as z:
        # Ambil file CSV pertama dari ZIP
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        if not csv_files:
            raise ValueError("ZIP tidak berisi file CSV.")
        with z.open(csv_files[0]) as f:
            df = pd.read_csv(f)

    # Validasi kolom yang dibutuhkan
    required_columns = [
        "track_name", "track_artist", "track_album_name", "track_popularity",
        "playlist_genre", "playlist_subgenre", "tempo",
        "duration_ms", "energy", "danceability"
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di CSV.")

    # Bersihkan data
    df_clean = df.dropna(subset=required_columns)

    # Buat kategori popularitas
    low = df_clean['track_popularity'].quantile(0.33)
    high = df_clean['track_popularity'].quantile(0.66)

    def categorize(x):
        if x <= low:
            return "Rendah"
        elif x > high:
            return "Tinggi"
        else:
            return np.nan

    df_clean['pop_category'] = df_clean['track_popularity'].apply(categorize)
    df_clean = df_clean.dropna(subset=['pop_category'])

    # Encode kategori
    label_enc = LabelEncoder()
    df_clean['pop_encoded'] = label_enc.fit_transform(df_clean['pop_category'])

    return df, df_clean, label_enc

# -------------------------------
# Judul Aplikasi
# -------------------------------
st.title("🎵 Sistem Rekomendasi & Prediksi Popularitas Musik")

# -------------------------------
# Upload File ZIP
# -------------------------------
uploaded_file = st.file_uploader("📁 Upload file ZIP berisi `spotify_songs.csv`", type="zip")
if not uploaded_file:
    st.warning("Silakan upload file ZIP terlebih dahulu.")
    st.stop()

try:
    df_raw, df_filtered, label_enc = load_data(uploaded_file)
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

# -------------------------------
# Eksplorasi Data
# -------------------------------
with st.expander("📊 Lihat Statistik Data"):
    st.dataframe(df_raw.head(10))
    st.write("Deskripsi Statistik:")
    st.write(df_raw.describe())
    st.write("Distribusi Genre:")
    st.bar_chart(df_raw['playlist_genre'].value_counts())

# -------------------------------
# Sidebar Pilihan Lagu
# -------------------------------
st.sidebar.header("🔎 Filter Lagu")
judul_lagu = st.sidebar.selectbox("Pilih Judul Lagu", df_filtered['track_name'].unique())

# -------------------------------
# Rekomendasi Lagu Mirip
# -------------------------------
st.subheader("🎧 Rekomendasi Lagu Mirip")

fitur_numerik = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
if 'pop_encoded' in fitur_numerik:
    fitur_numerik.remove('pop_encoded')

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df_filtered[fitur_numerik])
sim_matrix = cosine_similarity(X_scaled)

idx_lagu = df_filtered[df_filtered['track_name'] == judul_lagu].index[0]
similarities = list(enumerate(sim_matrix[idx_lagu]))
similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
top_similar = [i[0] for i in similarities[1:6]]

for i in top_similar:
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

# -------------------------------
# Prediksi Popularitas
# -------------------------------
st.subheader("📈 Prediksi Popularitas Lagu Baru")

X = df_filtered[fitur_numerik]
y = df_filtered['pop_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

st.markdown("Masukkan nilai fitur untuk prediksi:")

input_values = []
for fitur in fitur_numerik:
    min_val = float(df_filtered[fitur].min())
    max_val = float(df_filtered[fitur].max())
    mean_val = float(df_filtered[fitur].mean())

    if "_ms" in fitur or "duration" in fitur:
        val = st.slider(fitur, int(min_val), int(max_val), int(mean_val))
    else:
        val = st.slider(fitur, float(min_val), float(max_val), float(mean_val))
    input_values.append(val)

X_input = np.array([input_values])
y_pred = model.predict(X_input)[0]
hasil_kategori = label_enc.inverse_transform([y_pred])[0]

st.success(f"🎯 Prediksi popularitas lagu: **{hasil_kategori}**")
