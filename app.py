import streamlit as st
import pandas as pd
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ============================
# Load dan Persiapan Dataset
# ============================
@st.cache_data
def load_data():
    with zipfile.ZipFile("spotify_songs.csv.zip", "r") as zip_ref:
        zip_ref.extractall()
    df = pd.read_csv("spotify_songs.csv")
    df.dropna(subset=['track_name', 'playlist_genre', 'track_popularity'], inplace=True)
    return df

df = load_data()

# ============================
# Klasifikasi Popularitas
# ============================
@st.cache_data
def train_rf_model(df):
    df_rf = df.copy()
    df_rf['popularity_label'] = df_rf['track_popularity'].apply(lambda x: 'High' if x >= 70 else 'Low')
    le = LabelEncoder()
    df_rf['genre_encoded'] = le.fit_transform(df_rf['playlist_genre'])
    features = df_rf[['genre_encoded', 'track_popularity']]
    target = df_rf['popularity_label']
    rf = RandomForestClassifier(random_state=42)
    rf.fit(features, target)
    df_rf['predicted_popularity'] = rf.predict(features)
    return df_rf, rf

df_rf, rf_model = train_rf_model(df)

# ============================
# Setup Histori
# ============================
if "history" not in st.session_state:
    st.session_state.history = []

# ============================
# Navigasi Sidebar
# ============================
st.set_page_config(page_title="Sistem Rekomendasi Musik", layout="wide")
menu = st.sidebar.radio("📂 Navigasi", ["Home", "Rekomendasi", "Histori"])

# ============================
# Halaman HOME
# ============================
if menu == "Home":
    st.title("🎶 Sistem Rekomendasi Musik - Home")

    st.subheader("🔝 10 Lagu Terpopuler")
    top_10 = df_rf.sort_values(by="track_popularity", ascending=False).head(10)
    st.dataframe(top_10[['track_name', 'track_artist', 'track_popularity', 'predicted_popularity']])

    st.subheader("🎧 Top 5 Lagu Per Genre")
    for genre in df_rf['playlist_genre'].dropna().unique():
        st.markdown(f"##### 🎼 Genre: {genre}")
        top_per_genre = df_rf[df_rf['playlist_genre'] == genre].sort_values(by="track_popularity", ascending=False).head(5)
        st.dataframe(top_per_genre[['track_name', 'track_artist', 'track_popularity', 'predicted_popularity']])

# ============================
# Halaman Rekomendasi
# ============================
elif menu == "Rekomendasi":
    st.title("🎯 Rekomendasi Musik Berdasarkan Judul")

    input_method = st.radio("Pilih metode input judul lagu:", ["🔽 Dropdown", "⌨️ Input Manual"])

    if input_method == "🔽 Dropdown":
        selected_title = st.selectbox("Pilih Judul Lagu:", df_rf['track_name'].dropna().unique())
    else:
        selected_title = st.text_input("Ketik Judul Lagu:")

    if selected_title:
        match = df_rf[df_rf['track_name'].str.lower() == selected_title.lower()]
        if not match.empty:
            genre = match.iloc[0]['playlist_genre']
            selected_song = match.iloc[0]
            df_genre = df_rf[df_rf['playlist_genre'] == genre].copy()

            # ========== Genre-based Recommendation ==========
            st.markdown("## 🎧 Rekomendasi Berdasarkan Genre yang Sama")
            genre_recs = df_genre[df_genre['track_name'].str.lower() != selected_title.lower()]
            genre_recs = genre_recs.sort_values(by='predicted_popularity', ascending=False).head(5)

            for _, row in genre_recs.iterrows():
                st.markdown(f"✅ **{row['track_name']}** oleh *{row['track_artist']}*")
                st.caption(f"Genre: {row['playlist_genre']} | Popularitas: {row['predicted_popularity']}")

            # ========== Title Similarity (TF-IDF) ==========
            st.markdown("## 📝 Rekomendasi Berdasarkan Kemiripan Judul")

            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(df_genre['track_name'])

            index_input = df_genre[df_genre['track_name'].str.lower() == selected_title.lower()].index
            if len(index_input) > 0:
                idx = index_input[0]
                cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
                similar_indices = cosine_sim.argsort()[::-1][1:11]

                title_recs = df_genre.iloc[similar_indices]
                title_recs = title_recs.sort_values(by='predicted_popularity', ascending=False).head(5)

                for _, row in title_recs.iterrows():
                    st.markdown(f"🎵 **{row['track_name']}** oleh *{row['track_artist']}*")
                    st.caption(f"Genre: {row['playlist_genre']} | Popularitas: {row['predicted_popularity']}")

                # Simpan histori
                st.session_state.history.append({
                    "input": selected_title,
                    "genre": genre,
                    "rekomendasi_genre": genre_recs['track_name'].tolist(),
                    "rekomendasi_tfidf": title_recs['track_name'].tolist()
                })
            else:
                st.warning("Judul tidak ditemukan dalam genre terkait.")
        else:
            st.warning("Judul tidak ditemukan dalam data.")

# ============================
# Halaman Histori
# ============================
elif menu == "Histori":
    st.title("🕘 Riwayat Rekomendasi")
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"### {i}. Judul Input: {item['input']} (Genre: {item['genre']})")
            st.markdown("#### 🎧 Genre yang Sama:")
            for rec in item['rekomendasi_genre']:
                st.markdown(f"- {rec}")
            st.markdown("#### 📝 Kemiripan Judul:")
            for rec in item['rekomendasi_tfidf']:
                st.markdown(f"- {rec}")
    else:
        st.info("Belum ada histori rekomendasi.")
