import streamlit as st
import pandas as pd
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================
# Load dan Persiapan Data
# =====================
@st.cache_data
def load_data():
    # Extract jika zip
    with zipfile.ZipFile("spotify_songs.csv.zip", "r") as z:
        z.extractall()
    df = pd.read_csv("spotify_songs.csv")
    df.dropna(subset=['track_name', 'playlist_genre', 'track_popularity'], inplace=True)
    return df

df = load_data()

# Simpan histori rekomendasi
if "history" not in st.session_state:
    st.session_state.history = []

# =====================
# Halaman Navigasi
# =====================
st.set_page_config(page_title="Sistem Rekomendasi Musik", layout="wide")
menu = st.sidebar.radio("Navigasi", ["Home", "Rekomendasi", "Histori"])

# =====================
# HOME
# =====================
if menu == "Home":
    st.title("🎵 Musik Terpopuler")

    st.subheader("Top 10 Musik Terpopuler")
    top_10 = df.sort_values(by="track_popularity", ascending=False).head(10)
    st.table(top_10[['track_name', 'track_artist', 'track_popularity']])

    st.subheader("Top 5 Musik per Genre")
    genres = df['playlist_genre'].dropna().unique()
    for genre in genres:
        st.markdown(f"**Genre: {genre}**")
        top_genre = df[df['playlist_genre'] == genre].sort_values(by='track_popularity', ascending=False).head(5)
        st.table(top_genre[['track_name', 'track_artist', 'track_popularity']])

# =====================
# REKOMENDASI
# =====================
elif menu == "Rekomendasi":
    st.title("🔍 Rekomendasi Musik")
    
    option = st.radio("Pilih metode input", ["Dropdown Judul Lagu", "Input Manual"])

    if option == "Dropdown Judul Lagu":
        selected_title = st.selectbox("Pilih Judul Lagu", df['track_name'].unique())
    else:
        selected_title = st.text_input("Ketik Judul Lagu")

    if selected_title:
        match_row = df[df['track_name'].str.lower() == selected_title.lower()]
        if not match_row.empty:
            selected_genre = match_row.iloc[0]['playlist_genre']
            selected_album = match_row.iloc[0]['track_album_name']

            df_genre = df[df['playlist_genre'] == selected_genre]
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(df_genre['track_name'])

            index = df_genre[df_genre['track_name'].str.lower() == selected_title.lower()].index
            if len(index) > 0:
                selected_index = index[0]
                cosine_sim = cosine_similarity(tfidf_matrix[selected_index], tfidf_matrix).flatten()
                top_indices = cosine_sim.argsort()[::-1][1:6]

                st.success(f"Rekomendasi Lagu Berdasarkan Judul: {selected_title}")
                for i in top_indices:
                    rec = df_genre.iloc[i]
                    st.markdown(f"🎧 **{rec['track_name']}** oleh *{rec['track_artist']}*")
                    st.caption(f"Album: {rec['track_album_name']} | Genre: {rec['playlist_genre']}")

                # Simpan histori
                st.session_state.history.append({
                    "judul_input": selected_title,
                    "rekomendasi": [df_genre.iloc[i]['track_name'] for i in top_indices]
                })
            else:
                st.warning("Judul tidak ditemukan dalam genre yang sama.")
        else:
            st.warning("Judul lagu tidak ditemukan di data.")

# =====================
# HISTORI
# =====================
elif menu == "Histori":
    st.title("📜 Riwayat Pencarian")
    if st.session_state.history:
        for i, item in enumerate(st.session_state.history[::-1], 1):
            st.subheader(f"{i}. Input: {item['judul_input']}")
            for rec in item['rekomendasi']:
                st.markdown(f"- {rec}")
    else:
        st.info("Belum ada histori rekomendasi.")
