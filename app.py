import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import zipfile
import os

# ==================== DATA LOADING ====================
@st.cache_data

def load_data():
    zip_path = "spotify_songs.csv.zip"
    csv_filename = "spotify_songs.csv"

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()

    df = pd.read_csv(csv_filename)
    df.dropna(subset=["track_name", "playlist_genre", "track_popularity"], inplace=True)
    return df

df = load_data()

# ==================== HALAMAN UTAMA ====================
menu = st.sidebar.selectbox("Navigasi", ["Home", "Rekomendasi", "Histori"])

if menu == "Home":
    st.title("🎶 10 Musik Terpopuler")
    top10 = df.sort_values(by="track_popularity", ascending=False).head(10)
    st.table(top10[["track_name", "track_artist", "track_popularity"]])

    st.markdown("---")
    st.header("🎧 5 Musik Terpopuler dari Tiap Genre")
    genres = df["playlist_genre"].unique()
    for genre in genres:
        st.subheader(f"Genre: {genre}")
        top_genre = df[df["playlist_genre"] == genre]
        top5 = top_genre.sort_values(by="track_popularity", ascending=False).head(5)
        st.table(top5[["track_name", "track_artist", "track_popularity"]])

# ==================== HALAMAN REKOMENDASI ====================
elif menu == "Rekomendasi":
    st.title("🔍 Rekomendasi Musik Berdasarkan Judul & Genre")
    input_lagu = st.selectbox("Pilih Judul Lagu", df["track_name"].unique())

    if input_lagu:
        selected_index = df[df["track_name"] == input_lagu].index[0]
        genre_input = df.loc[selected_index, "playlist_genre"]

        df_genre = df[df["playlist_genre"] == genre_input].reset_index(drop=True)

        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(df_genre["track_name"])

        match_in_genre = df_genre[df_genre["track_name"] == input_lagu]
        if not match_in_genre.empty:
            idx_in_genre = match_in_genre.index[0]
            cosine_sim = cosine_similarity(tfidf_matrix[idx_in_genre], tfidf_matrix).flatten()
            top_indices = cosine_sim.argsort()[::-1][1:6]

            rekomendasi = df_genre.iloc[top_indices][
                ["track_name", "track_artist", "track_album_name", "playlist_genre"]
            ]

            st.success(f"Rekomendasi lagu dengan genre: {genre_input}")
            st.table(rekomendasi)

            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({
                "input": input_lagu,
                "genre": genre_input,
                "rekomendasi": rekomendasi.to_dict(orient="records")
            })
        else:
            st.error("Lagu tidak ditemukan dalam genre tersebut.")

# ==================== HALAMAN HISTORI ====================
elif menu == "Histori":
    st.title("📜 Riwayat Rekomendasi")
    if "history" in st.session_state:
        for i, hist in enumerate(st.session_state.history[::-1]):
            st.subheader(f"Input: {hist['input']} (Genre: {hist['genre']})")
            df_hist = pd.DataFrame(hist["rekomendasi"])
            st.table(df_hist)
    else:
        st.info("Belum ada histori rekomendasi.")
