import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

# === Load Dataset ===
@st.cache_data
def load_data():
    df = pd.read_csv("spotify_songs.csv")  # pastikan file sudah diekstrak
    return df.dropna(subset=["track_name", "playlist_genre", "track_popularity"])

df = load_data()

# === Setup Halaman ===
st.set_page_config(page_title="Sistem Rekomendasi Musik", layout="wide")

# === Inisialisasi Histori ===
if "history" not in st.session_state:
    st.session_state["history"] = []

# === Sidebar Navigasi ===
page = st.sidebar.selectbox("Navigasi", ["Home", "Rekomendasi", "Histori"])

# === Halaman Home ===
if page == "Home":
    st.title("🎧 10 Musik Terpopuler")
    top10 = df.sort_values("track_popularity", ascending=False).head(10)
    st.table(top10[["track_name", "track_artist", "track_popularity"]])
    
    st.markdown("---")
    st.header("🔥 5 Musik Terpopuler per Genre")

    for genre in df["playlist_genre"].unique():
        st.subheader(f"🎵 Genre: {genre}")
        top_genre = df[df["playlist_genre"] == genre].sort_values("track_popularity", ascending=False).head(5)
        st.table(top_genre[["track_name", "track_artist", "track_popularity"]])

# === Halaman Rekomendasi ===
elif page == "Rekomendasi":
    st.title("🔍 Rekomendasi Musik Berdasarkan Genre dan Judul")
    
    judul_input = st.text_input("Masukkan Judul Lagu")
    opsi_lagu = df[df["track_name"].str.contains(judul_input, case=False, na=False)]

    if not opsi_lagu.empty:
        pilihan = st.selectbox("Pilih Lagu", opsi_lagu["track_name"].unique())
        lagu_terpilih = df[df["track_name"] == pilihan].iloc[0]
        
        st.markdown(f"**Judul:** {lagu_terpilih['track_name']}  \n"
                    f"**Artis:** {lagu_terpilih['track_artist']}  \n"
                    f"**Album:** {lagu_terpilih['track_album_name']}  \n"
                    f"**Genre:** {lagu_terpilih['playlist_genre']}")

        # TF-IDF Judul
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(df["track_name"].fillna(""))
        idx = df[df["track_name"] == pilihan].index[0]
        cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

        df["similarity"] = cosine_sim
        rekomendasi = df[
            (df["playlist_genre"] == lagu_terpilih["playlist_genre"]) &
            (df["track_name"] != pilihan)
        ].sort_values("similarity", ascending=False).head(5)

        st.subheader("🎯 Rekomendasi Lagu")
        for _, row in rekomendasi.iterrows():
            st.markdown(f"- 🎵 **{row['track_name']}** oleh *{row['track_artist']}* _(Album: {row['track_album_name']})_")

        # Simpan ke histori
        st.session_state["history"].append({
            "judul_input": pilihan,
            "rekomendasi": rekomendasi[["track_name", "track_artist", "track_album_name"]].to_dict("records")
        })

    else:
        if judul_input:
            st.warning("Lagu tidak ditemukan. Coba masukkan kata kunci lain.")

# === Halaman Histori ===
elif page == "Histori":
    st.title("📜 Riwayat Pencarian dan Rekomendasi")

    if not st.session_state["history"]:
        st.info("Belum ada riwayat rekomendasi.")
    else:
        for i, item in enumerate(st.session_state["history"], 1):
            st.markdown(f"### 🔎 Pencarian ke-{i}: {item['judul_input']}")
            for lagu in item["rekomendasi"]:
                st.markdown(f"- 🎵 **{lagu['track_name']}** oleh *{lagu['track_artist']}* _(Album: {lagu['track_album_name']})_")
            st.markdown("---")
