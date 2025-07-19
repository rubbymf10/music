import streamlit as st
import pandas as pd
import zipfile
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fungsi untuk memuat data dari file zip
@st.cache_data
def load_data():
    zip_path = "spotify_songs.csv.zip"
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open(zip_ref.namelist()[0]) as file:
            df = pd.read_csv(file)
    return df

df = load_data()

# Simpan histori pencarian
if "history" not in st.session_state:
    st.session_state.history = []

# --- Halaman Navigasi ---
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Menu", ["Beranda", "Rekomendasi", "Histori"])

# --- Halaman Beranda ---
if page == "Beranda":
    st.title("🎵 Musik Terpopuler")
    
    st.subheader("Top 10 Musik")
    top10 = df.sort_values("track_popularity", ascending=False).head(10)
    st.dataframe(top10[["track_name", "track_artist", "track_album_name", "track_popularity"]])
    
    st.subheader("Top 5 Musik per Genre")
    for genre in df["playlist_genre"].dropna().unique():
        st.markdown(f"**🎧 Genre: {genre}**")
        top5 = df[df["playlist_genre"] == genre].sort_values("track_popularity", ascending=False).head(5)
        st.dataframe(top5[["track_name", "track_artist", "track_album_name", "track_popularity"]])

# --- Halaman Rekomendasi ---
elif page == "Rekomendasi":
    st.title("🔍 Rekomendasi Musik")

    judul_input = st.text_input("Masukkan judul lagu:")
    if judul_input:
        # Cari lagu yang cocok
        opsi_lagu = df[df["track_name"].str.contains(judul_input, case=False, na=False)]

        if not opsi_lagu.empty:
            selected_title = st.selectbox("Pilih lagu:", opsi_lagu["track_name"].unique())

            selected_data = df[df["track_name"] == selected_title].iloc[0]
            selected_genre = selected_data["playlist_genre"]

            genre_df = df[df["playlist_genre"] == selected_genre].copy()

            # TF-IDF untuk judul lagu
            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform(genre_df["track_name"].astype(str))
            selected_index = genre_df[genre_df["track_name"] == selected_title].index[0]
            cosine_sim = cosine_similarity(tfidf_matrix[selected_index], tfidf_matrix).flatten()

            genre_df["similarity"] = cosine_sim
            rekomendasi = genre_df.sort_values("similarity", ascending=False)[1:6]

            st.subheader("🎧 Rekomendasi Lagu Berdasarkan Genre & Judul:")
            for _, row in rekomendasi.iterrows():
                st.markdown(f"- **{row['track_name']}** oleh {row['track_artist']} — Album: *{row['track_album_name']}*")

            # Tambahkan ke histori
            st.session_state.history.append({
                "input": selected_title,
                "output": rekomendasi[["track_name", "track_artist", "track_album_name"]].values.tolist()
            })

        else:
            st.warning("Lagu tidak ditemukan. Coba judul lain.")

# --- Halaman Histori ---
elif page == "Histori":
    st.title("📜 Histori Pencarian")
    
    if not st.session_state.history:
        st.info("Belum ada histori.")
    else:
        for idx, record in enumerate(st.session_state.history):
            st.markdown(f"### 🔎 Pencarian {idx+1}: **{record['input']}**")
            for output in record["output"]:
                st.markdown(f"- **{output[0]}** oleh {output[1]} — Album: *{output[2]}*")
