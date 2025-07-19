import streamlit as st
import pandas as pd
import zipfile
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load Data
def load_data():
    with zipfile.ZipFile("spotify_songs.csv.zip", 'r') as zip_ref:
        zip_ref.extractall("data")
    df = pd.read_csv("data/spotify_songs.csv")
    return df

@st.cache_data

def preprocess(df):
    df.dropna(subset=['track_name', 'track_artist', 'playlist_genre', 'lyrics'], inplace=True)
    df = df.drop_duplicates(subset=['track_id'])
    return df

def get_top_popular(df, n=10):
    return df.sort_values(by='track_popularity', ascending=False).head(n)

def get_top_popular_by_genre(df, n=5):
    genres = df['playlist_genre'].unique()
    result = pd.DataFrame()
    for genre in genres:
        top = df[df['playlist_genre'] == genre].sort_values(by='track_popularity', ascending=False).head(n)
        result = pd.concat([result, top])
    return result

def recommend(df, title, manual_input=None):
    if manual_input:
        input_text = manual_input
    else:
        input_text = title

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['track_name'])

    if title not in df['track_name'].values:
        st.warning("Judul lagu tidak ditemukan. Menampilkan rekomendasi berdasarkan input manual.")
        selected_index = None
    else:
        selected_index = df[df['track_name'] == title].index[0]

    le = LabelEncoder()
    df['popularity_class'] = pd.qcut(df['track_popularity'], q=2, labels=['Low', 'High'])

    genre_based = df[df['playlist_genre'] == df.loc[selected_index, 'playlist_genre']] if selected_index is not None else df
    genre_based = genre_based.copy()

    tfidf_genre = tfidf.transform(genre_based['track_name'])
    cosine_genre = cosine_similarity(tfidf.transform([input_text]), tfidf_genre).flatten()
    genre_based['similarity'] = cosine_genre
    genre_based = genre_based.sort_values(by=['popularity_class', 'similarity'], ascending=[False, False]).head(5)

    cosine_all = cosine_similarity(tfidf.transform([input_text]), tfidf_matrix).flatten()
    df['similarity'] = cosine_all
    title_based = df.sort_values(by=['popularity_class', 'similarity'], ascending=[False, False]).head(5)

    return genre_based, title_based

# Main Streamlit App
st.set_page_config(page_title="Spotify Music Recommendation", layout="wide")

menu = st.sidebar.selectbox("Menu", ["Home", "Rekomendasi", "Histori"])

if 'history' not in st.session_state:
    st.session_state.history = []

df = preprocess(load_data())

if menu == "Home":
    st.title("🎵 Musik Terpopuler")
    st.subheader("Top 10 Musik Terpopuler")
    top = get_top_popular(df)
    st.dataframe(top[['track_name', 'track_artist', 'track_popularity']])

    st.subheader("Top 5 Musik per Genre")
    genre_top = get_top_popular_by_genre(df)
    st.dataframe(genre_top[['playlist_genre', 'track_name', 'track_artist', 'track_popularity']])

elif menu == "Rekomendasi":
    st.title("🔍 Rekomendasi Musik")
    judul = st.selectbox("Pilih Judul Lagu", df['track_name'].unique()[:50])
    manual_input = st.text_input("Atau masukkan judul lagu secara manual:")

    if st.button("Rekomendasikan"):
        genre_rec, title_rec = recommend(df, judul, manual_input)

        result = {
            "input": manual_input if manual_input else judul,
            "genre_rec": genre_rec[['track_name', 'track_artist', 'playlist_genre', 'track_album_name', 'popularity_class']],
            "title_rec": title_rec[['track_name', 'track_artist', 'playlist_genre', 'track_album_name', 'popularity_class']]
        }
        st.session_state.history.append(result)

        st.subheader("🎧 Rekomendasi Berdasarkan Genre")
        st.dataframe(result['genre_rec'])

        st.subheader("🎧 Rekomendasi Berdasarkan Judul")
        st.dataframe(result['title_rec'])

elif menu == "Histori":
    st.title("📜 Histori Rekomendasi")
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history)):
            st.write(f"### 🔍 Hasil Rekomendasi Ke-{len(st.session_state.history) - i}")
            st.markdown(f"**Input Lagu:** {item['input']}")
            st.markdown("**Berdasarkan Genre:**")
            st.dataframe(item['genre_rec'])
            st.markdown("**Berdasarkan Judul:**")
            st.dataframe(item['title_rec'])
    else:
        st.info("Belum ada histori rekomendasi.")
