import streamlit as st
import pandas as pd
import zipfile
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib

# ============================
# Load dan Persiapan Dataset
# ============================

@st.cache_data
def load_data():
    # Unzip file dan baca csv
    with zipfile.ZipFile("spotify_songs.csv.zip", 'r') as zip_ref:
        zip_ref.extractall("data")
    
    file_path = [f for f in os.listdir("data") if f.endswith(".csv")][0]
    df = pd.read_csv(f"data/{file_path}")
    return df

df = load_data()

# ============================
# Sidebar Navigasi
# ============================

st.sidebar.title("Navigasi")
page = st.sidebar.radio("Menu", ["Beranda", "Rekomendasi", "Histori"])

# ============================
# Halaman Beranda
# ============================

if page == "Beranda":
    st.title("🎧 Sistem Rekomendasi Musik - Spotify")
    st.write("""
    Aplikasi ini membantumu menemukan lagu yang cocok berdasarkan karakteristik kontennya. 
    Sistem menggunakan **Content-Based Filtering** dan model **Random Forest** untuk mengklasifikasikan popularitas lagu.
    """)

    st.markdown("#### Contoh Data")
    st.dataframe(df.head(10))

# ============================
# Halaman Rekomendasi
# ============================

elif page == "Rekomendasi":
    st.title("🔍 Rekomendasi Musik")

    df = df.dropna(subset=["lyrics", "track_popularity"])
    df = df[df["track_popularity"].apply(lambda x: isinstance(x, (int, float)))]

    df["popularity_label"] = df["track_popularity"].apply(lambda x: "Tinggi" if x >= 60 else "Rendah")

    tfidf = TfidfVectorizer(stop_words="english", max_features=500)
    tfidf_matrix = tfidf.fit_transform(df["lyrics"].astype(str))

    X = tfidf_matrix
    y = df["popularity_label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    st.markdown("### Masukkan Lirik Lagu")
    input_lyrics = st.text_area("Lirik", "")

    if st.button("Rekomendasikan"):
        if input_lyrics:
            input_vec = tfidf.transform([input_lyrics])
            prediction = rf.predict(input_vec)[0]

            st.success(f"🎵 Lagu ini diprediksi memiliki popularitas **{prediction}**")
        else:
            st.warning("Mohon masukkan lirik lagu terlebih dahulu.")

# ============================
# Halaman Histori
# ============================

elif page == "Histori":
    st.title("📜 Histori Evaluasi Model")

    df = df.dropna(subset=["lyrics", "track_popularity"])
    df = df[df["track_popularity"].apply(lambda x: isinstance(x, (int, float)))]

    df["popularity_label"] = df["track_popularity"].apply(lambda x: "Tinggi" if x >= 60 else "Rendah")

    tfidf = TfidfVectorizer(stop_words="english", max_features=500)
    tfidf_matrix = tfidf.fit_transform(df["lyrics"].astype(str))

    X = tfidf_matrix
    y = df["popularity_label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.text(cm)

    st.markdown("### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=False)
    st.text(report)
