import streamlit as st
import pandas as pd
import joblib
import os
import re
from google_play_scraper import reviews, Sort
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt

# Download stopwords bahasa Indonesia
nltk.download('stopwords', quiet=True)
ID_STOPWORDS = set(stopwords.words('indonesian'))

# Fungsi preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in ID_STOPWORDS and len(w) > 2]
    return ' '.join(words)

# Fungsi scraping ulasan
def scrape_playstore_reviews(app_id, count=100):
    try:
        result, _ = reviews(
            app_id,
            lang='id',
            country='ID',
            sort=Sort.NEWEST,
            count=count
        )
        return [r['content'] for r in result if r['content']]
    except Exception as e:
        st.error(f"Gagal mengambil ulasan: {str(e)}")
        return []

# Fungsi ekstrak App ID dari link
def extract_app_id(url_or_id):
    if 'play.google.com' in url_or_id:
        # contoh: https://play.google.com/store/apps/details?id=id.dana.danabijak
        import urllib.parse
        parsed = urllib.parse.urlparse(url_or_id)
        query = urllib.parse.parse_qs(parsed.query)
        app_id = query.get('id', [None])[0]
        if app_id:
            return app_id
        else:
            st.error("Link Play Store tidak valid.")
            return None
    else:
        return url_or_id.strip()

# Latih model jika belum ada
MODEL_PATH = 'sentiment_model.pkl'
if not os.path.exists(MODEL_PATH):
    with st.spinner("Melatih model sentimen..."):
        df = pd.read_csv('datasetdana.csv')
        df = df.dropna(subset=['content', 'pelabelan 3 kelas'])
        df['clean_content'] = df['content'].apply(preprocess_text)
        X = df['clean_content']
        y = df['pelabelan 3 kelas'].str.lower()

        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', SVC(kernel='linear', probability=True))
        ])
        model.fit(X, y)
        joblib.dump(model, MODEL_PATH)

# Muat model
model = joblib.load(MODEL_PATH)

# UI Streamlit
st.set_page_config(page_title="Analisis Sentimen Ulasan Play Store", layout="centered")
st.title("🔍 Analisis Sentimen Ulasan Aplikasi Play Store")
st.markdown("Masukkan **App ID** atau **Link Play Store** untuk menganalisis sentimen pengguna.")

# Input
input_type = st.radio("Pilih jenis input:", ("App ID", "Link Play Store"))
user_input = st.text_input("Masukkan di sini:")

# Slider untuk jumlah ulasan
jumlah_ulasan = st.slider(
    "Jumlah ulasan yang akan dianalisis:",
    min_value=10,
    max_value=200,
    value=100,
    step=10,
    help="Semakin banyak ulasan, semakin lama prosesnya."
)

if st.button("Analisis Sentimen"):
    if not user_input:
        st.warning("Harap isi App ID atau Link Play Store.")
    else:
        app_id = extract_app_id(user_input)
        if not app_id:
            st.stop()

        with st.spinner(f"Mengambil ulasan dari Play Store... (App ID: {app_id})"):
            ulasan = scrape_playstore_reviews(app_id, count=jumlah_ulasan)

        if not ulasan:
            st.error("Tidak ada ulasan yang berhasil diambil.")
            st.stop()

        # Prediksi
        hasil = []
        counts = {'positif': 0, 'netral': 0, 'negatif': 0}
        for teks in ulasan[:100]:
            clean = preprocess_text(teks)
            pred = model.predict([clean])[0]
            hasil.append({"Ulasan": teks, "Sentimen": pred})
            if pred in counts:
                counts[pred] += 1

        # Tampilkan hasil
        st.subheader("📊 Hasil Analisis")
        df_hasil = pd.DataFrame(hasil)
        st.dataframe(df_hasil)

        # Visualisasi
        st.subheader("📈 Distribusi Sentimen")
        fig, ax = plt.subplots()
        ax.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', startangle=140)
        ax.axis('equal')
        st.pyplot(fig)

        # Ringkasan
        total = sum(counts.values())
        st.info(f"Total ulasan dianalisis: {total}\n\n"
                f"Positif: {counts['positif']} | Netral: {counts['netral']} | Negatif: {counts['negatif']}")

# Catatan
st.markdown("---")
st.caption("Aplikasi ini menggunakan model yang dilatih dari dataset ulasan DANA dengan 3 kelas sentimen: positif, netral, negatif.")