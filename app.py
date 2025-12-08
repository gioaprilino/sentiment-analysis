import streamlit as st
import pandas as pd
import joblib
import os
import re
import urllib.parse
from google_play_scraper import reviews, Sort
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
import requests

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

# Ekstrak App ID dari URL
def extract_app_id(url_or_id):
    if 'play.google.com' in url_or_id:
        parsed = urllib.parse.urlparse(url_or_id)
        query = urllib.parse.parse_qs(parsed.query)
        app_id = query.get('id', [None])[0]
        if app_id:
            return app_id
        else:
            st.error("🔗 Link Play Store tidak valid. Pastikan mengandung parameter `id`.")
            return None
    else:
        return url_or_id.strip()

# Scrape ulasan
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
        st.error(f"⚠️ Gagal mengambil ulasan: {str(e)}")
        return []

# Latih model (jika belum ada)
MODEL_PATH = 'sentiment_model.pkl'
DATASET_PATH = 'datasetdana.csv'

if not os.path.exists(MODEL_PATH) and os.path.exists(DATASET_PATH):
    with st.spinner("🧠 Melatih model sentimen dari datasetdana.csv..."):
        df = pd.read_csv(DATASET_PATH)
        df = df.dropna(subset=['content', 'pelabelan 3 kelas'])
        df['clean_content'] = df['content'].apply(preprocess_text)
        X = df['clean_content']
        y = df['pelabelan 3 kelas'].str.lower().str.strip()

        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', SVC(kernel='linear', probability=True))
        ])
        model.fit(X, y)
        joblib.dump(model, MODEL_PATH)

# UI Utama
st.set_page_config(page_title="Analisis Sentimen Play Store", layout="centered")
st.title("🔍 Analisis Sentimen Ulasan Aplikasi Play Store")
st.markdown("Masukkan **App ID** atau **Link Play Store**, lalu analisis sentimen ulasan dalam Bahasa Indonesia.")

# Input pengguna
input_type = st.radio("Pilih jenis input:", ("App ID", "Link Play Store"), horizontal=True)
user_input = st.text_input("Masukkan di sini:")

jumlah_ulasan = st.slider(
    "Jumlah ulasan yang akan dianalisis:",
    min_value=10,
    max_value=200,
    value=100,
    step=10,
    help="Lebih sedikit = lebih cepat."
)

if st.button("🚀 Analisis Sentimen"):
    if not user_input.strip():
        st.warning("Harap isi App ID atau Link Play Store.")
        st.stop()

    app_id = extract_app_id(user_input)
    if not app_id:
        st.stop()

    with st.spinner(f"Mengambil {jumlah_ulasan} ulasan dari Play Store..."):
        ulasan = scrape_playstore_reviews(app_id, count=jumlah_ulasan)

    if not ulasan:
        st.error("Tidak ada ulasan yang berhasil diambil.")
        st.stop()

    # Muat model
    if not os.path.exists(MODEL_PATH):
        st.error("Model belum dilatih. Pastikan file `datasetdana.csv` tersedia.")
        st.stop()
    model = joblib.load(MODEL_PATH)

    # Prediksi
    hasil = []
    counts = {'positif': 0, 'netral': 0, 'negatif': 0}
    for teks in ulasan:
        clean = preprocess_text(teks)
        pred = model.predict([clean])[0]
        hasil.append({"Ulasan": teks, "Sentimen": pred})
        if pred in counts:
            counts[pred] += 1

    # Simpan hasil ke session state agar bisa dikoreksi
    st.session_state['hasil'] = hasil
    st.session_state['counts'] = counts

# Tampilkan hasil jika ada
if 'hasil' in st.session_state:
    hasil = st.session_state['hasil']
    counts = st.session_state['counts']
    df_hasil = pd.DataFrame(hasil)

    st.subheader("📊 Hasil Analisis Sentimen")
    
    # Tambahkan kolom 'selected' untuk multi-select
    df_hasil['selected'] = False

    # Tampilkan tabel (tanpa kolom 'selected' di tampilan akhir)
    st.write("Centang ulasan di bawah yang ingin Anda koreksi:")

    # Buat list pilihan
    selected_indices = st.multiselect(
        "Pilih nomor ulasan untuk dikoreksi:",
        options=df_hasil.index.tolist(),
        format_func=lambda x: f"{x+1}. {df_hasil.loc[x, 'Ulasan'][:50]}..."
    )

    # Dropdown untuk label koreksi
    if selected_indices:
        correct_label = st.selectbox("Label yang benar untuk ulasan terpilih:", ["positif", "netral", "negatif"])
        if st.button("💾 Simpan Koreksi Terpilih"):
            # Simpan hanya yang dipilih
            for idx in selected_indices:
                teks = df_hasil.loc[idx, 'Ulasan']
                new_row = pd.DataFrame([{
                    'userName': 'User Feedback',
                    'score': None,
                    'at': pd.Timestamp.now(),
                    'content': teks,
                    'pelabelan 3 kelas': correct_label
                }])
                new_row.to_csv(DATASET_PATH, mode='a', header=False, index=False)
            st.success(f"✅ {len(selected_indices)} ulasan berhasil disimpan ke datasetdana.csv!")

    # Tampilkan tabel hasil (tanpa kolom 'selected')
    st.dataframe(df_hasil[['Ulasan', 'Sentimen']])

    # Visualisasi
    st.subheader("📈 Distribusi Sentimen")
    fig, ax = plt.subplots()
    ax.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)

    total = sum(counts.values())
    st.info(f"**Total ulasan:** {total} | Positif: {counts['positif']} | Netral: {counts['netral']} | Negatif: {counts['negatif']}")

    # Estimasi keakuratan (berdasarkan confidence score dari SVM)
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        confidences = []
        for teks in df_hasil['Ulasan']:
            clean = preprocess_text(teks)
            proba = model.predict_proba([clean])[0]
            confidences.append(proba.max())
        avg_conf = sum(confidences) / len(confidences)
        st.metric(
            label="🔍 Estimasi Keakuratan Rata-Rata",
            value=f"{avg_conf:.1%}",
            help="Berdasarkan confidence score model SVM (semakin tinggi, semakin yakin model terhadap prediksinya)"
        )

# Tombol latih ulang model
st.markdown("---")
if st.button("🔄 Latih Ulang Model dengan Data Baru"):
    if not os.path.exists(DATASET_PATH):
        st.error("File datasetdana.csv tidak ditemukan!")
    else:
        with st.spinner("Memuat dataset dan melatih ulang model..."):
            df = pd.read_csv(DATASET_PATH)
            df = df.dropna(subset=['content', 'pelabelan 3 kelas'])
            df['clean_content'] = df['content'].apply(preprocess_text)
            X = df['clean_content']
            y = df['pelabelan 3 kelas'].str.lower().str.strip()

            model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('clf', SVC(kernel='linear', probability=True))
            ])
            model.fit(X, y)
            joblib.dump(model, MODEL_PATH)
        st.success("✅ Model berhasil dilatih ulang dengan data terbaru!")

def create_github_issue(title, body):
    url = f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/issues"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    data = {"title": title, "body": body}
    response = requests.post(url, json=data, headers=headers)
    return response.status_code == 201

# Kirim feedback ke GitHub Issue
if st.button("📤 Kirim Feedback ke GitHub (untuk pelatihan ulang)"):
    import requests
    import json
    GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
    GITHUB_USER = st.secrets.get("GITHUB_USER", "")
    REPO_NAME = st.secrets.get("REPO_NAME", "")

    if not all([GITHUB_TOKEN, GITHUB_USER, REPO_NAME]):
        st.error("GitHub Secrets belum dikonfigurasi.")
    else:
        # Ambil data koreksi (misal: dari session state)
        feedback_lines = []
        for idx in selected_indices:
            feedback_lines.append(
                json.dumps({
                    "content": df_hasil.loc[idx, "Ulasan"],
                    "label": correct_label
                }, ensure_ascii=False)
            )

        body = "\n".join(feedback_lines)
        url = f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/issues"
        res = requests.post(url, json={
            "title": "[FEEDBACK] Koreksi Sentimen dari Streamlit",
            "body": body,
            "labels": ["feedback"]
        }, headers={"Authorization": f"token {GITHUB_TOKEN}"})

        if res.status_code == 201:
            st.success("✅ Feedback berhasil dikirim ke GitHub! Model akan dilatih ulang otomatis.")
        else:
            st.error("❌ Gagal mengirim ke GitHub.")    

st.caption("💡 Tips: Setelah menyimpan beberapa koreksi, latih ulang model agar akurasinya meningkat!")