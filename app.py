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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# KONFIGURASI & PREPROCESSING
# ==========================================

# Download stopwords bahasa Indonesia
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

ID_STOPWORDS = set(stopwords.words('indonesian'))

MODEL_PATH = 'sentiment_model.pkl'
DATASET_PATH = 'datasetdana.csv'

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

# ==========================================
# UI UTAMA
# ==========================================
st.set_page_config(page_title="Analisis Sentimen Play Store (UAS)", layout="wide")

st.title("🔍 Analisis Sentimen Ulasan Aplikasi Play Store")
st.markdown("""
Aplikasi ini melakukan **Analisis Sentimen** menggunakan metode **Support Vector Machine (SVM)**.
Menu di sebelah kiri (sidebar) digunakan untuk **Melatih & Mengevaluasi Model (Tahap 6-8)**.
""")

# ==========================================
# SIDEBAR: PELATIHAN & EVALUASI MODEL
# ==========================================
with st.sidebar:
    st.header("⚙️ Panel Pengembang (UAS)")
    st.info("Gunakan menu ini untuk melatih ulang model dan melihat hasil evaluasi (Split Data).")
    
    if st.button("🔄 Latih & Evaluasi Model (Split Data)"):
        if not os.path.exists(DATASET_PATH):
            st.error(f"File {DATASET_PATH} tidak ditemukan!")
        else:
            with st.spinner("⏳ Sedang memproses Tahap 6, 7, dan 8..."):
                # 1. Load Data
                df = pd.read_csv(DATASET_PATH)
                df = df.dropna(subset=['content', 'pelabelan 3 kelas'])
                
                # 2. Preprocessing
                df['clean_content'] = df['content'].apply(preprocess_text)
                X = df['clean_content']
                y = df['pelabelan 3 kelas'].str.lower().str.strip()

                # --- TAHAP 6: SPLIT DATA ---
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # --- TAHAP 7: TRAINING ---
                model = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=5000)),
                    ('clf', SVC(kernel='linear', probability=True))
                ])
                model.fit(X_train, y_train)
                
                # Simpan model
                joblib.dump(model, MODEL_PATH)
                
                # --- TAHAP 8: EVALUASI ---
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                cm = confusion_matrix(y_test, y_pred)
                
                # Simpan hasil ke session state untuk ditampilkan
                st.session_state['eval_results'] = {
                    'accuracy': acc,
                    'report': report,
                    'cm': cm,
                    'classes': model.classes_,
                    'n_train': len(X_train),
                    'n_test': len(X_test)
                }
            st.success("✅ Model berhasil dilatih & dievaluasi!")

    # Tampilkan Hasil Evaluasi di Sidebar jika ada
    if 'eval_results' in st.session_state:
        res = st.session_state['eval_results']
        st.divider()
        st.subheader("📊 Hasil Evaluasi Terkini")
        
        # Metric Akurasi
        st.metric("Akurasi Model", f"{res['accuracy']:.2%}")
        
        st.write(f"**Data Training:** {res['n_train']} baris")
        st.write(f"**Data Testing:** {res['n_test']} baris")
        
        # Tampilkan Report Singkat
        st.markdown("---")
        st.caption("Detail per Kelas:")
        report_df = pd.DataFrame(res['report']).transpose()
        st.dataframe(report_df.style.format("{:.2f}"), height=200)

# ==========================================
# BAGIAN TENGAH: INFERENCE (PREDIKSI)
# ==========================================

# Cek apakah model sudah ada
if not os.path.exists(MODEL_PATH):
    st.warning("⚠️ Model belum tersedia. Silakan klik tombol 'Latih & Evaluasi Model' di sidebar sebelah kiri terlebih dahulu.")
else:
    # Input Pengguna
    col1, col2 = st.columns([2, 1])
    
    with col1:
        input_type = st.radio("Pilih jenis input:", ("App ID", "Link Play Store"), horizontal=True)
        user_input = st.text_input("Masukkan Link / ID Aplikasi:", placeholder="Contoh: id.dana")
    
    with col2:
        jumlah_ulasan = st.slider("Jumlah ulasan:", 10, 200, 100, step=10)
        st.write("") # Spacer
        analyze_btn = st.button("🚀 Analisis Sentimen", type="primary", use_container_width=True)

    if analyze_btn:
        if not user_input.strip():
            st.warning("Harap isi App ID atau Link Play Store.")
            st.stop()

        app_id = extract_app_id(user_input)
        if not app_id:
            st.stop()

        # Load Model
        model = joblib.load(MODEL_PATH)

        with st.spinner(f"Mengambil {jumlah_ulasan} ulasan..."):
            ulasan = scrape_playstore_reviews(app_id, count=jumlah_ulasan)

        if not ulasan:
            st.error("Tidak ada ulasan yang berhasil diambil.")
        else:
            # Prediksi
            hasil = []
            counts = {'positif': 0, 'netral': 0, 'negatif': 0}
            
            progress_bar = st.progress(0)
            for i, teks in enumerate(ulasan):
                clean = preprocess_text(teks)
                pred = model.predict([clean])[0]
                hasil.append({"Ulasan": teks, "Sentimen": pred})
                if pred in counts:
                    counts[pred] += 1
                progress_bar.progress((i + 1) / len(ulasan))
            
            progress_bar.empty()

            # Simpan ke session state
            st.session_state['hasil'] = hasil
            st.session_state['counts'] = counts

    # Tampilkan Hasil Prediksi
    if 'hasil' in st.session_state:
        hasil = st.session_state['hasil']
        counts = st.session_state['counts']
        df_hasil = pd.DataFrame(hasil)

        st.divider()
        st.subheader("📈 Visualisasi Hasil")
        
        col_chart, col_stat = st.columns([1, 1])
        
        with col_chart:
            # Pie Chart Matplotlib
            fig, ax = plt.subplots(figsize=(4, 4))
            colors = ['#66b3ff', '#99ff99', '#ff9999'] # Biru, Hijau, Merah
            # Pastikan urutan warna sesuai label keys
            ax.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', startangle=140, colors=colors)
            ax.axis('equal')
            st.pyplot(fig, use_container_width=False)

        with col_stat:
            st.info(f"**Total Ulasan Dianalisis:** {sum(counts.values())}")
            st.success(f"Positif: {counts['positif']}")
            st.warning(f"Netral: {counts['netral']}")
            st.error(f"Negatif: {counts['negatif']}")

        st.subheader("📝 Tabel Detail Ulasan")
        st.dataframe(df_hasil, use_container_width=True)
        
        # Fitur Koreksi Data (Feedback Loop)
        with st.expander("🛠️ Koreksi Prediksi (Untuk Menambah Data Latih)"):
            st.write("Jika ada prediksi salah, pilih ulasan dan simpan label yang benar.")
            
            # Pilihan Multi-select
            selected_indices = st.multiselect(
                "Pilih ulasan yang ingin dikoreksi:",
                options=df_hasil.index.tolist(),
                format_func=lambda x: f"{df_hasil.loc[x, 'Sentimen'].upper()} - {df_hasil.loc[x, 'Ulasan'][:75]}..."
            )
            
            if selected_indices:
                correct_label = st.selectbox("Label yang benar:", ["positif", "netral", "negatif"])
                if st.button("💾 Simpan Koreksi ke Dataset"):
                    new_data = []
                    for idx in selected_indices:
                        teks = df_hasil.loc[idx, 'Ulasan']
                        new_data.append({
                            'userName': 'User Feedback',
                            'score': None,
                            'at': pd.Timestamp.now(),
                            'content': teks,
                            'pelabelan 3 kelas': correct_label
                        })
                    
                    df_new = pd.DataFrame(new_data)
                    df_new.to_csv(DATASET_PATH, mode='a', header=False, index=False)
                    st.success(f"Berhasil menyimpan {len(new_data)} data baru! Silakan latih ulang model di Sidebar.")