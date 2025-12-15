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
import altair as alt # Library grafik interaktif
import requests
import json

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Analisis Sentimen DANA",
    page_icon="📊",
    layout="wide" # Menggunakan layout lebar
)

# --- 2. SETUP & FUNGSI BANTUAN ---

# Download stopwords bahasa Indonesia
nltk.download('stopwords', quiet=True)
ID_STOPWORDS = set(stopwords.words('indonesian'))

# Fungsi preprocessing teks
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in ID_STOPWORDS and len(w) > 2]
    return ' '.join(words)

# Ekstrak App ID dari URL Play Store
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

# Scrape ulasan dari Google Play
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

# Path File
MODEL_PATH = 'sentiment_model.pkl'
DATASET_PATH = 'datasetdana.csv'
METRICS_PATH = 'metrics.json'

# Load Model (Cached supaya performa cepat)
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

# Pelatihan Otomatis jika model belum ada
if not os.path.exists(MODEL_PATH) and os.path.exists(DATASET_PATH):
    with st.spinner("🧠 Sedang melatih model awal..."):
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

# --- 3. SIDEBAR (INPUT & ADMIN) ---

with st.sidebar:
    st.header("⚙️ Konfigurasi")
    input_type = st.radio("Jenis Input:", ("App ID", "Link Play Store"))
    user_input = st.text_input("Masukkan ID / Link:", value="id.dana")
    
    jumlah_ulasan = st.slider(
        "Jumlah Ulasan:",
        min_value=10,
        max_value=2000,
        value=500,
        step=50,
        help="Semakin banyak ulasan, semakin lama prosesnya."
    )
    
    tombol_analisis = st.button("🚀 Mulai Analisis", type="primary")
    
    st.markdown("---")
    st.markdown("**Menu Admin:**")
    
    # Tombol Latih Ulang
    if st.button("🔄 Latih Ulang Model"):
        if not os.path.exists(DATASET_PATH):
            st.error("Dataset tidak ditemukan!")
        else:
            with st.spinner("Melatih ulang model dengan data terbaru..."):
                # Load ulang dataset
                df = pd.read_csv(DATASET_PATH)
                df = df.dropna(subset=['content', 'pelabelan 3 kelas'])
                df['clean_content'] = df['content'].apply(preprocess_text)
                X = df['clean_content']
                y = df['pelabelan 3 kelas'].str.lower().str.strip()

                # Pipeline pelatihan
                model = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=5000)),
                    ('clf', SVC(kernel='linear', probability=True))
                ])
                model.fit(X, y)
                joblib.dump(model, MODEL_PATH)
                
                # Hapus cache agar model baru termuat
                load_model.clear()
            st.success("✅ Model berhasil diperbarui!")

    # --- FITUR LAPORAN KLASIFIKASI (YANG DIKEMBALIKAN) ---
    if os.path.exists(METRICS_PATH):
        st.markdown("---")
        st.caption("📊 Performa Model Saat Ini")
        try:
            with open(METRICS_PATH, 'r') as f:
                metrics = json.load(f)
            
            # Tampilkan Akurasi Utama
            st.metric("Akurasi Test", f"{metrics['accuracy']:.1%}")
            
            # Tampilkan Detail dalam Expander (Buka/Tutup) agar rapi
            with st.expander("📝 Lihat Detail Klasifikasi"):
                st.write("**Parameter Terbaik:**")
                if 'best_params' in metrics:
                    for k, v in metrics['best_params'].items():
                        # Membersihkan nama parameter agar lebih enak dibaca
                        param_name = k.split('__')[1] if '__' in k else k
                        st.caption(f"- {param_name}: `{v}`")
                
                st.divider()
                st.write("**Laporan per Kelas:**")
                if 'classification_report' in metrics:
                    # Membuat dataframe dari classification_report
                    report_df = pd.DataFrame(metrics['classification_report']).transpose()
                    # Menampilkan tabel kecil
                    st.dataframe(
                        report_df.style.format(precision=2), 
                        width='stetch',
                    )
        except Exception as e:
            st.caption("Gagal memuat metrik.")

# --- 4. LOGIKA UTAMA APLIKASI ---

st.title("📊 Dashboard Analisis Sentimen Play Store")
st.markdown("""
Aplikasi ini menganalisis ulasan aplikasi secara otomatis menggunakan **Machine Learning**.
Masukkan ID Aplikasi atau Link Play Store di sidebar untuk memulai.
""")

if tombol_analisis:
    if not user_input.strip():
        st.warning("⚠️ Harap isi App ID atau Link Play Store.")
        st.stop()

    app_id = extract_app_id(user_input)
    if not app_id:
        st.stop()

    # Scraping Data
    with st.spinner(f"🔍 Sedang mengambil {jumlah_ulasan} ulasan terbaru..."):
        ulasan = scrape_playstore_reviews(app_id, count=jumlah_ulasan)

    if not ulasan:
        st.error("❌ Tidak ada ulasan yang berhasil diambil.")
        st.stop()

    # Load Model
    model = load_model()
    if not model:
        st.error("Model belum siap. Silakan latih ulang di sidebar.")
        st.stop()

    # Prediksi Sentimen
    hasil = []
    counts = {'positif': 0, 'netral': 0, 'negatif': 0}
    
    progress_bar = st.progress(0)
    total_ulasan = len(ulasan)
    
    for i, teks in enumerate(ulasan):
        clean = preprocess_text(teks)
        pred = model.predict([clean])[0]
        # Hitung confidence score
        proba = model.predict_proba([clean])[0]
        confidence = proba.max()
        
        hasil.append({
            "Ulasan": teks, 
            "Sentimen": pred,
            "Confidence": confidence
        })
        if pred in counts:
            counts[pred] += 1
        
        # Update progress bar
        if i % (total_ulasan // 10 + 1) == 0:
            progress_bar.progress((i + 1) / total_ulasan)
            
    progress_bar.progress(1.0)
    
    # Simpan hasil ke session state
    st.session_state['hasil'] = hasil
    st.session_state['counts'] = counts

# --- 5. TAMPILAN HASIL (METRIK & GRAFIK) ---

if 'hasil' in st.session_state:
    hasil = st.session_state['hasil']
    counts = st.session_state['counts']
    df_hasil = pd.DataFrame(hasil)

    # 1. KPI Metrics (Angka Besar)
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    
    total = sum(counts.values())
    
    with col1:
        st.metric("Total Ulasan", total)
    with col2:
        st.metric("Positif", counts['positif'], delta=f"{(counts['positif']/total)*100:.1f}%", delta_color="normal")
    with col3:
        st.metric("Netral", counts['netral'], delta_color="off")
    with col4:
        st.metric("Negatif", counts['negatif'], delta=f"-{(counts['negatif']/total)*100:.1f}%", delta_color="inverse")

    # 2. Grafik & Tabel Berdampingan
    col_chart, col_table = st.columns([1, 2]) # Rasio lebar 1:2

    with col_chart:
        st.subheader("📈 Distribusi Sentimen")
        
        # Grafik Donut Interaktif dengan Altair
        source = pd.DataFrame({
            'Kategori': list(counts.keys()),
            'Jumlah': list(counts.values())
        })
        
        # Definisi Warna (Hijau, Abu, Merah)
        domain = ['positif', 'netral', 'negatif']
        range_ = ['#28a745', '#6c757d', '#dc3545'] 

        base = alt.Chart(source).encode(
            theta=alt.Theta("Jumlah", stack=True)
        )

        pie = base.mark_arc(outerRadius=120, innerRadius=80).encode(
            color=alt.Color("Kategori", scale=alt.Scale(domain=domain, range=range_), legend=alt.Legend(title="Sentimen")),
            order=alt.Order("Jumlah", sort="descending"),
            tooltip=["Kategori", "Jumlah", alt.Tooltip("Jumlah", format=".0f")]
        )

        text = base.mark_text(radius=140).encode(
            text=alt.Text("Jumlah", format=".0f"),
            order=alt.Order("Jumlah", sort="descending"),
            color=alt.value("black")  
        )

        st.altair_chart(pie + text, width='stretch')
        
        # Tampilkan Estimasi Akurasi
        if 'Confidence' in df_hasil.columns:
            avg_conf = df_hasil['Confidence'].mean()
            st.caption(f"🤖 Tingkat Keyakinan Model Rata-rata: **{avg_conf:.1%}**")
        else:
            st.caption("🤖 Confidence score tidak tersedia untuk data ini.")

    with col_table:
        st.subheader("📝 Detail Ulasan")
        
        st.info("Centang ulasan di bawah untuk mengoreksi sentimen, lalu klik 'Simpan Koreksi'.")
        
        # Data Editor (Tabel interaktif)
        df_display = df_hasil.copy()
        df_display['Pilih'] = False
        
        edited_df = st.data_editor(
            df_display[['Pilih', 'Sentimen', 'Ulasan']],
            column_config={
                "Pilih": st.column_config.CheckboxColumn(
                    "Koreksi?",
                    help="Pilih untuk mengoreksi ulasan ini",
                    default=False,
                ),
                "Sentimen": st.column_config.TextColumn(
                    "Sentimen",
                    width="medium",
                    validate="^(positif|netral|negatif)$"
                ),
                "Ulasan": st.column_config.TextColumn(
                    "Isi Ulasan",
                    width="large"
                )
            },
            disabled=["Ulasan", "Sentimen"], # Edit teks via selectbox di bawah
            hide_index=True,
            width='stretch',
            height=400
        )

        # Logika Penyimpanan Koreksi
        selected_rows = edited_df[edited_df['Pilih'] == True]
        
        if not selected_rows.empty:
            st.write(f"**{len(selected_rows)} ulasan dipilih.**")
            col_corr1, col_corr2 = st.columns([2, 1])
            with col_corr1:
                correct_label = st.selectbox("Ubah sentimen menjadi:", ["positif", "netral", "negatif"])
            with col_corr2:
                if st.button("💾 Simpan Koreksi"):
                    for idx, row in selected_rows.iterrows():
                        teks = row['Ulasan']
                        new_row = pd.DataFrame([{
                            'userName': 'User Feedback',
                            'score': None,
                            'at': pd.Timestamp.now(),
                            'content': teks,
                            'pelabelan 3 kelas': correct_label
                        }])
                        # Append ke CSV
                        new_row.to_csv(DATASET_PATH, mode='a', header=False, index=False)
                    
                    st.success(f"✅ Berhasil menyimpan {len(selected_rows)} data koreksi!")
                    st.rerun()

    # --- 6. BAGIAN FEEDBACK (OPSIONAL) ---
    st.divider()
    with st.expander("🛠️ Kirim Laporan ke Developer (GitHub Issues)"):
        st.write("Jika menemukan banyak kesalahan prediksi, kirim laporan ini untuk perbaikan model.")
        if st.button("📤 Kirim Laporan Feedback"):
            GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
            GITHUB_USER = st.secrets.get("GITHUB_USER", "")
            REPO_NAME = st.secrets.get("REPO_NAME", "")
            
            if not all([GITHUB_TOKEN, GITHUB_USER, REPO_NAME]):
                st.error("GitHub Secrets belum dikonfigurasi di `.streamlit/secrets.toml`")
            elif selected_rows.empty:
                st.warning("Pilih dulu ulasan yang ingin dilaporkan di tabel atas.")
            else:
                feedback_body = []
                for _, row in selected_rows.iterrows():
                    feedback_body.append(json.dumps({
                        "text": row['Ulasan'],
                        "corrected_label": correct_label
                    }, ensure_ascii=False))
                
                body_str = "\n".join(feedback_body)
                url = f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/issues"
                res = requests.post(url, json={
                    "title": "[FEEDBACK] Koreksi Sentimen User",
                    "body": body_str,
                    "labels": ["feedback", "improvement"]
                }, headers={"Authorization": f"token {GITHUB_TOKEN}"})

                if res.status_code == 201:
                    st.success("Laporan terkirim ke GitHub!")
                else:
                    st.error(f"Gagal mengirim. Status: {res.status_code}")