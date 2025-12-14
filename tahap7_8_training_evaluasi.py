import pandas as pd
import re
import nltk
import joblib
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# ==============================================================================
# KONFIGURASI & PREPROCESSING (Sama seperti sebelumnya)
# ==============================================================================
print("⚙️  Konfigurasi awal...")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ID_STOPWORDS = set(stopwords.words('indonesian'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in ID_STOPWORDS and len(w) > 2]
    return ' '.join(words)

# 1. Load Data
print("📂 Membaca dataset...")
try:
    df = pd.read_csv('datasetdana.csv')
except FileNotFoundError:
    print("❌ Error: File 'datasetdana.csv' tidak ditemukan.")
    exit()

df = df.dropna(subset=['content', 'pelabelan 3 kelas'])
df['clean_content'] = df['content'].apply(preprocess_text)

X = df['clean_content']
y = df['pelabelan 3 kelas'].str.lower().str.strip()

# 2. Split Data (Tahap 6 - Diulang agar variabel tersedia)
print("✂️  Split Data (80% Train, 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================================================================
# TAHAP 7: MODEL SELECTION & TRAINING
# ==============================================================================
print("\n🤖 MEMULAI TAHAP 7: TRAINING MODEL...")

# Kita menggunakan algoritma SVM (Support Vector Machine) seperti di app.py
# Pipeline menggabungkan TF-IDF (pengubah kata jadi angka) dan SVM
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', SVC(kernel='linear', probability=True))
])

# Melatih model dengan data training (X_train dan y_train)
pipeline.fit(X_train, y_train)
print("✅ Model berhasil dilatih!")

# (Opsional) Simpan model hasil training ini
joblib.dump(pipeline, 'model_uas_final.pkl')
print("💾 Model disimpan sebagai 'model_uas_final.pkl'")

# ==============================================================================
# TAHAP 8: MODEL EVALUATION
# ==============================================================================
print("\n📝 MEMULAI TAHAP 8: EVALUASI MODEL...")

# Menguji model dengan data testing (X_test)
y_pred = pipeline.predict(X_test)

# Menghitung metrik evaluasi
akurasi = accuracy_score(y_test, y_pred)
laporan = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("-" * 50)
print(f"🎯 AKURASI MODEL: {akurasi:.2%}")
print("-" * 50)
print("\n📊 Laporan Klasifikasi Rinci (Classification Report):")
print(laporan)
print("-" * 50)

# Menampilkan Confusion Matrix (Tabel Kebenaran)
print("\nkinerja detail per kelas (Confusion Matrix):")
# Baris = Kunci Jawaban (Asli), Kolom = Jawaban Model (Prediksi)
labels = pipeline.classes_
print(f"{'':<10} | {' | '.join([f'Pred {l}' for l in labels])}")
for i, label_asli in enumerate(labels):
    row_data = ' | '.join([f"{num:>8}" for num in cm[i]])
    print(f"Asli {label_asli:<5} | {row_data}")

# (Opsional) Visualisasi Confusion Matrix jika menggunakan Jupyter/bisa menampilkan gambar
try:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix - Evaluasi Model SVM")
    plt.savefig('hasil_evaluasi_confusion_matrix.png') # Simpan gambar
    print("\n🖼️  Gambar Confusion Matrix disimpan sebagai 'hasil_evaluasi_confusion_matrix.png'")
except Exception as e:
    print("\n(Visualisasi gambar dilewati)")