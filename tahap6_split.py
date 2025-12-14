import pandas as pd
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords

# --- Konfigurasi (Sama seperti di app.py) ---
# Pastikan library NLTK stopwords sudah terdownload
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
    # Menghapus kata yang kurang dari 3 huruf dan stopwords
    words = [w for w in text.split() if w not in ID_STOPWORDS and len(w) > 2]
    return ' '.join(words)

# --- MULAI TAHAP 6 ---

# 1. Load Data
# Pastikan file datasetdana.csv ada di folder yang sama dengan script ini
print("📂 Membaca dataset...")
try:
    df = pd.read_csv('datasetdana.csv')
    print(f"   -> Berhasil dimuat: {len(df)} baris data.")
except FileNotFoundError:
    print("❌ Error: File 'datasetdana.csv' tidak ditemukan.")
    exit()

# 2. Cleaning Data (Hapus data kosong)
df = df.dropna(subset=['content', 'pelabelan 3 kelas'])

# 3. Preprocessing
print("🧹 Melakukan preprocessing teks...")
df['clean_content'] = df['content'].apply(preprocess_text)

# Definisikan Fitur (X) dan Target (y)
X = df['clean_content']
y = df['pelabelan 3 kelas'].str.lower().str.strip()

# 4. SPLIT DATA (80% Train, 20% Test)
print("✂️  Melakukan Split Data (80:20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- OUTPUT UNTUK LAPORAN ---
print("\n" + "="*40)
print("✅ HASIL TAHAP 6: SPLIT DATA SELESAI")
print("="*40)
print(f"Total Data Bersih : {len(df)}")
print(f"Jumlah Data Training (Latih) : {len(X_train)} ({len(X_train)/len(df):.0%})")
print(f"Jumlah Data Testing (Uji)    : {len(X_test)} ({len(X_test)/len(df):.0%})")
print("="*40)

print("\nDistribusi Label di Data Training:")
print(y_train.value_counts())