# train_model.py
import pandas as pd
import numpy as np
import re
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# Setup
nltk.download('stopwords', quiet=True)
ID_STOPWORDS = set(stopwords.words('indonesian'))
DATASET_PATH = 'datasetdana.csv'
MODEL_PATH = 'sentiment_model.pkl'
METRICS_PATH = 'metrics.json'
LABEL_COL = 'pelabelan 3 kelas'

# Preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in ID_STOPWORDS and len(w) > 2]
    return ' '.join(words)

# Load & prepare data
df = pd.read_csv(DATASET_PATH, on_bad_lines='skip')
df = df.dropna(subset=['content', LABEL_COL])
df['label'] = df[LABEL_COL].astype(str).str.lower().str.strip()
df['clean_content'] = df['content'].apply(preprocess_text)

# Filter valid labels only
valid_labels = ['positif', 'netral', 'negatif']
df = df[df['label'].isin(valid_labels)]

print(f"📊 Menggunakan {len(df)} ulasan untuk pelatihan.")

# Split data (80% train, 20% test)
X = df['clean_content']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Pipeline & Hyperparameter Tuning
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', SVC(probability=True, random_state=42))
])

param_grid = {
    'tfidf__max_features': [3000, 5000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.1, 1, 10],
    'clf__kernel': ['linear']
    # Gunakan 'linear' untuk kecepatan & interpretasi (SVM linear cocok untuk teks)
}

print("🔍 Memulai hyperparameter tuning...")
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluasi
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred, labels=valid_labels)

# Simpan metrik
metrics = {
    "accuracy": float(acc),
    "best_params": {str(k): v for k, v in grid.best_params_.items()},
    "classification_report": report
}
with open(METRICS_PATH, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

# Simpan model
joblib.dump(best_model, MODEL_PATH)

# === Simpan visualisasi EDA & evaluasi ===
os.makedirs('assets', exist_ok=True)

# EDA: distribusi label
plt.figure(figsize=(6, 4))
df['label'].value_counts().plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Distribusi Label Sentimen')
plt.ylabel('Jumlah')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('assets/eda_label_distribution.png')
plt.close()

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=valid_labels, yticklabels=valid_labels, cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('assets/confusion_matrix.png')
plt.close()

print(f"✅ Pelatihan selesai!")
print(f"📊 Akurasi: {acc:.2%}")
print(f"📁 Model disimpan di: {MODEL_PATH}")
print(f"📈 Metrik disimpan di: {METRICS_PATH}")
print(f"🖼️ Visualisasi disimpan di folder: assets/")