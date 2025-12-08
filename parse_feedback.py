import json
import pandas as pd
import sys
import os
from datetime import datetime

DATASET_PATH = 'datasetdana.csv'
FEEDBACK_FILE = 'feedback.txt'

def parse_and_append():
    if not os.path.exists(FEEDBACK_FILE):
        print("⚠️ Tidak ada file feedback.txt")
        return

    with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            content = data.get('content', '').strip()
            label = data.get('label', '').strip().lower()
            if content and label in ['positif', 'netral', 'negatif']:
                new_rows.append({
                    'userName': 'User Feedback',
                    'score': None,
                    'at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'content': content,
                    'pelabelan 3 kelas': label
                })
        except json.JSONDecodeError:
            # Abaikan baris yang bukan JSON valid
            continue

    if not new_rows:
        print("ℹ️ Tidak ada feedback valid ditemukan.")
        return

    # Muat dataset yang ada
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
    else:
        # Jika dataset tidak ada, buat dataframe kosong dengan kolom yang benar
        df = pd.DataFrame(columns=['userName', 'score', 'at', 'content', 'pelabelan 3 kelas'])

    # Tambahkan baris baru
    df_new = pd.DataFrame(new_rows)
    df = pd.concat([df, df_new], ignore_index=True)

    # Simpan kembali
    df.to_csv(DATASET_PATH, index=False)
    print(f"✅ {len(new_rows)} entri feedback berhasil ditambahkan ke {DATASET_PATH}")

if __name__ == '__main__':
    parse_and_append()