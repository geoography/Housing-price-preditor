# -*- coding: utf-8 -*-
"""
train_and_save.py
─────────────────
Jalankan script ini di Colab (setelah training selesai) untuk
menyimpan model dan metadata ke file .pkl yang akan dipakai Flask.

Pastikan variabel berikut sudah ada di environment Colab:
  - rf_model   : trained RandomForestRegressor
  - X_train    : training features (untuk ambil nama kolom)
"""

import joblib
import json

# ── 1. Simpan model Random Forest ────────────────────────────────────
joblib.dump(lr_model, "lr_model.pkl")
print("✅  lr_model.pkl tersimpan")

# ── 2. Simpan daftar nama fitur (urutan kolom harus sama saat predict) 
feature_names = X_train.columns.tolist()
with open("feature_names.json", "w") as f:
    json.dump(feature_names, f)
print("✅  feature_names.json tersimpan")
print(f"    Fitur ({len(feature_names)}): {feature_names}")

# ── 3. (Opsional) Download ke lokal kalau pakai Colab ────────────────
try:
    from google.colab import files
    files.download("rf_model.pkl")
    files.download("feature_names.json")
    print("✅  File didownload ke komputer lokal")
except ImportError:
    print("ℹ️  Bukan environment Colab — skip download otomatis")
