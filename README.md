# 🏠 Housing Price Predictor

Final project GDGoC ML — prediksi harga rumah menggunakan Machine Learning (Linear Regression).
Link demo: https://joprii25-housing-price-predictor.hf.space

---

## ⚙️ Setup

1. Clone/download project ini, dengan struktur:
   ```
   project/
   ├── app.py
   ├── lr_model.pkl
   ├── feature_names.json
   ├── requirements.txt
   ├── train_and_save.py
   └── templates/
       └── index.html
   ```
   > ⚠️ `index.html` harus berada di dalam folder `templates/`, karena Flask meng-serve-nya lewat `render_template()`.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Jalankan server:
   ```bash
   python app.py
   ```

4. Buka browser ke `http://localhost:5000`

---

## 🚀 Usage

### GUI
Buka `http://localhost:5000`, isi form (Square Feet, Bedrooms, Bathrooms, Year Built, Neighborhood), lalu klik **Predict Price**. Hasil estimasi, range harga, dan disclaimer akan langsung muncul.

### API

**`GET /health`** — cek status API.

**`GET /features`** — daftar fitur dan aturan validasi.

**`POST /predict`** — prediksi harga rumah.

Request body:
```json
{
  "SquareFeet": 1500,
  "Bedrooms": 3,
  "Bathrooms": 2,
  "YearBuilt": 2005,
  "Neighborhood_Urban": 0,
  "Neighborhood_Suburb": 1
}
```

Response:
```json
{
  "predicted_price": 236181.31,
  "price_range": { "low": 186823.31, "high": 285539.31 },
  "currency": "USD",
  "model_used": "LinearRegression",
  "model_r2": 0.5756,
  "disclaimer": "Harga ini adalah estimasi awal — sekitar 57.6% dari variasi harga rumah bisa diprediksi dari data ini. Faktor lain seperti lokasi strategis, fasilitas sekitar, kondisi bangunan, dan tren pasar terkini juga turut menentukan harga jual sebenarnya. Rentang di atas dihitung dari ± RMSE model (≈ $49,358)."
}
```

Encoding Neighborhood:

| Tipe | Neighborhood_Urban | Neighborhood_Suburb |
|---|---|---|
| Urban | 1 | 0 |
| Suburb | 0 | 1 |
| Rural | 0 | 0 |

---

## 📊 Results

Tiga model dilatih dan dibandingkan:

| Model | R² | RMSE |
|---|---|---|
| **Linear Regression** | **0.5756** ✅ | **49,358** |
| Random Forest | 0.5184 | 52,574 |
| XGBoost | < LR | — |

**Linear Regression** dipilih sebagai model final karena performa terbaik di antara ketiganya.

> R² ~0.576 berarti hanya ~57.6% variasi harga rumah yang bisa dijelaskan oleh fitur yang tersedia (SquareFeet, Bedrooms, Bathrooms, YearBuilt, Neighborhood). Sisanya dipengaruhi faktor di luar data, seperti lokasi strategis, fasilitas sekitar, kondisi bangunan, dan tren pasar.
