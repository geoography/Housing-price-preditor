# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify, render_template
import joblib
import json
import numpy as np
import os

app = Flask(__name__)

# ── Load model & feature names saat server start ─────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "lr_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "feature_names.json")

# Metrik evaluasi model (dari hasil training di Colab)
MODEL_R2   = 0.5756
MODEL_RMSE = 49358

try:
    model         = joblib.load(MODEL_PATH)
    feature_names = json.load(open(FEATURE_PATH))
    print(f"✅  Model loaded — {len(feature_names)} fitur")
except FileNotFoundError as e:
    raise RuntimeError(
        f"File model tidak ditemukan: {e}\n"
        "Jalankan train_and_save.py di Colab dulu, lalu taruh "
        "lr_model.pkl dan feature_names.json di folder yang sama dengan app.py"
    )


# ── Aturan validasi per fitur ─────────────────────────────────────────
FIELD_RULES = {
    "SquareFeet": {
        "type": float, "min": 50,    "max": 50000,
        "label": "Luas rumah (SquareFeet)"
    },
    "Bedrooms": {
        "type": int,   "min": 1,     "max": 20,
        "label": "Jumlah kamar tidur (Bedrooms)"
    },
    "Bathrooms": {
        "type": int,   "min": 1,     "max": 20,
        "label": "Jumlah kamar mandi (Bathrooms)"
    },
    "YearBuilt": {
        "type": int,   "min": 1800,  "max": 2025,
        "label": "Tahun dibangun (YearBuilt)"
    },
    "Neighborhood_Urban": {
        "type": int,   "min": 0,     "max": 1,
        "label": "Neighborhood_Urban (0 atau 1)"
    },
    "Neighborhood_Suburb": {
        "type": int,   "min": 0,     "max": 1,
        "label": "Neighborhood_Suburb (0 atau 1)"
    },
}

def validate_input(data):
    """
    Validasi input request secara menyeluruh.
    Return (errors: list, cleaned: dict)
    - errors  : list string pesan error, kosong jika valid
    - cleaned : dict nilai yang sudah dikonversi ke tipe yang benar
    """
    errors  = []
    cleaned = {}

    for field, rules in FIELD_RULES.items():
        # ── 1. Field tidak ada ───────────────────────────────────────
        if field not in data:
            errors.append(f"'{field}' wajib diisi.")
            continue

        val = data[field]

        # ── 2. Nilai None / null ─────────────────────────────────────
        if val is None:
            errors.append(f"'{field}' tidak boleh null.")
            continue

        # ── 3. Tipe data salah ───────────────────────────────────────
        try:
            val = rules["type"](val)
        except (ValueError, TypeError):
            errors.append(
                f"'{field}' harus bertipe {rules['type'].__name__}, "
                f"tapi menerima: {repr(data[field])}"
            )
            continue

        # ── 4. Range nilai ───────────────────────────────────────────
        if not (rules["min"] <= val <= rules["max"]):
            errors.append(
                f"'{field}' harus antara {rules['min']} dan {rules['max']}, "
                f"tapi menerima: {val}"
            )
            continue

        cleaned[field] = val

    # ── 5. Validasi one-hot neighborhood (tidak boleh keduanya = 1) ──
    urban  = cleaned.get("Neighborhood_Urban",  0)
    suburb = cleaned.get("Neighborhood_Suburb", 0)
    if urban == 1 and suburb == 1:
        errors.append(
            "Neighborhood_Urban dan Neighborhood_Suburb tidak boleh keduanya bernilai 1. "
            "Pilih salah satu: Urban=(1,0), Suburb=(0,1), Rural=(0,0)."
        )

    return errors, cleaned


# ────────────────────────────────────────────────────────────────────
# GET / — serve HTML GUI
# ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# ────────────────────────────────────────────────────────────────────
# GET /health
# ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Cek apakah API berjalan normal."""
    return jsonify({
        "status":     "ok",
        "model":      type(model).__name__,
        "n_features": len(feature_names)
    }), 200


# ────────────────────────────────────────────────────────────────────
# GET /features
# ────────────────────────────────────────────────────────────────────
@app.route("/features", methods=["GET"])
def features():
    """Menampilkan daftar fitur dan aturan validasinya."""
    base_features  = ["SquareFeet", "Bedrooms", "Bathrooms", "YearBuilt"]
    neigh_features = [f for f in feature_names if f.startswith("Neighborhood")]

    return jsonify({
        "total_features":        len(feature_names),
        "base_features":         base_features,
        "neighborhood_features": neigh_features,
        "all_features":          feature_names,
        "validation_rules": {
            field: {"type": rules["type"].__name__, "min": rules["min"], "max": rules["max"]}
            for field, rules in FIELD_RULES.items()
        },
        "neighborhood_encoding": {
            "Urban":  {"Neighborhood_Urban": 1, "Neighborhood_Suburb": 0},
            "Suburb": {"Neighborhood_Urban": 0, "Neighborhood_Suburb": 1},
            "Rural":  {"Neighborhood_Urban": 0, "Neighborhood_Suburb": 0},
        },
        "example_request": {
            "SquareFeet": 1500, "Bedrooms": 3, "Bathrooms": 2,
            "YearBuilt": 2005,
            "Neighborhood_Urban": 0, "Neighborhood_Suburb": 1
        }
    }), 200


# ────────────────────────────────────────────────────────────────────
# POST /predict
# ────────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediksi harga rumah.

    Request body (JSON):
    {
        "SquareFeet": 1500,
        "Bedrooms": 3,
        "Bathrooms": 2,
        "YearBuilt": 2005,
        "Neighborhood_Urban": 0,
        "Neighborhood_Suburb": 1
    }

    Response sukses:
    {
        "predicted_price": 312500.75,
        "currency": "USD",
        "model_used": "LinearRegression"
    }

    Response error:
    {
        "error": "Deskripsi error",
        "details": ["list pesan validasi jika ada"]
    }
    """
    # ── 1. Validasi Content-Type ─────────────────────────────────────
    if not request.is_json:
        return jsonify({
            "error":  "Content-Type harus application/json",
            "detail": f"Menerima: {request.content_type}"
        }), 415

    # ── 2. Validasi JSON bisa di-parse ───────────────────────────────
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({
            "error":  "Request body bukan JSON yang valid",
            "detail": "Pastikan body berupa JSON object yang well-formed"
        }), 400

    # ── 3. Validasi JSON bukan list/string ───────────────────────────
    if not isinstance(data, dict):
        return jsonify({
            "error":  "Request body harus berupa JSON object, bukan array/string",
            "detail": f"Menerima tipe: {type(data).__name__}"
        }), 400

    # ── 4. Validasi isi field (tipe, range, kelengkapan) ─────────────
    errors, cleaned = validate_input(data)
    if errors:
        return jsonify({
            "error":   f"Terdapat {len(errors)} kesalahan validasi",
            "details": errors,
            "tip":     "Panggil GET /features untuk melihat aturan validasi lengkap"
        }), 400

    # ── 5. Susun input sesuai urutan fitur training ──────────────────
    try:
        input_values = [float(cleaned[f]) for f in feature_names]
        input_array  = np.array(input_values).reshape(1, -1)
    except Exception as e:
        return jsonify({"error": f"Gagal menyusun input: {str(e)}"}), 500

    # ── 6. Prediksi ──────────────────────────────────────────────────
    try:
        prediction = model.predict(input_array)[0]
    except Exception as e:
        return jsonify({"error": f"Prediksi gagal: {str(e)}"}), 500

    # ── 7. Validasi hasil prediksi masuk akal ────────────────────────
    if prediction < 0:
        return jsonify({
            "error":  "Hasil prediksi tidak valid (harga negatif)",
            "detail": "Coba periksa kembali nilai input yang diberikan"
        }), 500

    return jsonify({
        "predicted_price": round(float(prediction), 2),
        "price_range": {
            "low":  round(max(0.0, float(prediction) - MODEL_RMSE), 2),
            "high": round(float(prediction) + MODEL_RMSE, 2),
        },
        "currency":        "USD",
        "model_used":      type(model).__name__,
        "model_r2":        MODEL_R2,
        "disclaimer": (
            "Harga ini adalah estimasi awal yaitu sekitar "
            f"{MODEL_R2*100:.1f}% dari variasi harga rumah yang dapat diprediski. "
            "Faktor lain seperti lokasi strategis, fasilitas sekitar, kondisi bangunan, "
            "dan tren pasar terkini juga turut menentukan harga jual sebenarnya. "
        ),
    }), 200


# ────────────────────────────────────────────────────────────────────
# Handler error global
# ────────────────────────────────────────────────────────────────────
@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request", "detail": str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error":      "Endpoint tidak ditemukan",
        "available":  ["/", "/health", "/features", "/predict"]
    }), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        "error":  "HTTP method tidak diizinkan untuk endpoint ini",
        "detail": str(e)
    }), 405

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error", "detail": str(e)}), 500


# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
