from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from pydub import AudioSegment
import tempfile, os, io, logging
from config import SR, TARGET_DURATION, IMG_SIZE, N_MELS, MODEL_PATH

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load model saat server start
try:
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    logging.info("Model berhasil dimuat!")
except Exception as e:
    logging.error(f"Gagal load model: {e}")
    MODEL = None

ALLOWED_EXTENSIONS = {".wav", ".ogg", ".aac", ".m4a", ".mp3", ".webm"}


def convert_to_wav(input_path, output_path):
    """Konversi berbagai format audio ke WAV menggunakan pydub"""
    ext = os.path.splitext(input_path)[1].lower()

    format_map = {
        ".ogg":  "ogg",
        ".aac":  "aac",
        ".m4a":  "mp4",
        ".mp3":  "mp3",
        ".webm": "webm",   # format dari MediaRecorder browser
        ".wav":  "wav",
    }

    fmt = format_map.get(ext, "wav")
    audio = AudioSegment.from_file(input_path, format=fmt)
    audio = audio.set_channels(1).set_frame_rate(SR)  # mono, 22050 Hz
    audio.export(output_path, format="wav")


def wav_to_melspec_array(file_path):
    """WAV → Mel-Spectrogram → array siap prediksi model"""
    audio, _ = librosa.load(file_path, sr=SR, duration=TARGET_DURATION)
    target_len = int(TARGET_DURATION * SR)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    mel    = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    ax.axis("off")
    librosa.display.specshow(mel_db, sr=SR, ax=ax)
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, 224, 224, 3)


def run_prediction(file_path):
    """Jalankan prediksi, return dict hasil"""
    features   = wav_to_melspec_array(file_path)
    prediction = MODEL.predict(features, verbose=0)

    prob_real = float(prediction[0][0])
    prob_fake = 1.0 - prob_real
    label     = "REAL" if prob_real >= 0.5 else "DEEPFAKE"

    return {
        "label":      label,
        "confidence": round(max(prob_real, prob_fake) * 100, 1),
        "prob_real":  round(prob_real * 100, 1),
        "prob_fake":  round(prob_fake * 100, 1),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model belum tersedia"}), 503

    if "audio" not in request.files:
        return jsonify({"error": "Tidak ada file yang dikirim"}), 400

    file = request.files["audio"]
    ext  = os.path.splitext(file.filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Format {ext} tidak didukung. Gunakan WAV, OGG, AAC, atau rekam langsung."}), 400

    # Cek ukuran (max 20MB)
    file.seek(0, 2)
    if file.tell() > 20 * 1024 * 1024:
        return jsonify({"error": "File terlalu besar, maksimal 20MB"}), 400
    file.seek(0)

    tmp_input  = None
    tmp_wav    = None

    try:
        # Simpan file asli ke temp
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            file.save(tmp.name)
            tmp_input = tmp.name

        # Kalau bukan WAV, konversi dulu
        if ext != ".wav":
            tmp_wav = tmp_input.replace(ext, "_converted.wav")
            convert_to_wav(tmp_input, tmp_wav)
            process_path = tmp_wav
        else:
            process_path = tmp_input

        result = run_prediction(process_path)
        return jsonify(result)

    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"error": "Gagal memproses audio, coba lagi"}), 500

    finally:
        for path in [tmp_input, tmp_wav]:
            if path and os.path.exists(path):
                os.unlink(path)


if __name__ == "__main__":
    app.run(debug=True)