# VoiceGuard_PraktikumAI_Kelompok5

Aplikasi Flask untuk mendeteksi keaslian suara rekaman asli manusia (real) atau hasil manipulasi buatan AI (deepfake).

## Model
Model `.keras` tidak masuk ke repositori karena ukuran file yang besar. Silakan unduh melalui tautan berikut:
* [Download model_ai_voice.keras](https://drive.google.com/file/d/1fJlomxGB22LFOZu66KEfSabyaFxpL071/view?usp=sharing)

## Cara Menjalankan Lokal
1. Clone repo ini ke komputer kamu
2. Unduh model dari link di atas, lalu masukkan ke dalam folder proyek: `models/model_ai_voice.keras`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
Konfigurasi FFmpeg (Wajib untuk fitur rekam suara):

Windows: Unduh FFmpeg, taruh di folder aman (misal D:\ffmpeg), lalu daftarkan folder D:\ffmpeg\bin ke Environment Variables (Path) Windows. Restart VS Code setelah mendaftarkannya.

Mac/Linux: Jalankan brew install ffmpeg atau sudo apt install ffmpeg di terminal.

Jalankan aplikasi:

Bash
python app.py
Buka browser dan akses http://localhost:5000
