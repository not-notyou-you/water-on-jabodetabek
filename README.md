# Water on Jabodetabek (TOPIK 1)

Aplikasi Streamlit untuk memprediksi peta genangan Jabodetabek pada observasi Sentinel-1 berikutnya, memakai model rekomendasi notebook `modeling_v9.ipynb` (ContextCNN, TorchScript). Ada dua mode:

- **Simulasi skenario**: atur curah hujan dan jarak hari lewat slider atau preset Kemarau, Normal, dan Ekstrem.
- **Cek tanggal**: pilih tanggal (atau tekan Cek hari ini). Aplikasi mengunduh curah hujan GPM IMERG aktual dari NASA (wajib kredensial Earthdata) dan, bila diaktifkan, komposit MODIS terbaru dari layanan subset ORNL (tanpa login).

## Struktur folder

```
repo/
├── requirements.txt
├── README_topik1.md
├── .streamlit/config.toml
└── topik1/
    ├── app.py
    ├── inti.py
    ├── tampilan.py
    ├── data_nasa.py
    ├── .streamlit/config.toml
    └── artefak/
        ├── model.ts
        ├── meta.json
        ├── kondisi_awal.npz
        ├── frame_0_a.npz
        ├── frame_1_a.npz
        ├── frame_2_a.npz
        └── verify.json
```

Salin semua isi `deploy/topik1/` dari notebook ke `topik1/artefak/`, termasuk semua `frame_*.npz`.

## Menjalankan di komputer sendiri

```
pip install -r requirements.txt
streamlit run topik1/app.py
```

Kredensial NASA bisa diketik langsung di aplikasi, atau disimpan di `.streamlit/secrets.toml` (jangan diunggah ke GitHub):

```
NASA_EARTHDATA_TOKEN = "..."
NASA_EARTHDATA_USER = "..."
NASA_EARTHDATA_PASSWORD = "..."
```

## Deploy ke Streamlit Cloud

1. Di github.com, klik **New repository**, beri nama (misalnya `skripsi-banjir`), pilih **Public**, lalu **Create repository**.
2. Klik **Add file → Upload files**, seret `requirements.txt`, `README_topik1.md`, dan folder `topik1` (Chrome atau Edge mempertahankan subfolder). Pastikan semua file di `topik1/artefak/` ikut. Klik **Commit changes**.
3. Folder berawalan titik (`.streamlit`) sering terlewat. Bila belum ada, buat lewat **Add file → Create new file** dengan nama `.streamlit/config.toml`, lalu tempel isinya.
4. Buka share.streamlit.io, masuk dengan GitHub, klik **Create app → Deploy a public app from GitHub**.
5. Pilih repo, branch `main`, **Main file path** `topik1/app.py`, buka **Advanced settings**, pilih **Python 3.11**.
6. Opsional: di **Advanced settings → Secrets**, tempel tiga baris kredensial NASA di atas agar tidak perlu diketik ulang. Secrets tidak terlihat oleh pengunjung, tetapi semua pengunjung app bisa memakai fitur Cek tanggal dengan akunmu.
7. Klik **Deploy**. Pemasangan pertama butuh beberapa menit karena mengunduh torch.

## Menyiapkan akun NASA Earthdata

1. Daftar di urs.earthdata.nasa.gov.
2. Di profil, buka **Applications → Authorized Apps → Approve More Applications**, setujui **NASA GESDISC DATA ARCHIVE**. Tanpa ini, unduhan GPM ditolak (401).
3. Buka **Generate Token**, salin token. Token lebih aman dari password dan berlaku sekitar 60 hari.

## Catatan pemakaian

- Saat pertama dibuka, aplikasi menjalankan satu prediksi verifikasi terhadap `verify.json`. Setiap prediksi baru butuh sekitar `inferensi_cpu_detik` detik. Cek tanggal butuh tambahan waktu untuk mengunduh 7 hari data GPM dan 4 band MODIS.
- Notifikasi muncul di pojok kanan atas saat prediksi dimulai, berhasil, atau gagal, dan status proses tampil di bawah panel input. Jangan mengubah input saat proses berjalan karena Streamlit akan membatalkannya.
- GPM IMERG Final terbit sekitar 3,5 bulan setelah tanggal pengamatan. Untuk tanggal baru, aplikasi otomatis memakai versi Late, lalu Early. Untuk hari ini, data harian sering belum lengkap sehingga jendela hujan diakhiri 1 sampai 2 hari sebelumnya.
- Riwayat Sentinel-1 tetap tiga observasi terakhir pada artefak. Hasil Cek tanggal untuk tanggal jauh setelahnya adalah estimasi kasar.
- Di free tier, aplikasi tidur bila lama tidak dibuka. Buka beberapa menit sebelum sidang dan jalankan satu prediksi lebih dulu.