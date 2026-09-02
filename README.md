# Mining Notísia Online no Deteksaun Topiku

Sistem Python untuk menambang berita dari **beberapa portal berita
Timor-Leste sekaligus** (Tatoli, Timor Post, Independente, dan sumber lain
yang bisa ditambahkan) dan mendeteksi topik yang sedang dibahas secara
otomatis. Pipeline: **RSS/Web (multi-sumber) → Collection → Preprocessing →
Topic Detection → Dashboard**, dibangun dengan `feedparser`, `scikit-learn`
(TF-IDF + K-Means), dan **Streamlit** untuk visualisasi interaktif.

## ✨ Fitur Utama

1. **Multi-Sumber Berita** — koleksi dari beberapa portal sekaligus
   (dikonfigurasi di `config/settings.yaml` bagian `sources`, tinggal
   tambah entri baru untuk sumber lain — tidak perlu ubah kode). Semua
   sumber digabung ke satu dataset terpadu dengan kolom `source` menandai
   asalnya, dan bisa difilter per sumber di dashboard.
2. **RSS Collection + Web Scraping Penuh** — untuk tiap sumber, ambil
   daftar artikel terbaru (judul, link, tanggal, ringkasan) via `feedparser`,
   LALU kunjungi setiap link artikelnya untuk menarik **isi lengkap,
   kategori, dan gambar thumbnail** lewat `requests` + `BeautifulSoup`
   (bisa dimatikan dan pakai ringkasan RSS saja lewat config). Hasil disimpan
   ke `data/raw/` (snapshot per-sumber + master file kumulatif gabungan ter-dedup).
3. **Preprocessing Multi-Bahasa** — cleaning teks (lowercase, hapus tanda
   baca/HTML), stopword removal, dan **stemming/lemmatization** untuk
   **4 bahasa**: Tetun, Indonesia, **Portugis**, dan English — sesuai
   kebutuhan media Timor-Leste yang memakai kombinasi bahasa resmi
   (Tetun & Portugis) plus Indonesia/Inggris. Simpan ke `data/cleaned/`.
4. **Feature Extraction (TF-IDF)** — representasi vektor numerik via
   `TfidfVectorizer`, model disimpan ke `data/models/`.
5. **Topic Detection (K-Means)** — clustering berita per topik LINTAS SUMBER
   (topik yang sama dari media berbeda akan mengelompok bersama), jumlah
   cluster optimal dicari otomatis lewat **silhouette score**, model
   disimpan ke `data/models/`.
6. **Keyword Extraction** — top-10 term per cluster + label topik, disimpan
   ke `data/keywords/`.
7. **Trend Dashboard (Streamlit)** — topik utama, distribusi berita per
   cluster, **tren topik dari waktu ke waktu**, **topik trending**,
   **wordcloud per cluster**, dan filter **bahasa, kategori, sumber
   berita, dan rentang waktu (termasuk input jumlah hari sendiri 1–30)**.

## 📰 Sumber Berita yang Didukung

| Sumber | Status | Catatan |
|---|---|---|
| **Tatoli** (`tatoli.tl`) | ✅ Aktif & teruji | Portal berita nasional resmi |
| **Timor Post** (`timorpost.com`) | ✅ Aktif | Ada juga edisi Português (`pt.timorpost.com`) & Indonesia (`id.timorpost.com`) — bisa ditambah sebagai sumber terpisah |
| **Independente** (`independente.tl`) | ✅ Aktif | Kemungkinan berbasis Joomla — cek path RSS manual kalau gagal |
| **TimorNews** (`timornews.tl`) | ⚠️ Nonaktif (perlu verifikasi) | Domain persis ini tidak ditemukan saat konfigurasi dibuat — cek nama domain yang benar dulu |

Tambah/kurangi sumber di `config/settings.yaml` bagian `sources` — lihat
komentar di file itu untuk detail tiap sumber.

## 📁 Struktur Folder

```
news_topic_mining/
│
├── data/
│   ├── raw/                # Hasil RSS feed mentah dari Tatoli
│   ├── cleaned/             # Data setelah preprocessing (stopwords, stemming, dll.)
│   ├── models/               # TF-IDF vectorizer & model K-Means
│   └── keywords/               # Hasil ekstraksi kata kunci/topik
│
├── src/
│   ├── scraping/
│   │   ├── rss_scraper.py        # Parsing RSS dengan feedparser (discovery + fallback ringkasan)
│   │   └── web_scraper.py         # Scraping isi artikel lengkap+kategori+gambar (requests+BeautifulSoup)
│   ├── preprocessing/
│   │   ├── text_cleaner.py       # Cleaning, deteksi bahasa, stemming
│   │   ├── stopwords_data.py      # Stopwords Tetun/Indonesia/English
│   │   └── pipeline.py             # Orkestrasi preprocessing
│   ├── feature_extraction/
│   │   └── tfidf_vectorizer.py      # Ekstraksi fitur TF-IDF
│   ├── clustering/
│   │   └── kmeans_model.py           # Clustering K-Means + pencarian k optimal
│   ├── keywords/
│   │   └── keyword_extractor.py       # Ekstraksi top-N term per cluster
│   └── dashboard/
│       └── trend_dashboard.py          # Dashboard Streamlit
│
├── notebooks/
│   └── rss_demo.ipynb        # Demo eksplorasi feedparser & preprocessing
│
├── config/
│   └── settings.yaml         # URL feed, path, parameter clustering
│
├── logs/                     # Log koleksi & training
├── tests/                    # Unit test
├── requirements.txt
├── README.md
└── main.py                   # Pipeline end-to-end
```

## ⚙️ Instalasi

```bash
cd news_topic_mining
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Semua stemmer bekerja **offline**: Sastrawi (Indonesia) memakai kamus
> bawaan, Porter Stemmer (English) sepenuhnya algoritmik, dan Tetun memakai
> heuristik suffix-stripping manual — tidak perlu mengunduh korpus tambahan.

## 🕸️ Mode Koleksi: Scraping Penuh vs RSS Ringkasan

Diatur di `config/settings.yaml` bagian `scraping.full_content`:

| Mode | `full_content` | Yang ditarik | Kecepatan |
|---|---|---|---|
| **Full scraping** (default) | `true` | Isi artikel lengkap + kategori + gambar thumbnail | Lebih lambat (1 request/artikel) |
| **RSS saja** | `false` | Judul + ringkasan singkat saja | Lebih cepat |

Cara kerja mode **full scraping**:
1. RSS dipakai untuk *discovery* — menemukan daftar link artikel terbaru.
2. Untuk artikel yang **belum pernah** punya isi lengkap tersimpan, sistem
   mengunjungi halamannya satu per satu (dengan jeda `scraping.delay_seconds`
   antar request — etika scraping yang sopan) dan menarik isi lengkap,
   kategori, dan gambar via `BeautifulSoup`.
3. Artikel yang sudah pernah di-scrape **tidak di-scrape ulang** pada run
   berikutnya, jadi tetap ringan meski dijalankan berkali-kali.
4. Jumlah artikel baru yang di-scrape per run dibatasi oleh
   `scraping.max_articles_per_run` supaya tidak membebani server Tatoli.

> ⚠️ Struktur HTML situs bisa berubah sewaktu-waktu (ganti tema, dll). Kalau
> hasil `content`/`category`/`image_url` tiba-tiba kosong, cek HTML halaman
> artikel Tatoli (klik kanan → Inspect di browser) dan sesuaikan daftar CSS
> selector di `config/settings.yaml` bagian `scraping.selectors` — **tidak
> perlu mengubah kode Python**.

## 🚀 Menjalankan Pipeline (CLI)

```bash
python main.py
# paksa mode tertentu tanpa ubah config:
python main.py --mode full   # scraping isi lengkap+kategori+gambar
python main.py --mode rss    # ringkasan RSS saja (lebih cepat)
# atau override feed & rentang cluster:
python main.py --feed-url "https://tatoli.tl/feed/" --min-k 2 --max-k 8
```

Pipeline akan:
1. Menarik daftar artikel terbaru dari RSS Tatoli (dan, jika mode full,
   men-scrape isi lengkap tiap artikel barunya), menggabungkan dengan
   riwayat yang sudah terkumpul (`data/raw/tatoli_master.csv`, dedup via `news_id`).
2. Membersihkan teks (memakai isi lengkap jika tersedia, fallback ke
   ringkasan RSS), mendeteksi bahasa, dan melakukan stemming →
   `data/cleaned/tatoli_cleaned_master.csv`.
3. Membangun representasi TF-IDF → `data/models/tatoli_tfidf_vectorizer.joblib`.
4. Melatih K-Means dengan k optimal (silhouette score) →
   `data/models/tatoli_kmeans.joblib`.
5. Mengekstrak top-10 keyword tiap topik → `data/keywords/tatoli_keywords_*.json`.
6. Menampilkan ringkasan hasil di terminal.

Karena feed RSS berubah dari waktu ke waktu, **jalankan pipeline ini secara
berkala** (mis. via cron job harian) agar `data/raw` dan `data/cleaned`
terus terakumulasi — ini yang membuat grafik tren pada dashboard bermakna.

## 📊 Menjalankan Dashboard

```bash
streamlit run src/dashboard/trend_dashboard.py
```

Dashboard versi terbaru punya tampilan lebih interaktif:
- **Header ringkasan** dengan status "data terakhir diperbarui X menit lalu".
- **Kartu statistik** (total berita, jumlah topik, kategori, bahasa).
- Tab **🏠 Beranda** — sorotan 3 topik teratas & 5 berita terbaru.
- Tab **🔥 Trending** — topik yang sedang naik daun (berdasarkan lonjakan
  jumlah berita dalam 3/7/14 hari terakhir dibanding riwayatnya), lengkap
  dengan daftar berita terbaru dari topik-topik tersebut.
- Tab **📊 Distribusi Topik**, **📈 Tren Topik** (line chart waktu ke waktu),
  **☁️ Wordcloud** per cluster, **🔑 Kata Kunci**, **📰 Berita** (dengan kolom
  pencarian & sortir), dan **🖼️ Galeri** thumbnail berita terbaru.
- Filter **bahasa** (Tetun/Indonesia/Portugis/English), **kategori**,
  **sumber berita**, dan **rentang waktu** (7 hari terakhir / 30 hari
  terakhir / masukkan jumlah hari sendiri 1–30 / kustom tanggal) — berlaku
  ke semua tab sekaligus.
- Kotak pencarian bebas di sidebar untuk mencari kata kunci pada judul/isi berita.
- Tombol **🔄 Tarik Data Sekarang** untuk update manual kapan saja.

### 🔁 Auto-refresh (data otomatis sinkron dengan web Tatoli)

Di sidebar, aktifkan toggle **Auto-refresh** dan atur intervalnya (5–60 menit).
Dashboard akan otomatis menarik data terbaru dari Tatoli setiap interval
tersebut **tanpa perlu klik manual** — cocok dibiarkan terbuka di satu tab
browser sebagai layar pemantauan berita berjalan (live monitoring).

Fitur ini butuh paket tambahan `streamlit-autorefresh` (sudah ada di
`requirements.txt`). Kalau paket ini belum terpasang, dashboard tetap
berjalan normal, hanya saja toggle auto-refresh akan nonaktif otomatis dan
kamu perlu klik **Tarik Data Sekarang** secara manual.

> ⚠️ Auto-refresh berjalan selama TAB BROWSER dashboard terbuka & tidak
> di-sleep oleh sistem operasi/browser (mis. laptop masuk mode tidur).
> Untuk pembaruan data terjadwal yang benar-benar berjalan 24/7 di background
> tanpa tergantung browser terbuka, gunakan cron job yang menjalankan
> `python main.py` secara berkala (lihat bagian Deploy Online di bawah).

## 🌐 Deploy Online (Akses dari Mana Saja)

Ada 3 opsi, dari yang paling mudah ke yang paling fleksibel:

### Opsi 1 — Streamlit Community Cloud (GRATIS, paling mudah, direkomendasikan)

Cocok untuk pemakaian pribadi/demo. **Catatan penting**: penyimpanan di
Community Cloud bersifat *ephemeral* — kalau aplikasi di-restart (tidur
karena tidak dipakai lama, atau redeploy), isi folder `data/` yang dibuat
saat runtime (hasil scraping) akan **ter-reset**. Ini bukan masalah besar
karena aplikasi akan otomatis menarik ulang data dari Tatoli saat dibuka
lagi — hanya saja riwayat tren jangka panjang tidak akan tersimpan permanen
di tier gratis ini.

1. Buat akun GitHub gratis di https://github.com/signup (kalau belum punya).
2. Buat repository baru (mis. `tatoli-topic-analyzer`), upload seluruh isi
   folder `news_topic_mining/` ke repo tersebut (lewat GitHub Desktop, web
   upload, atau `git push` biasa).
3. Buka https://share.streamlit.io, login pakai akun GitHub.
4. Klik **New app** → pilih repo yang baru dibuat.
5. Isi **Main file path** dengan: `src/dashboard/trend_dashboard.py`
6. Klik **Deploy**. Tunggu beberapa menit — Streamlit Cloud otomatis
   membaca `requirements.txt` dan meng-install semua dependency.
7. Selesai — kamu dapat URL publik seperti
   `https://tatoli-topic-analyzer.streamlit.app` yang bisa diakses dari
   perangkat mana pun tanpa instalasi apa pun di sisi pengunjung.

### Opsi 2 — VPS/Server Sendiri (persisten, kontrol penuh)

Cocok kalau butuh data tersimpan permanen & auto-update terjadwal 24/7.

```bash
# di VPS (Ubuntu/Debian)
git clone <repo-kamu>.git && cd news_topic_mining
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# jalankan dashboard permanen di background
nohup streamlit run src/dashboard/trend_dashboard.py \
    --server.port=8501 --server.address=0.0.0.0 > logs/streamlit.log 2>&1 &

# jadwalkan pembaruan data otomatis tiap 30 menit (independen dari browser)
crontab -e
# tambahkan baris:
*/30 * * * * cd /path/ke/news_topic_mining && venv/bin/python main.py >> logs/cron.log 2>&1
```

Supaya bisa diakses lewat domain/HTTPS, pasang **Nginx** sebagai reverse
proxy ke port 8501, lalu aktifkan sertifikat gratis dengan **Certbot**
(Let's Encrypt). Ini di luar cakupan sistem ini, tapi merupakan setup
standar untuk aplikasi Streamlit — banyak tutorial "deploy streamlit nginx
certbot" yang bisa diikuti.

### Opsi 3 — Docker (portabel, mudah dipindah ke provider mana pun)

```bash
docker build -t tatoli-topic-analyzer .
docker run -p 8501:8501 -v $(pwd)/data:/app/data tatoli-topic-analyzer
```

Flag `-v $(pwd)/data:/app/data` **wajib** dipakai supaya data tetap ada
setelah container di-restart. Image Docker ini juga bisa langsung dipakai
di provider seperti Railway, Render, Fly.io, atau Google Cloud Run — semua
mendukung deploy langsung dari Dockerfile.

## 🧪 Menjalankan Unit Test

```bash
python -m unittest discover -s tests -v
```

## 🧩 Modul Penting

| Modul | Fungsi |
|---|---|
| `src/scraping/rss_scraper.py` | Fetch & parsing RSS dengan `feedparser`, dedup & simpan ke `data/raw`. |
| `src/scraping/web_scraper.py` | Kunjungi tiap link artikel, tarik isi lengkap+kategori+gambar via `BeautifulSoup`, hanya scrape artikel yang belum pernah diambil. |
| `src/preprocessing/stopwords_data.py` | Stopwords Tetun/Indonesia/English (kurasi manual, offline). |
| `src/preprocessing/text_cleaner.py` | Cleaning teks, deteksi bahasa heuristik, stemming per bahasa. |
| `src/preprocessing/pipeline.py` | Orkestrasi preprocessing dari `data/raw` → `data/cleaned`. |
| `src/feature_extraction/tfidf_vectorizer.py` | `TfidfFeatureExtractor`: fit/transform/save/load TF-IDF. |
| `src/clustering/kmeans_model.py` | `TopicKMeans`: pencarian k optimal (silhouette + inertia), fit/save/load. |
| `src/keywords/keyword_extractor.py` | Ekstraksi top-N term per cluster + label topik, simpan JSON. |
| `src/dashboard/trend_dashboard.py` | Dashboard Streamlit end-to-end (distribusi, tren, wordcloud). |
| `main.py` | Pipeline CLI: RSS → preprocessing → TF-IDF → K-Means → keywords. |

## 📝 Catatan Bahasa & Stemming

- **Deteksi bahasa** memakai heuristik overlap-stopwords — cukup akurat
  untuk judul/ringkasan berita pendek, tapi bisa keliru pada teks yang
  sangat pendek atau campur-bahasa (code-mixing).
- **Stemming Tetun** memakai heuristik suffix-stripping manual (bukan
  linguistik penuh), karena belum ada library stemmer Tetun yang mapan.
  Bisa disempurnakan lebih lanjut jika tersedia kamus/aturan morfologi
  Tetun yang lebih lengkap.
- **Stemming Indonesia** memakai Sastrawi (kamus bawaan, akurat untuk
  bahasa berita formal).
- **Stemming Portugis** memakai Snowball Stemmer dari NLTK (algoritmik,
  tidak butuh download corpus) — cocok untuk berita berbahasa Portugis
  dari media seperti Timor Post & Independente.
- **Stemming English** memakai Porter Stemmer dari NLTK (algoritmik, tidak
  butuh download corpus).

## 🗺️ Roadmap Lanjutan (opsional)

- Tambahkan sumber RSS lain (portal berita Timor-Leste lainnya) untuk
  perbandingan cakupan topik antar media.
- Sentiment analysis per topik/berita.
- Penjadwalan otomatis (cron/Airflow) untuk koleksi RSS berkala.
- Model deteksi bahasa berbasis ML sebagai pengganti heuristik stopwords.
