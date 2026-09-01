"""
Streamlit Dashboard — Mining Notísia Online no Deteksaun Topiku.

Jalankan dengan:
    streamlit run src/dashboard/trend_dashboard.py

Fitur:
- Desain interaktif: header ringkasan, kartu statistik, tab navigasi, pencarian.
- Auto-refresh: dashboard bisa menarik RSS Tatoli & memperbarui data secara
  berkala tanpa perlu klik manual (interval bisa diatur di sidebar).
- Topik utama, distribusi berita per cluster, tren waktu, wordcloud, galeri.
- Filter bahasa & kategori, pencarian kata kunci pada judul/isi berita.
"""
import os
import sys
import json
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import load_config, setup_logging  # noqa: E402
from main import run_pipeline  # noqa: E402

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

LANG_LABELS = {"id": "Indonesia", "en": "English", "tet": "Tetun", "unknown": "Lainnya"}
LAST_FETCH_FILE = os.path.join(PROJECT_ROOT, "logs", "last_fetch.txt")

st.set_page_config(
    page_title="Mining Notísia Online no Deteksaun Topiku",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1300px; }

    .tatoli-header {
        background: linear-gradient(135deg, #b3151a 0%, #7a0f13 100%);
        padding: 1.6rem 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.2rem;
    }
    .tatoli-header h1 { margin: 0; font-size: 1.7rem; }
    .tatoli-header p { margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }

    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.8rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.18);
        font-size: 0.8rem;
        margin-top: 0.6rem;
    }

    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border-left: 4px solid #b3151a;
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }

    .news-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.7rem;
    }
    .news-card h4 { margin: 0 0 0.3rem 0; font-size: 1.0rem; }
    .news-card .meta { font-size: 0.8rem; opacity: 0.7; margin-bottom: 0.3rem; }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
    }
</style>
"""


# ----------------------------------------------------------------------
# Helpers: persisted "last fetch" timestamp (dipakai untuk auto-refresh)
# ----------------------------------------------------------------------
def _read_last_fetch_time():
    if os.path.exists(LAST_FETCH_FILE):
        try:
            with open(LAST_FETCH_FILE, "r") as f:
                return datetime.fromisoformat(f.read().strip())
        except Exception:
            return None
    return None


def _write_last_fetch_time(ts: datetime):
    os.makedirs(os.path.dirname(LAST_FETCH_FILE), exist_ok=True)
    with open(LAST_FETCH_FILE, "w") as f:
        f.write(ts.isoformat())


def _format_elapsed(ts: datetime) -> str:
    if ts is None:
        return "belum pernah"
    delta = datetime.now() - ts
    minutes = int(delta.total_seconds() // 60)
    if minutes < 1:
        return "baru saja"
    if minutes < 60:
        return f"{minutes} menit lalu"
    hours = minutes // 60
    return f"{hours} jam lalu"


@st.cache_resource
def get_config():
    config = load_config()
    setup_logging(config["paths"]["logs"])
    return config


def load_latest_from_disk(config):
    cleaned_master = os.path.join(config["paths"]["cleaned"], "tatoli_cleaned_master.csv")
    keywords_latest = os.path.join(config["paths"]["keywords"], "tatoli_keywords_latest.json")

    if not os.path.exists(cleaned_master):
        return None

    df = pd.read_csv(cleaned_master)
    topics_payload = None
    if os.path.exists(keywords_latest):
        with open(keywords_latest, "r", encoding="utf-8") as f:
            topics_payload = json.load(f)

    return {"df": df, "topics_payload": topics_payload}


def _topics_to_payload(result):
    return {
        "source": "tatoli",
        "topics": [
            {
                "cluster_id": int(cid),
                "label": result["topic_labels"].get(cid, ""),
                "top_terms": words,
                "news_count": int((result["cleaned_df"]["cluster"] == cid).sum()),
            }
            for cid, words in result["topics"].items()
        ],
    }


def _execute_pipeline(config, feed_url, min_k, max_k, mode):
    result = run_pipeline(config, feed_url=feed_url, min_k=min_k, max_k=max_k, collection_mode=mode)
    st.session_state["news_result"] = {
        "df": result["cleaned_df"],
        "topics_payload": _topics_to_payload(result),
    }
    _write_last_fetch_time(datetime.now())
    return result


# ----------------------------------------------------------------------
# Render sections
# ----------------------------------------------------------------------
def render_header(last_fetch):
    st.markdown(
        f"""
        <div class="tatoli-header">
            <h1>📰 Mining Notísia Online no Deteksaun Topiku</h1>
            <p>Analisis topik berita otomatis dari RSS &amp; halaman artikel Tatoli
            (<a href="https://tatoli.tl" style="color:#ffe1b3;" target="_blank">tatoli.tl</a>)
            — koleksi, pembersihan teks, deteksi topik TF-IDF + K-Means, dan tren interaktif.</p>
            <span class="status-pill">🕓 Data terakhir diperbarui: {_format_elapsed(last_fetch)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stats(df: pd.DataFrame):
    total_news = len(df)
    total_topics = df["cluster"].nunique() if "cluster" in df.columns else 0
    langs = df["language"].value_counts().to_dict() if "language" in df.columns else {}
    n_categories = df["category"].nunique() if "category" in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Total Berita", f"{total_news:,}")
    c2.metric("🧩 Jumlah Topik", total_topics)
    c3.metric("🗂️ Kategori Terdeteksi", n_categories if n_categories else "-")
    c4.metric("🌐 Bahasa Terdeteksi", ", ".join(sorted(langs.keys())) if langs else "-")


def render_overview(df: pd.DataFrame, labels: dict):
    st.subheader("🔥 Sorotan Hari Ini")
    dist = df["cluster"].value_counts().reset_index()
    dist.columns = ["cluster", "jumlah"]
    dist["topik"] = dist["cluster"].map(lambda c: labels.get(c, labels.get(str(c), f"Cluster {c}")))
    top3 = dist.sort_values("jumlah", ascending=False).head(3)

    cols = st.columns(len(top3)) if len(top3) else [st]
    for col, (_, row) in zip(cols, top3.iterrows()):
        with col:
            st.markdown(
                f"""<div class="news-card">
                    <div class="meta">Topik trending</div>
                    <h4>🏷️ {row['topik']}</h4>
                    <div class="meta">{int(row['jumlah'])} berita</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.subheader("🆕 Berita Terbaru")
    recent = df.copy()
    if "published_at" in recent.columns:
        recent["published_at"] = pd.to_datetime(recent["published_at"], errors="coerce")
        recent = recent.sort_values("published_at", ascending=False)
    recent = recent.head(5)

    for _, row in recent.iterrows():
        topik = labels.get(row.get("cluster"), labels.get(str(row.get("cluster")), ""))
        pub = row.get("published_at")
        pub_str = pub.strftime("%d %b %Y, %H:%M") if pd.notna(pub) else "-"
        link = row.get("link", "")
        title = row.get("title", "(tanpa judul)")
        title_html = f'<a href="{link}" target="_blank" style="text-decoration:none;">{title}</a>' if link else title
        st.markdown(
            f"""<div class="news-card">
                <div class="meta">{pub_str} • {row.get('category', '') or 'Umum'} • {topik}</div>
                <h4>{title_html}</h4>
            </div>""",
            unsafe_allow_html=True,
        )


def render_topic_distribution(df: pd.DataFrame, labels: dict):
    st.subheader("📊 Distribusi Berita per Cluster")
    dist = df["cluster"].value_counts().reset_index()
    dist.columns = ["cluster", "jumlah_berita"]
    dist["topik"] = dist["cluster"].map(lambda c: labels.get(c, labels.get(str(c), f"Cluster {c}")))
    dist = dist.sort_values("jumlah_berita", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig_bar = px.bar(dist, x="topik", y="jumlah_berita", color="topik",
                          title="Jumlah Berita per Topik")
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    with col2:
        fig_pie = px.pie(dist, names="topik", values="jumlah_berita",
                          title="Proporsi Topik", hole=0.35)
        st.plotly_chart(fig_pie, use_container_width=True)


def render_topic_trend(df: pd.DataFrame, labels: dict):
    st.subheader("📈 Tren Topik dari Waktu ke Waktu")
    tdf = df.copy()
    tdf["published_at"] = pd.to_datetime(tdf["published_at"], errors="coerce")
    tdf = tdf.dropna(subset=["published_at"])
    if tdf.empty:
        st.info("Belum cukup data tanggal untuk menampilkan tren. Biarkan auto-refresh berjalan "
                 "beberapa kali (atau jalankan koleksi manual di hari berbeda) agar tren terbentuk.")
        return

    tdf["topik"] = tdf["cluster"].map(lambda c: labels.get(c, labels.get(str(c), f"Cluster {c}")))
    tdf["date"] = tdf["published_at"].dt.date

    trend = tdf.groupby(["date", "topik"]).size().reset_index(name="jumlah")
    fig = px.line(trend, x="date", y="jumlah", color="topik", markers=True,
                  title="Tren Jumlah Berita per Topik dari Waktu ke Waktu")
    st.plotly_chart(fig, use_container_width=True)


def render_wordclouds(df: pd.DataFrame, labels: dict):
    st.subheader("☁️ Wordcloud per Cluster")
    clusters = sorted(df["cluster"].dropna().unique())
    if not clusters:
        st.info("Belum ada cluster untuk ditampilkan.")
        return

    cols = st.columns(min(3, len(clusters)))
    for i, cid in enumerate(clusters):
        text_blob = " ".join(df[df["cluster"] == cid]["text_final"].fillna(""))
        if not text_blob.strip():
            continue
        wc = WordCloud(width=500, height=350, background_color="white",
                        colormap="Reds", collocations=False).generate(text_blob)

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        label = labels.get(cid, labels.get(str(cid), f"Cluster {cid}"))
        ax.set_title(f"Cluster {cid} — {label}", fontsize=10)

        with cols[i % len(cols)]:
            st.pyplot(fig)
            plt.close(fig)


def render_keywords_list(topics_payload):
    st.subheader("🔑 Kata Kunci Utama per Topik")
    if not topics_payload:
        st.info("Data kata kunci belum tersedia.")
        return
    for t in topics_payload.get("topics", []):
        with st.expander(f"Cluster {t['cluster_id']} — {t['label']} ({t.get('news_count', 0)} berita)"):
            st.write(", ".join(t["top_terms"]))


def render_news_table(df: pd.DataFrame, labels: dict, search_query: str):
    st.subheader("📰 Daftar Berita")
    show_df = df.copy()
    show_df["topik"] = show_df["cluster"].map(lambda c: labels.get(c, labels.get(str(c), f"Cluster {c}")))

    if search_query:
        q = search_query.lower()
        text_pool = (show_df.get("title", "").fillna("") + " " + show_df.get("text_clean", "").fillna(""))
        show_df = show_df[text_pool.str.lower().str.contains(q)]

    sort_option = st.radio("Urutkan:", ["Terbaru", "Paling Relevan (topik)"], horizontal=True)
    if sort_option == "Terbaru" and "published_at" in show_df.columns:
        show_df["published_at"] = pd.to_datetime(show_df["published_at"], errors="coerce")
        show_df = show_df.sort_values("published_at", ascending=False)
    else:
        show_df = show_df.sort_values("cluster")

    cols = ["title", "topik", "category", "language", "published_at", "link"]
    cols = [c for c in cols if c in show_df.columns]
    st.caption(f"Menampilkan {min(len(show_df), 300)} dari {len(show_df)} berita.")
    st.dataframe(show_df[cols].head(300), use_container_width=True, height=420)


def render_gallery(df: pd.DataFrame, labels: dict, n: int = 12):
    st.subheader("🖼️ Galeri Berita Terbaru")
    if "image_url" not in df.columns:
        st.info("Belum ada data gambar. Aktifkan mode scraping penuh (`scraping.full_content: true` "
                 "di config/settings.yaml) untuk menarik thumbnail artikel.")
        return

    gdf = df.copy()
    gdf["image_url"] = gdf["image_url"].fillna("")
    gdf = gdf[gdf["image_url"].str.strip() != ""]
    if "published_at" in gdf.columns:
        gdf["published_at"] = pd.to_datetime(gdf["published_at"], errors="coerce")
        gdf = gdf.sort_values("published_at", ascending=False)

    gdf = gdf.head(n)
    if gdf.empty:
        st.info("Belum ada berita dengan gambar thumbnail pada data saat ini.")
        return

    cols = st.columns(4)
    for i, (_, row) in enumerate(gdf.iterrows()):
        with cols[i % 4]:
            try:
                st.image(row["image_url"], use_container_width=True)
            except Exception:
                st.caption("(gambar tidak dapat dimuat)")
            topik = labels.get(row.get("cluster"), labels.get(str(row.get("cluster")), ""))
            st.markdown(f"**{row['title']}**")
            st.caption(f"{row.get('category', '')} • {topik}")
            if row.get("link"):
                st.markdown(f"[Baca selengkapnya]({row['link']})")


def render_language_filter(df: pd.DataFrame):
    langs_present = [l for l in df["language"].dropna().unique() if l in LANG_LABELS]
    options = ["Semua"] + [LANG_LABELS.get(l, l) for l in langs_present]
    choice = st.selectbox("🌐 Bahasa", options)
    if choice == "Semua":
        return df
    inv_map = {v: k for k, v in LANG_LABELS.items()}
    return df[df["language"] == inv_map.get(choice, choice)]


def render_category_filter(df: pd.DataFrame):
    if "category" not in df.columns:
        return df
    categories_present = sorted(
        c for c in df["category"].fillna("").unique() if isinstance(c, str) and c.strip()
    )
    if not categories_present:
        return df
    options = ["Semua"] + categories_present
    choice = st.selectbox("🗂️ Kategori", options)
    if choice == "Semua":
        return df
    return df[df["category"] == choice]


def render_date_filter(df: pd.DataFrame):
    """Filter waktu: pilihan cepat (7/30 hari terakhir) atau rentang tanggal kustom."""
    if "published_at" not in df.columns:
        return df

    dcol = pd.to_datetime(df["published_at"], errors="coerce")
    valid_dates = dcol.dropna()
    if valid_dates.empty:
        st.selectbox("🗓️ Waktu", ["Semua waktu"], disabled=True)
        return df

    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()

    preset = st.selectbox(
        "🗓️ Waktu",
        ["Semua waktu", "7 hari terakhir", "30 hari terakhir", "Rentang kustom"],
    )

    if preset == "Semua waktu":
        return df

    if preset == "7 hari terakhir":
        start, end = max_date - timedelta(days=7), max_date
    elif preset == "30 hari terakhir":
        start, end = max_date - timedelta(days=30), max_date
    else:
        picked = st.date_input(
            "Pilih rentang tanggal",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(picked, (list, tuple)) and len(picked) == 2:
            start, end = picked
        else:
            # Kalau pengguna baru pilih satu tanggal (belum lengkap), tampilkan semua dulu.
            return df

    mask = (dcol.dt.date >= start) & (dcol.dt.date <= end)
    return df[mask]


# ----------------------------------------------------------------------
# Trending: topik/berita yang sedang naik daun (lonjakan volume terbaru)
# ----------------------------------------------------------------------
def compute_trending_topics(df: pd.DataFrame, labels: dict, window_days: int = 3, min_articles: int = 2) -> pd.DataFrame:
    """
    Skor trending sederhana per topik: gabungan antara VOLUME berita terbaru
    (dalam `window_days` hari terakhir) dan RASIO kebaruan (seberapa besar
    porsi berita topik itu yang baru muncul belakangan, dibanding riwayatnya).
    Topik dengan banyak berita baru DAN porsi baru yang tinggi = paling trending.
    """
    tdf = df.copy()
    if "published_at" not in tdf.columns or "cluster" not in tdf.columns:
        return pd.DataFrame()
    tdf["published_at"] = pd.to_datetime(tdf["published_at"], errors="coerce")
    tdf = tdf.dropna(subset=["published_at"])
    if tdf.empty:
        return pd.DataFrame()

    max_date = tdf["published_at"].max()
    window_start = max_date - pd.Timedelta(days=window_days)

    recent = tdf[tdf["published_at"] >= window_start]
    recent_counts = recent["cluster"].value_counts()
    total_counts = tdf["cluster"].value_counts()

    rows = []
    for cid, recent_count in recent_counts.items():
        if recent_count < min_articles:
            continue
        total_count = total_counts.get(cid, recent_count)
        recency_ratio = recent_count / total_count if total_count else 0
        rows.append({
            "cluster": cid,
            "topik": labels.get(cid, labels.get(str(cid), f"Cluster {cid}")),
            "recent_count": int(recent_count),
            "total_count": int(total_count),
            "recency_ratio": recency_ratio,
        })

    trending_df = pd.DataFrame(rows)
    if trending_df.empty:
        return trending_df

    trending_df["trend_score"] = trending_df["recent_count"] * (0.5 + trending_df["recency_ratio"])
    return trending_df.sort_values("trend_score", ascending=False).reset_index(drop=True)


def render_trending(df: pd.DataFrame, labels: dict):
    st.subheader("🔥 Topik & Berita Trending")

    window_label = st.radio(
        "Hitung tren berdasarkan aktivitas:", ["3 hari terakhir", "7 hari terakhir", "14 hari terakhir"],
        horizontal=True,
    )
    window_days = {"3 hari terakhir": 3, "7 hari terakhir": 7, "14 hari terakhir": 14}[window_label]

    trending_df = compute_trending_topics(df, labels, window_days=window_days)
    if trending_df.empty:
        st.info(
            "Belum cukup data bertanggal untuk menghitung tren. Biarkan auto-refresh berjalan "
            "beberapa hari (atau jalankan koleksi manual di hari berbeda) agar tren mulai terbentuk."
        )
        return

    top_trending = trending_df.head(5)
    cols = st.columns(len(top_trending))
    for rank, (col, (_, row)) in enumerate(zip(cols, top_trending.iterrows()), start=1):
        with col:
            st.markdown(
                f"""<div class="news-card">
                    <div class="meta">🔥 #{rank} Trending</div>
                    <h4>{row['topik']}</h4>
                    <div class="meta">{row['recent_count']} berita baru • {row['total_count']} total</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("**📰 Berita terbaru dari topik-topik trending:**")

    tdf = df.copy()
    tdf["published_at"] = pd.to_datetime(tdf["published_at"], errors="coerce")
    trending_cluster_ids = top_trending["cluster"].tolist()
    trending_news = (
        tdf[tdf["cluster"].isin(trending_cluster_ids)]
        .sort_values("published_at", ascending=False)
        .head(15)
    )

    if trending_news.empty:
        st.info("Belum ada berita spesifik untuk topik trending saat ini.")
        return

    for _, row in trending_news.iterrows():
        topik = labels.get(row.get("cluster"), labels.get(str(row.get("cluster")), ""))
        pub = row.get("published_at")
        pub_str = pub.strftime("%d %b %Y, %H:%M") if pd.notna(pub) else "-"
        link = row.get("link", "")
        title = row.get("title", "(tanpa judul)")
        title_html = (
            f'<a href="{link}" target="_blank" style="text-decoration:none;">{title}</a>' if link else title
        )
        st.markdown(
            f"""<div class="news-card">
                <div class="meta">🔥 {pub_str} • {row.get('category', '') or 'Umum'} • {topik}</div>
                <h4>{title_html}</h4>
            </div>""",
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    config = get_config()

    with st.sidebar:
        st.header("⚙️ Pengaturan")

        with st.expander("🔌 Sumber Data", expanded=False):
            feed_url = st.text_input("URL RSS Feed", value=config["rss"]["feed_url"])
            mode = st.radio(
                "Mode koleksi",
                ["full", "rss"],
                format_func=lambda m: "Isi lengkap + kategori + gambar" if m == "full" else "Ringkasan RSS saja (lebih cepat)",
                index=0 if config.get("scraping", {}).get("full_content", True) else 1,
            )

        with st.expander("🧩 Clustering", expanded=False):
            min_k = st.slider("Jumlah cluster minimum", 2, 10, config["clustering"]["min_k"])
            max_k = st.slider("Jumlah cluster maksimum", min_k, 15, max(min_k, config["clustering"]["max_k"]))

        st.divider()
        st.subheader("🔄 Auto-refresh")
        if AUTOREFRESH_AVAILABLE:
            auto_refresh = st.toggle("Aktifkan auto-refresh", value=False)
            refresh_minutes = st.slider("Interval (menit)", 5, 60, 15, disabled=not auto_refresh)
        else:
            auto_refresh = False
            refresh_minutes = 15
            st.caption(
                "⚠️ Paket `streamlit-autorefresh` belum terpasang. Jalankan "
                "`pip install streamlit-autorefresh` untuk mengaktifkan fitur ini."
            )

        run_button = st.button("🔄 Tarik Data Sekarang", type="primary", use_container_width=True)

        st.divider()
        search_query = st.text_input("🔍 Cari berita (judul/isi)", value="")

        st.caption(
            "Setiap koleksi menggabungkan berita baru dari Tatoli dengan riwayat yang sudah "
            "terkumpul (dedup otomatis), lalu melatih ulang model topik dari seluruh data."
        )

    if auto_refresh and AUTOREFRESH_AVAILABLE:
        st_autorefresh(interval=refresh_minutes * 60 * 1000, key="tatoli_autorefresh")

    if "news_result" not in st.session_state:
        existing = load_latest_from_disk(config)
        st.session_state["news_result"] = existing

    last_fetch = _read_last_fetch_time()
    should_auto_run = False
    if auto_refresh:
        if last_fetch is None or (datetime.now() - last_fetch).total_seconds() >= refresh_minutes * 60:
            should_auto_run = True

    if run_button or should_auto_run:
        with st.spinner("Menarik data terbaru dari Tatoli, membersihkan teks, dan mendeteksi topik..."):
            try:
                result = _execute_pipeline(config, feed_url, min_k, max_k, mode)
                extra = ""
                if "scraped_new_articles" in result["raw_paths"]:
                    extra = f", {result['raw_paths']['scraped_new_articles']} artikel di-scrape penuh"
                st.toast(
                    f"✅ Data diperbarui! {result['raw_paths']['new_entries']} berita baru{extra}.",
                    icon="✅",
                )
                last_fetch = datetime.now()
            except Exception as e:
                st.error(f"Terjadi kesalahan saat menjalankan pipeline: {e}")

    render_header(last_fetch)

    data = st.session_state.get("news_result")
    if not data:
        st.info("Klik **Tarik Data Sekarang** di sidebar untuk memulai pengumpulan berita pertama kali.")
        return

    df = data["df"]
    topics_payload = data["topics_payload"]
    labels = {t["cluster_id"]: t["label"] for t in topics_payload.get("topics", [])} if topics_payload else {}

    if "cluster" not in df.columns:
        st.warning("Data belum memiliki hasil clustering. Klik **Tarik Data Sekarang** untuk menjalankan analisis.")
        return

    render_stats(df)
    st.divider()

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        filtered_df = render_language_filter(df)
    with fcol2:
        filtered_df = render_category_filter(filtered_df)
    with fcol3:
        filtered_df = render_date_filter(filtered_df)

    tabs = st.tabs([
        "🏠 Beranda", "🔥 Trending", "📊 Distribusi Topik", "📈 Tren Topik",
        "☁️ Wordcloud", "🔑 Kata Kunci", "📰 Berita", "🖼️ Galeri",
    ])
    with tabs[0]:
        render_overview(filtered_df, labels)
    with tabs[1]:
        render_trending(filtered_df, labels)
    with tabs[2]:
        render_topic_distribution(filtered_df, labels)
    with tabs[3]:
        render_topic_trend(filtered_df, labels)
    with tabs[4]:
        render_wordclouds(filtered_df, labels)
    with tabs[5]:
        render_keywords_list(topics_payload)
    with tabs[6]:
        render_news_table(filtered_df, labels, search_query)
    with tabs[7]:
        render_gallery(filtered_df, labels)


if __name__ == "__main__":
    main()
