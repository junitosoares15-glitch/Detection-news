#!/usr/bin/env python3
"""
main.py — Pipeline utama Mining Notísia Online no Deteksaun Topiku.

Alur: RSS/Web (multi-sumber) -> Collection (data/raw) -> Preprocessing
(data/cleaned) -> Feature Extraction TF-IDF (data/models) -> Topic Detection
K-Means (data/models) -> Keyword Extraction (data/keywords) -> ringkasan hasil.

Sumber berita dikonfigurasi di config/settings.yaml bagian `sources` — bisa
lebih dari satu portal berita sekaligus (Tatoli, Timor Post, Independente,
dst). Semua sumber yang `enabled: true` akan ditarik dan digabung menjadi
satu dataset terpadu (kolom `source` menandai asal tiap berita).

Contoh pemakaian:
    python main.py
    python main.py --min-k 2 --max-k 8
    python main.py --feed-url "https://tatoli.tl/feed/"   # override: hanya 1 sumber ini

Untuk dashboard interaktif, jalankan:
    streamlit run src/dashboard/trend_dashboard.py
"""
import argparse
import logging
import sys

from src.utils import load_config, setup_logging
from src.scraping.rss_scraper import collect_rss
from src.scraping.web_scraper import collect_full_articles
from src.preprocessing.pipeline import run_preprocessing
from src.feature_extraction.tfidf_vectorizer import run_feature_extraction
from src.clustering.kmeans_model import run_clustering
from src.keywords.keyword_extractor import run_keyword_extraction

import pandas as pd

# Identifier generik untuk nama file model (TF-IDF, KMeans, keywords) — sudah
# tidak "tatoli"-spesifik karena sekarang bisa menggabungkan banyak sumber.
MODEL_NAME = "news"


def _resolve_sources(config: dict, feed_url_override: str = None) -> list:
    """
    Tentukan daftar sumber yang akan dikoleksi. Kalau `feed_url_override`
    diberikan (mis. lewat --feed-url di CLI), pakai HANYA itu sebagai satu
    sumber. Kalau tidak, pakai semua entri `sources` yang enabled: true di
    config/settings.yaml (fallback ke `rss.feed_url` lama kalau `sources`
    kosong, demi kompatibilitas mundur).
    """
    if feed_url_override:
        return [{"name": "Custom", "feed_url": feed_url_override, "enabled": True}]

    sources = [s for s in config.get("sources", []) if s.get("enabled", True)]
    if sources:
        return sources

    # Fallback lama: satu sumber dari config["rss"]
    return [{
        "name": config["rss"].get("source_name", "Tatoli"),
        "feed_url": config["rss"]["feed_url"],
        "enabled": True,
    }]


def _collect_all_sources(config: dict, sources: list, mode: str) -> dict:
    """
    Loop koleksi untuk SEMUA sumber yang dikonfigurasi. Kegagalan pada satu
    sumber (mis. feed URL salah/situs down) TIDAK menghentikan sumber lain —
    dicatat sebagai peringatan dan pipeline lanjut dengan sumber yang berhasil.
    """
    logger = logging.getLogger("main")
    scraping_cfg = config.get("scraping", {})

    per_source_results = []
    master_csv = None

    for src in sources:
        name, feed_url = src["name"], src["feed_url"]
        try:
            if mode == "full":
                result = collect_full_articles(
                    feed_url,
                    config["paths"]["raw"],
                    source_name=name,
                    max_articles=scraping_cfg.get("max_articles_per_run"),
                    delay_seconds=scraping_cfg.get("delay_seconds", 1.5),
                    selectors=scraping_cfg.get("selectors"),
                )
                logger.info(
                    "[%s] Artikel baru di-scrape penuh: %s (gagal: %s) | Berita baru: %s",
                    name, result.get("scraped_new_articles", 0),
                    result.get("failed_articles", 0), result["new_entries"],
                )
            else:
                result = collect_rss(feed_url, config["paths"]["raw"], source_name=name)
                logger.info("[%s] Berita baru pada run ini: %s", name, result["new_entries"])

            result["source_name"] = name
            per_source_results.append(result)
            master_csv = result["master_csv"]  # sama untuk semua sumber (file gabungan)

        except Exception as e:
            logger.warning(
                "[%s] Gagal mengambil data dari sumber ini: %s. Melanjutkan ke sumber lain.",
                name, e,
            )
            per_source_results.append({"source_name": name, "error": str(e)})

    successful = [r for r in per_source_results if "error" not in r]
    if not successful:
        raise RuntimeError(
            "Semua sumber berita gagal diambil. Cek koneksi internet, atau "
            "verifikasi ulang feed_url tiap sumber di config/settings.yaml."
        )

    return {
        "master_csv": master_csv,
        "total_unique": successful[-1]["total_unique"],
        "per_source": per_source_results,
        "total_new_entries": sum(r.get("new_entries", 0) for r in successful),
        "total_scraped_articles": sum(r.get("scraped_new_articles", 0) for r in successful),
        "total_failed_articles": sum(r.get("failed_articles", 0) for r in successful),
    }


def run_pipeline(config: dict, feed_url: str = None, min_k=None, max_k=None, collection_mode: str = None):
    logger = logging.getLogger("main")

    scraping_cfg = config.get("scraping", {})
    mode = collection_mode or ("full" if scraping_cfg.get("full_content", True) else "rss")
    sources = _resolve_sources(config, feed_url_override=feed_url)

    source_names = ", ".join(s["name"] for s in sources)
    logger.info("=== Memulai pipeline Mining Notísia (mode=%s) | Sumber: %s ===", mode, source_names)

    # 1) Collection: ambil & simpan berita mentah dari SEMUA sumber ke data/raw
    logger.info("[1/5] Mengambil data dari %s sumber ...", len(sources))
    raw_paths = _collect_all_sources(config, sources, mode)
    logger.info(
        "Total berita baru: %s | Total berita unik (kumulatif, semua sumber): %s",
        raw_paths["total_new_entries"], raw_paths["total_unique"],
    )

    # 2) Preprocessing: cleaning, deteksi bahasa, stopword removal, stemming
    logger.info("[2/5] Menjalankan preprocessing teks (cleaning + stemming) ...")
    cleaned_master_path = run_preprocessing(raw_paths["master_csv"], config["paths"]["cleaned"])

    cleaned_df = pd.read_csv(cleaned_master_path)

    # 3) Feature Extraction: TF-IDF
    logger.info("[3/5] Membangun representasi TF-IDF ...")
    clustering_cfg = config.get("clustering", {})
    texts = cleaned_df["text_final"].fillna("").tolist()
    X, extractor = run_feature_extraction(
        texts,
        models_dir=config["paths"]["models"],
        name=MODEL_NAME,
        max_features=clustering_cfg.get("max_features", 5000),
        ngram_range=(clustering_cfg.get("ngram_min", 1), clustering_cfg.get("ngram_max", 2)),
        min_df=clustering_cfg.get("min_df", 2),
    )

    # 4) Topic Detection: K-Means (k optimal via silhouette score)
    logger.info("[4/5] Menjalankan clustering K-Means untuk deteksi topik ...")
    labels, topic_model = run_clustering(
        X,
        models_dir=config["paths"]["models"],
        name=MODEL_NAME,
        min_k=min_k or clustering_cfg.get("min_k", 2),
        max_k=max_k or clustering_cfg.get("max_k", 10),
    )
    cleaned_df["cluster"] = labels
    cleaned_df.to_csv(cleaned_master_path, index=False, encoding="utf-8-sig")

    # 5) Keyword Extraction
    logger.info("[5/5] Mengekstrak kata kunci tiap topik ...")
    top_n = config.get("keywords", {}).get("top_n_terms", 10)
    topics, topic_labels, keywords_path = run_keyword_extraction(
        extractor.vectorizer, topic_model.kmeans, cleaned_df,
        config["paths"]["keywords"], top_n=top_n, name=MODEL_NAME,
    )

    logger.info("=== Pipeline selesai ===")

    return {
        "raw_paths": raw_paths,
        "cleaned_master_path": cleaned_master_path,
        "cleaned_df": cleaned_df,
        "extractor": extractor,
        "topic_model": topic_model,
        "topics": topics,
        "topic_labels": topic_labels,
        "keywords_path": keywords_path,
    }


def print_summary(result: dict, target_per_source: int = None):
    df = result["cleaned_df"]
    raw_paths = result["raw_paths"]

    print("\n" + "=" * 70)
    if "source" in df.columns:
        source_counts = df["source"].value_counts().to_dict()
        print("Berita per sumber (progres koleksi):")
        for src, count in source_counts.items():
            if target_per_source:
                pct = min(100, round(count / target_per_source * 100, 1))
                bar_filled = int(pct / 5)
                bar = "█" * bar_filled + "░" * (20 - bar_filled)
                print(f"  - {src:<15} {count:>5} / {target_per_source} [{bar}] {pct}%")
            else:
                print(f"  - {src}: {count}")
    print(f"Total berita (kumulatif, semua sumber) : {len(df)}")
    print(f"Jumlah topik terdeteksi                : {result['topic_model'].best_k}")
    if raw_paths.get("total_scraped_articles"):
        print(f"Artikel di-scrape penuh pada run ini   : {raw_paths['total_scraped_articles']}")
        if raw_paths.get("total_failed_articles"):
            print(f"Artikel gagal di-scrape (fallback RSS) : {raw_paths['total_failed_articles']}")
    failed_sources = [r["source_name"] for r in raw_paths.get("per_source", []) if "error" in r]
    if failed_sources:
        print(f"⚠️  Sumber yang GAGAL diambil pada run ini: {', '.join(failed_sources)}")
    print("-" * 70)
    print("Topik terdeteksi:")
    for cid, words in result["topics"].items():
        size = int((df["cluster"] == cid).sum())
        label = result["topic_labels"].get(cid, "")
        print(f"  [Cluster {cid}] ({size} berita) {label}")
        print(f"      top terms: {', '.join(words)}")
    print("-" * 70)
    print(f"Data mentah (master gabungan) -> {result['raw_paths']['master_csv']}")
    print(f"Data bersih (master gabungan) -> {result['cleaned_master_path']}")
    print(f"Keywords/topik                -> {result['keywords_path']}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Mining Notísia Online no Deteksaun Topiku — pipeline utama"
    )
    parser.add_argument(
        "--feed-url", default=None,
        help="Override: pakai HANYA satu URL RSS feed ini, abaikan daftar `sources` di config.",
    )
    parser.add_argument("--config", default=None, help="Path ke settings.yaml (opsional)")
    parser.add_argument("--min-k", type=int, default=None, help="Jumlah cluster minimum")
    parser.add_argument("--max-k", type=int, default=None, help="Jumlah cluster maksimum")
    parser.add_argument(
        "--mode", choices=["full", "rss"], default=None,
        help="'full' = scraping isi artikel lengkap+kategori+gambar; 'rss' = ringkasan RSS saja. "
             "Default: ikuti config/settings.yaml (scraping.full_content).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config["paths"]["logs"])

    try:
        result = run_pipeline(
            config, feed_url=args.feed_url, min_k=args.min_k, max_k=args.max_k,
            collection_mode=args.mode,
        )
    except Exception as e:
        logging.getLogger("main").exception("Pipeline gagal: %s", e)
        print(f"\n[ERROR] Pipeline gagal: {e}\n", file=sys.stderr)
        sys.exit(1)

    target = config.get("scraping", {}).get("target_articles_per_source")
    print_summary(result, target_per_source=target)


if __name__ == "__main__":
    main()
