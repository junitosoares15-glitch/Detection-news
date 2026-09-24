#!/usr/bin/env python3
"""
main.py — Pipeline utama Mining Notísia Online no Deteksaun Topiku.

Alur: RSS Tatoli -> Koleksaun (data/raw) -> Preprocessamentu (data/cleaned)
-> Feature Extraction TF-IDF (data/models) -> Deteksaun Topiku K-Means (data/models)
-> Extrasaun Keyword (data/keywords) -> rezumu rezultadu.

Exemplu uza:
    python main.py
    python main.py --feed-url "https://tatoli.tl/feed/" --min-k 2 --max-k 8

Ba dashboard interativu, halo run:
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

SOURCE_NAME = "tatoli"


def run_pipeline(config: dict, feed_url: str = None, min_k=None, max_k=None, collection_mode: str = None):
    logger = logging.getLogger("main")

    feed_url = feed_url or config["rss"]["feed_url"]
    source_name = config["rss"].get("source_name", "Tatoli")
    scraping_cfg = config.get("scraping", {})

    mode = collection_mode or ("full" if scraping_cfg.get("full_content", True) else "rss")

    logger.info("=== Hahu pipeline Mining Notísia (mode=%s) ba feed: %s ===", mode, feed_url)

    # 1) Koleksaun: simu & rai notísia brutu iha data/raw
    if mode == "full":
        logger.info("[1/5] Simu RSS (discovery) + scraping kontaudu kompletu ...")
        raw_paths = collect_full_articles(
            feed_url,
            config["paths"]["raw"],
            source_name=source_name,
            max_articles=scraping_cfg.get("max_articles_per_run"),
            delay_seconds=scraping_cfg.get("delay_seconds", 1.5),
            selectors=scraping_cfg.get("selectors"),
        )
        logger.info(
            "Artigu foun scrape kompletu: %s (fail: %s) | Total notísia uniku: %s",
            raw_paths.get("scraped_new_articles", 0), raw_paths.get("failed_articles", 0),
            raw_paths["total_unique"],
        )
    else:
        logger.info("[1/5] Simu RSS feed (rezumu deit) ...")
        raw_paths = collect_rss(feed_url, config["paths"]["raw"], source_name=source_name)
        new_entries = raw_paths.get("new_entries", raw_paths.get("new_articles", 0))
        total_unique = raw_paths.get("total_unique", raw_paths.get("total", len(raw_paths.get("all_entries", []))))
        logger.info(
            "Notísia foun iha run ida-ne’e: %s | Total notísia uniku simu: %s",
            new_entries, total_unique,
        )

    # 2) Preprocessamentu: limpeza, deteksaun lian, stopword removal, stemming
    logger.info("[2/5] Halo preprocessamentu ba teks (limpeza + stemming) ...")
    cleaned_master_path = run_preprocessing(raw_paths["master_csv"], config["paths"]["cleaned"])

    cleaned_df = pd.read_csv(cleaned_master_path)

    # 3) Feature Extraction: TF-IDF
    logger.info("[3/5] Hahu halo representasaun TF-IDF ...")
    clustering_cfg = config.get("clustering", {})
    texts = cleaned_df["text_final"].fillna("").tolist()
    X, extractor = run_feature_extraction(
        texts,
        models_dir=config["paths"]["models"],
        name=SOURCE_NAME,
        max_features=clustering_cfg.get("max_features", 5000),
        ngram_range=(clustering_cfg.get("ngram_min", 1), clustering_cfg.get("ngram_max", 2)),
        min_df=clustering_cfg.get("min_df", 2),
    )

    # 4) Deteksaun Topiku: K-Means (k otimal via silhouette score)
    logger.info("[4/5] Halo clustering K-Means ba deteksaun topiku ...")
    labels, topic_model = run_clustering(
        X,
        models_dir=config["paths"]["models"],
        name=SOURCE_NAME,
        min_k=min_k or clustering_cfg.get("min_k", 2),
        max_k=max_k or clustering_cfg.get("max_k", 10),
    )
    cleaned_df["cluster"] = labels
    cleaned_df.to_csv(cleaned_master_path, index=False, encoding="utf-8-sig")

    # 5) Extrasaun Keyword
    logger.info("[5/5] Extrai liafuan-chave husi topiku ida-idak ...")
    top_n = config.get("keywords", {}).get("top_n_terms", 10)
    topics, topic_labels, keywords_path = run_keyword_extraction(
        extractor.vectorizer, topic_model.kmeans, cleaned_df,
        config["paths"]["keywords"], top_n=top_n, name=SOURCE_NAME,
    )

    logger.info("=== Pipeline remata ===")

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


def print_summary(result: dict):
    df = result["cleaned_df"]
    raw_paths = result["raw_paths"]

    print("\n" + "=" * 70)
    print(f"Total notísia (kumulativu) : {len(df)}")
    print(f"Numeru topiku detektadu    : {result['topic_model'].best_k}")
    if "scraped_new_articles" in raw_paths:
        print(f"Artigu scrape kompletu iha run ida-ne’e : {raw_paths['scraped_new_articles']}")
        if raw_paths.get("failed_articles"):
            print(f"Artigu fail scrape (fallback ba rezumu RSS) : {raw_paths['failed_articles']}")
    print("-" * 70)
    print("Topiku detektadu:")
    for cid, words in result["topics"].items():
        size = int((df["cluster"] == cid).sum())
        label = result["topic_labels"].get(cid, "")
        print(f"  [Cluster {cid}] ({size} notísia) {label}")
        print(f"      liafuan-chave: {', '.join(words)}")
    print("-" * 70)
    print(f"Dadus brutu (master)  -> {result['raw_paths']['master_csv']}")
    print(f"Dadus limpu (master)  -> {result['cleaned_master_path']}")
    print(f"Keywords/topiku       -> {result['keywords_path']}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Mining Notísia Online no Deteksaun Topiku — pipeline utama"
    )
    parser.add_argument("--feed-url", default=None, help="URL RSS feed (default husi config/settings.yaml)")
    parser.add_argument("--config", default=None, help="Path ba settings.yaml (opsional)")
    parser.add_argument("--min-k", type=int, default=None, help="Numeru cluster minimum")
    parser.add_argument("--max-k", type=int, default=None, help="Numeru cluster maksimum")
    parser.add_argument(
        "--mode", choices=["full", "rss"], default=None,
        help="'full' = scraping kontaudu kompletu+kategoria+imajen; 'rss' = rezumu RSS deit. "
             "Default: tuir config/settings.yaml (scraping.full_content).",
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
        logging.getLogger("main").exception("Pipeline fail: %s", e)
        print(f"\n[ERRO] Pipeline fail: {e}\n", file=sys.stderr)
        sys.exit(1)

    print_summary(result)


if __name__ == "__main__":
    main()
