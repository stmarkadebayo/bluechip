from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET_ID = "McAuley-Lab/Yelp"
FALLBACK_DATASET_ID = "yelp_review_full"

YELP_REVIEW_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Yelp/resolve/main/"
    "raw/yelp_academic_dataset_review.json?download=true"
)
YELP_BUSINESS_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Yelp/resolve/main/"
    "raw/yelp_academic_dataset_business.json?download=true"
)


@dataclass(frozen=True)
class DownloadTarget:
    name: str
    kind: str
    url: str
    path: Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Yelp review data from HuggingFace and normalise to project JSONL format."
    )
    parser.add_argument("--output-dir", default="data/raw/yelp")
    parser.add_argument("--processed-dir", default="data/processed/yelp")
    parser.add_argument("--limit", type=int, default=0, help="Optional max reviews for local experiments.")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--skip-download", action="store_true", help="Skip download, only normalise local files.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    processed_dir = Path(args.processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        raw_reviews = _download_yelp(
            output_dir=output_dir,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        raw_reviews = _load_local_json(output_dir / "yelp_reviews.json") or _load_local_json(
            output_dir / "yelp_reviews.jsonl"
        )
        if not raw_reviews:
            raw_reviews = _try_load_jsonl(output_dir)

    if not raw_reviews:
        print("No Yelp review data found. Trying fallback dataset via HuggingFace datasets library...")
        raw_reviews = _download_via_datasets_library(args.limit)
        if not raw_reviews:
            raise SystemExit(
                "Could not obtain Yelp review data. "
                "The Yelp academic dataset requires an official download flow. "
                "Try downloading manually from https://www.yelp.com/dataset and placing "
                "yelp_academic_dataset_review.json in data/raw/yelp/."
            )

    _normalise_and_save(raw_reviews, processed_dir, args.limit)


def _download_via_datasets_library(limit: int) -> list[dict[str, Any]]:
    """Attempt to download Yelp data using the HuggingFace datasets library."""
    for dataset_id in (DATASET_ID, FALLBACK_DATASET_ID):
        try:
            from datasets import load_dataset

            print(f"Attempting to load '{dataset_id}' via HuggingFace datasets library...")
            dataset = load_dataset(dataset_id, split="train", streaming=True)
            rows: list[dict[str, Any]] = []
            for index, row in enumerate(dataset):
                rows.append(dict(row))
                if limit and index + 1 >= limit:
                    break
            print(f"  Downloaded {len(rows)} rows from '{dataset_id}'.")
            return rows
        except ImportError:
            print("  HuggingFace datasets library not installed. Install with: pip install datasets")
        except Exception as exc:
            print(f"  Failed to load '{dataset_id}': {exc}")
    return []


def _download_yelp(
    output_dir: Path,
    retries: int,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    """Download Yelp data via direct HuggingFace URLs."""
    review_path = output_dir / "yelp_reviews.json"
    business_path = output_dir / "yelp_businesses.json"

    targets = [
        DownloadTarget(name="yelp_reviews", kind="reviews", url=YELP_REVIEW_URL, path=review_path),
        DownloadTarget(name="yelp_businesses", kind="businesses", url=YELP_BUSINESS_URL, path=business_path),
    ]

    for target in targets:
        print(f"Downloading {target.name}...")
        success = _download_with_retries(target, retries=retries, timeout_seconds=timeout_seconds)
        if success:
            size_mb = target.path.stat().st_size / (1024 * 1024)
            print(f"  {target.name}: {size_mb:.1f} MB -> {target.path}")
        else:
            print(f"  {target.name}: download failed or unavailable.")

    raw_reviews = _load_local_json(review_path)
    if raw_reviews:
        businesses = _load_local_json(business_path)
        if businesses:
            raw_reviews = _attach_business_info(raw_reviews, businesses)
    return raw_reviews


def _download_with_retries(target: DownloadTarget, retries: int, timeout_seconds: int) -> bool:
    """Download a single file with retry logic similar to download_amazon_hf.py."""
    if target.path.exists():
        print(f"  {target.name}: already exists at {target.path}")
        return True

    for attempt in range(1, retries + 1):
        try:
            _download_once(target, timeout_seconds)
            return True
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            print(f"  download warning: {target.name} attempt {attempt}/{retries}: {exc}")
            time.sleep(min(attempt * 3, 30))
    return False


def _download_once(target: DownloadTarget, timeout_seconds: int) -> None:
    """Download a file from URL, with streaming writes."""
    import json as _json

    headers = {"User-Agent": "bluechip-hackathon-downloader/0.1"}
    request = urllib.request.Request(target.url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read()
        try:
            payload = _json.loads(raw.decode("utf-8"))
        except (_json.JSONDecodeError, UnicodeDecodeError):
            payload = _parse_json_lines(raw.decode("utf-8", errors="replace"))

        target.path.parent.mkdir(parents=True, exist_ok=True)
        target.path.write_text(_json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _parse_json_lines(text: str) -> list[dict[str, Any]]:
    """Try to parse JSON lines format."""
    import json as _json

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(_json.loads(line))
        except _json.JSONDecodeError:
            continue
    return rows


def _load_local_json(path: Path) -> list[dict[str, Any]]:
    """Load JSON data from a local file."""
    import json as _json

    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
        return _json.loads(text)
    except (_json.JSONDecodeError, OSError):
        return []


def _try_load_jsonl(directory: Path) -> list[dict[str, Any]]:
    """Try loading any JSONL file in the directory."""
    for path in sorted(directory.glob("*.jsonl")):
        try:
            rows = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    import json as _json
                    rows.append(_json.loads(line))
            if rows:
                print(f"  Loaded {len(rows)} rows from {path}")
                return rows
        except Exception:
            continue
    return []


def _attach_business_info(
    reviews: list[dict[str, Any]],
    businesses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Enrich review rows with business name and categories."""
    if not isinstance(reviews, list):
        return reviews
    if not isinstance(businesses, list):
        return reviews

    business_map: dict[str, dict[str, Any]] = {}
    for biz in businesses:
        bid = str(biz.get("business_id") or "")
        if bid:
            business_map[bid] = biz

    for review in reviews:
        bid = str(review.get("business_id") or "")
        biz = business_map.get(bid, {})
        if not review.get("name"):
            review["name"] = biz.get("name") or review.get("business_name") or ""
        if not review.get("categories"):
            cats = biz.get("categories")
            if isinstance(cats, str):
                review["categories"] = cats
            elif isinstance(cats, list):
                review["categories"] = ", ".join(str(c) for c in cats)

    return reviews


def _normalise_and_save(
    raw_reviews: list[dict[str, Any]],
    processed_dir: Path,
    limit: int,
) -> None:
    """Normalise Yelp reviews to the project JSONL format and save."""
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from scripts.data_utils import stable_review_id, write_json, write_jsonl

    reviews: list[dict[str, Any]] = []
    reviews_by_item: dict[str, list[dict[str, Any]]] = {}
    item_ids: set[str] = set()

    print(f"\nNormalising {len(raw_reviews)} Yelp review rows...")

    for index, row in enumerate(raw_reviews):
        if limit and index >= limit:
            break
        if not isinstance(row, dict):
            continue

        user_id = str(row.get("user_id") or row.get("reviewer_id") or "")
        item_id = str(row.get("business_id") or row.get("item_id") or "")
        if not user_id or not item_id:
            continue

        rating = float(row.get("stars") or row.get("rating") or row.get("label") or 0)
        if isinstance(rating, int) and rating > 5:
            rating = float(rating)
        review_text = str(row.get("text") or row.get("review") or "")
        if not review_text or rating < 1 or rating > 5:
            continue

        timestamp = _parse_timestamp(row)
        item_name = str(row.get("name") or row.get("business_name") or item_id)
        category = str(row.get("categories") or row.get("category") or "yelp")

        normalised = {
            "review_id": stable_review_id(user_id, item_id, timestamp, index),
            "user_id": user_id,
            "item_id": item_id,
            "item_name": item_name,
            "rating": rating,
            "review": review_text,
            "category": category,
            "timestamp": timestamp,
        }
        reviews.append(normalised)
        reviews_by_item.setdefault(item_id, []).append(normalised)
        item_ids.add(item_id)

    items: list[dict[str, Any]] = []
    for item_id in sorted(item_ids):
        item_reviews = reviews_by_item[item_id]
        item_name = item_reviews[0]["item_name"]
        category = item_reviews[0]["category"]
        for review in item_reviews:
            review["item_name"] = item_name
            review["category"] = category
        avg = sum(review["rating"] for review in item_reviews) / len(item_reviews)
        snippets = [r["review"].replace("\n", " ")[:120] for r in item_reviews[:3]]
        items.append(
            {
                "item_id": item_id,
                "name": item_name,
                "category": category,
                "metadata": {"rating_number": len(item_reviews)},
                "summary": " ".join(snippets)[:360],
                "average_rating": round(avg, 3),
            }
        )

    write_jsonl(processed_dir / "reviews.jsonl", reviews)
    write_jsonl(processed_dir / "items.jsonl", items)
    write_json(
        processed_dir / "dataset_stats.json",
        {
            "source": "Yelp",
            "reviews": len(reviews),
            "items": len(items),
            "users": len({review["user_id"] for review in reviews}),
            "category": "yelp",
        },
    )

    print(f"  Normalised: {len(reviews)} reviews, {len(items)} items")
    print(f"  Saved to: {processed_dir}")
    print(f"  - {processed_dir / 'reviews.jsonl'}")
    print(f"  - {processed_dir / 'items.jsonl'}")
    print(f"  - {processed_dir / 'dataset_stats.json'}")


def _parse_timestamp(row: dict[str, Any]) -> int:
    """Parse Yelp date format into a Unix timestamp."""
    raw_date = row.get("date") or row.get("timestamp") or ""
    if isinstance(raw_date, (int, float)):
        return int(raw_date)
    if not isinstance(raw_date, str) or not raw_date.strip():
        return 0
    try:
        from datetime import datetime

        return int(datetime.strptime(raw_date[:10], "%Y-%m-%d").timestamp())
    except (ValueError, ImportError):
        return 0


if __name__ == "__main__":
    main()
