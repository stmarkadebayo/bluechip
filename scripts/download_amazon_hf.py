from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CATEGORIES = (
    "All_Beauty",
    "Digital_Music",
    "Magazine_Subscriptions",
    "Gift_Cards",
    "Subscription_Boxes",
)

CATEGORIES = (
    "All_Beauty",
    "Amazon_Fashion",
    "Appliances",
    "Arts_Crafts_and_Sewing",
    "Automotive",
    "Baby_Products",
    "Beauty_and_Personal_Care",
    "Books",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry",
    "Digital_Music",
    "Electronics",
    "Gift_Cards",
    "Grocery_and_Gourmet_Food",
    "Handmade_Products",
    "Health_and_Household",
    "Health_and_Personal_Care",
    "Home_and_Kitchen",
    "Industrial_and_Scientific",
    "Kindle_Store",
    "Magazine_Subscriptions",
    "Movies_and_TV",
    "Musical_Instruments",
    "Office_Products",
    "Patio_Lawn_and_Garden",
    "Pet_Supplies",
    "Software",
    "Sports_and_Outdoors",
    "Subscription_Boxes",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
    "Unknown",
    "Video_Games",
)

REVIEW_BYTES = {
    "All_Beauty": 326_611_506,
    "Digital_Music": 78_823_304,
    "Magazine_Subscriptions": 33_297_013,
    "Gift_Cards": 50_231_035,
    "Subscription_Boxes": 8_953_020,
}

METADATA_BYTES = {
    "All_Beauty": 212_990_142,
    "Digital_Music": 67_097_002,
    "Magazine_Subscriptions": 4_096_541,
    "Gift_Cards": 2_038_189,
    "Subscription_Boxes": 1_399_864,
}

REVIEW_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    "raw/review_categories/{category}.jsonl?download=true"
)
METADATA_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    "raw/meta_categories/meta_{category}.jsonl?download=true"
)


@dataclass(frozen=True)
class DownloadTarget:
    category: str
    kind: str
    url: str
    path: Path
    expected_bytes: int


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and validate selected Amazon Reviews 2023 categories from Hugging Face."
    )
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--categories", nargs="*", choices=CATEGORIES, default=list(DEFAULT_CATEGORIES))
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Download every Amazon Reviews 2023 review category. This is hundreds of GB.",
    )
    parser.add_argument("--with-metadata", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any target is incomplete.")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument(
        "--max-seconds-per-file",
        type=int,
        default=0,
        help="Optional wall-clock cap for each file attempt; 0 means no cap.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    categories = list(CATEGORIES) if args.all_categories else args.categories
    targets = build_targets(output_dir, categories, include_metadata=args.with_metadata)

    incomplete = []
    for target in targets:
        status = status_for(target)
        print(format_status(target, status))
        if args.check_only:
            if not _is_usable_status(status):
                incomplete.append(target)
            continue

        final_status = download_target(
            target,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
            max_seconds_per_file=args.max_seconds_per_file,
        )
        print(format_status(target, final_status))
        if not _is_usable_status(final_status):
            incomplete.append(target)

    if incomplete and args.strict:
        raise SystemExit(1)


def build_targets(
    output_dir: Path,
    categories: list[str],
    include_metadata: bool,
) -> list[DownloadTarget]:
    targets = []
    for category in categories:
        targets.append(
            DownloadTarget(
                category=category,
                kind="reviews",
                url=REVIEW_URL.format(category=category),
                path=output_dir / f"{category}.jsonl",
                expected_bytes=REVIEW_BYTES.get(category, 0),
            )
        )
        if include_metadata:
            targets.append(
                DownloadTarget(
                    category=category,
                    kind="metadata",
                    url=METADATA_URL.format(category=category),
                    path=output_dir / f"meta_{category}.jsonl",
                    expected_bytes=METADATA_BYTES.get(category, 0),
                )
            )
    return targets


def status_for(target: DownloadTarget) -> str:
    if not target.path.exists():
        return "missing"
    size = target.path.stat().st_size
    if target.expected_bytes <= 0:
        return f"present_unverified:{size}"
    if size == target.expected_bytes:
        return "complete"
    if size > target.expected_bytes:
        return f"oversized:{size}"
    return f"partial:{size}/{target.expected_bytes}"


def download_target(
    target: DownloadTarget,
    retries: int,
    timeout_seconds: int,
    max_seconds_per_file: int,
) -> str:
    target.path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        status = status_for(target)
        if _is_usable_status(status):
            return status

        try:
            _download_once(
                target,
                timeout_seconds=timeout_seconds,
                max_seconds_per_file=max_seconds_per_file,
            )
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            print(f"download warning: {target.path.name} attempt {attempt}/{retries}: {exc}")
            time.sleep(min(attempt * 2, 20))

    return status_for(target)


def _download_once(
    target: DownloadTarget,
    timeout_seconds: int,
    max_seconds_per_file: int,
) -> None:
    existing = target.path.stat().st_size if target.path.exists() else 0
    started = time.monotonic()
    headers = {"User-Agent": "bluechip-hackathon-downloader/0.1"}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    request = urllib.request.Request(target.url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        append = existing > 0 and getattr(response, "status", 200) == 206
        mode = "ab" if append else "wb"
        with target.path.open(mode) as handle:
            while True:
                if max_seconds_per_file and time.monotonic() - started > max_seconds_per_file:
                    raise TimeoutError(f"download exceeded {max_seconds_per_file}s cap")
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                handle.write(chunk)


def format_status(target: DownloadTarget, status: str) -> str:
    return f"{target.category} {target.kind}: {status} -> {target.path}"


def _is_usable_status(status: str) -> bool:
    return status == "complete" or status.startswith(("oversized:", "present_unverified:"))


if __name__ == "__main__":
    main()
