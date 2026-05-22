from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test a running Bluechip API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    _wait_for_health(base_url, args.timeout)
    _post_json(
        f"{base_url}/api/simulate-review",
        {
            "user_persona": "A practical Lagos diner who values quiet service and fair prices.",
            "user_history": [
                {
                    "item_id": "history-1",
                    "item_name": "Quiet Bowl",
                    "rating": 5,
                    "review": "Quiet, affordable, and quick.",
                    "category": "restaurant",
                }
            ],
            "target_item": {
                "item_id": "target-1",
                "name": "Calm Grill",
                "category": "restaurant",
                "metadata": {"ambience": "quiet", "review_count": 10},
                "summary": "A quiet grill with fast service.",
                "average_rating": 4.5,
            },
            "locale": "Nigeria",
        },
        required_keys=("trace_id", "predicted_rating", "review"),
    )
    recommend_payload = _post_json(
        f"{base_url}/api/recommend",
        {
            "user_persona": "A practical Lagos diner who values quiet service and fair prices.",
            "user_history": [],
            "context": "Dinner where conversation is easy.",
            "candidate_items": [
                {
                    "item_id": "candidate-1",
                    "name": "Calm Grill",
                    "category": "restaurant",
                    "metadata": {"ambience": "quiet", "review_count": 10},
                    "summary": "A quiet grill with fast service.",
                    "average_rating": 4.5,
                }
            ],
            "limit": 1,
        },
        required_keys=("trace_id", "recommendations", "candidate_diagnostics"),
    )
    if not recommend_payload["recommendations"]:
        raise SystemExit("recommend returned no recommendations")
    print("API smoke passed: health, simulate-review, and recommend")


def _wait_for_health(base_url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        try:
            payload = _get_json(f"{base_url}/api/health")
        except Exception as exc:
            last_error = str(exc)
            time.sleep(1)
            continue
        if payload == {"status": "ok"}:
            return
        last_error = f"unexpected health payload: {payload}"
        time.sleep(1)
    raise SystemExit(f"health check failed: {last_error}")


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict, required_keys: tuple[str, ...]) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"{url} failed with HTTP {exc.code}: {body}") from exc
    missing = [key for key in required_keys if key not in result]
    if missing:
        raise SystemExit(f"{url} response missing keys: {missing}")
    return result


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"API smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
