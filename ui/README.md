# UI Demo

The demo is a dependency-free static browser UI mounted by FastAPI at `/ui/`.

Tabs:

- Simulate Review: calls `POST /api/simulate-review`.
- Recommend: calls `POST /api/recommend`.
- Metrics: calls `GET /api/metrics` and `GET /api/traces`.

Each tab shows request inputs, structured JSON output, a human-readable summary, traces, and ranking score components when available.

Run it with:

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/ui/
```
