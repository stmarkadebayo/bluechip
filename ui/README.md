# Bluechip Demo Console

The demo is a dependency-free static browser UI mounted by FastAPI at `/ui/`.

Flows:

- Task B Recommend: calls `POST /api/profile-user` and `POST /api/recommend`.
- Task A Review: calls `POST /api/profile-user` and `POST /api/simulate-review`.
- Runtime Trace: calls `GET /api/metrics` and `GET /api/traces`.

The first screen is the usable demo console. It includes cold-start, cross-domain, and strict-reviewer examples; persona/history inputs; product or candidate JSON editors; ranked product output; generated review output; candidate sources; score components; profile evidence; validation status; trace steps; and raw JSON.

Run it with:

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/ui/
```

Submission demo flow:

```text
docs/product/DEMO_SCRIPT.md
```
