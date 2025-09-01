Public mode instructions:

Render (no-key):
- Remove API_KEY from env or set it empty. Redeploy.

Local demo:
- export ALLOWED_ORIGINS='*'
- unset API_KEY; export MAX_BODY_BYTES=10485760
- ./run_gunicorn.sh  # or uvicorn backend.api.main:app --host 0.0.0.0 --port 8080

Browser paths:
- /docs, /openapi.json, /healthz, /readyz are public.
