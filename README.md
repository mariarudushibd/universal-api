# Universal API

A universal API built with FastAPI, deployed on Vercel.

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
vercel dev
```

## Deploy

```bash
vercel deploy
```

## Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check
- `GET /api/hello?name=World` - Hello endpoint
