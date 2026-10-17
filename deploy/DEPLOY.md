# Deploying Kessler

## One-box Docker (recommended)

On any host with Docker (a $6/mo VPS is plenty - the pipeline peaks around
1.5 GB RAM during screening):

```bash
git clone <your-repo> kessler && cd kessler/deploy
cp ../.env.example ../.env        # fill in SPACETRACK_* and ANTHROPIC_API_KEY
docker compose up -d --build      # app + postgres, healthchecked
docker compose exec app python -m backend.app.ingestion.pipeline
```

The app listens on :8000. Set `KESSLER_REFRESH_INTERVAL_H` in the compose
environment to enable scheduled refreshes.

## TLS + domain

Put any reverse proxy in front. Caddy is the least-config option - on the
host (or as a compose service):

```
your.domain.com {
    reverse_proxy localhost:8000
}
```

Caddy provisions and renews certificates automatically. nginx + certbot works
identically if preferred.

## CI

`.github/workflows/ci.yml` runs on every push: the full backend suite on
SQLite, a syntax check of every frontend module, and the full suite again
against a real Postgres 16 service through the dialect adapter.

## Operational notes

- Back up the Postgres volume (or `data/kessler.db` on SQLite) - everything
  else is regenerable from the public sources.
- Space-Track etiquette: keep refresh intervals >= 1 h; the pipeline issues a
  bounded number of queries per run.
- The Kessler AI endpoint spends Anthropic API tokens per request; scope the
  key and monitor usage if the deployment is shared.
