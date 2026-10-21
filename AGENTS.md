# AGENTS.md — Personal Website

## Run

```bash
uvicorn app.main:app --reload --port 8002
# or
docker compose up
```

App runs on **port 8002**.

Deploy is Coolify (Docker Compose build pack, auto-deploy on push to `main`). `docker-compose.yml` intentionally publishes no ports — Coolify routes to internal port 8002. `docker-compose.override.yml` adds `127.0.0.1:8002` for local dev only.

Tests: `python3 tests/test_blog_security.py` (or `pytest tests`).

## Structure

- `app/main.py` — FastAPI entrypoint; routes for `/`, `/blog`, `/blog/{slug}`, `/sitemap.xml`, `/robots.txt`
- `app/templates/` — Jinja2 templates (index, blog index, blog post)
- `app/blog/*.md` — Blog posts as Markdown with YAML frontmatter
- `app/static/` — CSS, images, sitemap, robots.txt

## Blog posts

Markdown files in `app/blog/` with frontmatter:
```yaml
---
title: Post Title
date: DD-MM-YYYY or YYYY-MM-DD
description: Short summary
tags: ["tag1", "tag2"]
---
```
New `.md` file = new blog post (no DB). Slug is the filename without extension.

## Important quirks

- **Must run from repo root** — paths like `app/templates/`, `app/blog/` are relative to CWD
- **No database** — posts parsed from files on every request (cached per request only)
- **Date sorting** uses string comparison (reverse alphabetical) — inconsistent date formats will break sort order
- **No tests, no linter/formatter config** exists in this repo
- **Analytics**: Plausible self-hosted at `plausible.ananyapathak.xyz`
