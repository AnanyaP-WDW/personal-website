---
name: blog-post
description: Write or edit blog posts in this personal website repo. Load when the user says "new blog post", "add a post", "write a blog", or edits a file under app/blog/.
---

# Blog posts

Markdown files live in `app/blog/`. Each file = one post. Slug is the filename without `.md`.

## Frontmatter format

```yaml
---
title: My Post Title
date: DD-MM-YYYY
description: A short summary shown on the blog index.
tags: ["tag1", "tag2"]
---
```

Use `DD-MM-YYYY` format (not `YYYY-MM-DD`) — some posts use it, and the blog index sorts posts by reverse alphabetical date string. Mixed formats break sort order.

## Creating a new post

1. Create `app/blog/<slug>.md` with frontmatter + markdown body
2. Update `app/static/sitemap.xml` to include the new URL
3. Run `uvicorn app.main:app --reload --port 8002` from repo root to verify

## Conventions

- Keep filenames short, hyphen-separated, all lowercase
- The sitemap is hand-maintained — don't forget to update it
- No database — posts are parsed from file on every request
- Templates use Jinja2 with `{{ post.content | safe }}` for the HTML body
- MathJax is loaded in post template for LaTeX (`$...$` inline, `$$...$$` block)
- Code blocks use fenced format with language tag for syntax highlighting
