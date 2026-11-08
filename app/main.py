from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
import markdown
import nh3
import logging
import os
import re
import struct
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict
import glob

# Docs/schema are not needed publicly.
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

CSP = (
    "default-src 'self'; "
    "script-src 'self' https://cdn.jsdelivr.net https://plausible.ananyapathak.xyz; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "font-src 'self' https://cdn.jsdelivr.net; "
    "connect-src 'self' https://plausible.ananyapathak.xyz; "
    "object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = CSP
    return response


# Configure templates
templates = Jinja2Templates(directory="app/templates")

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")


def image_dimensions(src: str):
    """Width/height for a local PNG, so images reserve space before loading."""
    if not src.startswith("/static/"):
        return None
    try:
        with open(os.path.join("app", src.lstrip("/")), "rb") as f:
            head = f.read(24)
    except OSError:
        return None
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", head[16:24])


IMG_TAG = re.compile(r"<img\b[^>]*>")


def add_image_attrs(html_content: str) -> str:
    """Lazy-load local images and give them intrinsic size (avoids layout shift)."""

    def repl(match: re.Match) -> str:
        tag = match.group(0).rstrip("/>").rstrip()
        if "loading=" in tag:
            return tag + ">"
        attrs = ' loading="lazy" decoding="async"'
        src = re.search(r'src="([^"]*)"', tag)
        size = image_dimensions(src.group(1)) if src else None
        if size:
            attrs += f' width="{size[0]}" height="{size[1]}"'
        return tag + attrs + ">"

    return IMG_TAG.sub(repl, html_content)


def needs_math(html_content: str) -> bool:
    """Whether a post actually uses LaTeX, so MathJax is only loaded when needed."""
    return bool(re.search(r"\\\(|\\\[|\$\$", html_content))


def get_related_posts(post: Dict, posts: List[Dict], limit: int = 3) -> List[Dict]:
    """Other posts ranked by shared tags, then recency."""
    tags = set(post["tags"])
    others = [p for p in posts if p["slug"] != post["slug"]]
    others.sort(key=lambda p: (len(tags & set(p["tags"])), p["date"]), reverse=True)
    return others[:limit]


def normalize_date(value: str) -> str:
    """Return ISO (YYYY-MM-DD) date or empty string if unparseable."""
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d-%m-%y"):
        try:
            return datetime.strptime(value.strip(), fmt).date().isoformat()
        except ValueError:
            pass
    return ""


def parse_blog_post(file_path: str) -> Dict:
    """Parse a markdown blog post and extract metadata and content."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split frontmatter and content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1].strip()
            markdown_content = parts[2].strip()
        else:
            frontmatter = ""
            markdown_content = content
    else:
        frontmatter = ""
        markdown_content = content

    # Parse frontmatter
    metadata = {}
    for line in frontmatter.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip("\"'")

            # Handle tags as list
            if key == "tags":
                # Parse list format ["tag1", "tag2"]
                if value.startswith("[") and value.endswith("]"):
                    tags_str = value[1:-1]  # Remove brackets
                    metadata[key] = [
                        tag.strip().strip("\"'")
                        for tag in tags_str.split(",")
                        if tag.strip()
                    ]
                else:
                    metadata[key] = [value]
            else:
                metadata[key] = value

    # Convert markdown to HTML
    html_content = markdown.markdown(
        markdown_content,
        extensions=["codehilite", "fenced_code", "toc", "tables", "pymdownx.arithmatex"],
        extension_configs={
            "pymdownx.arithmatex": {
                "generic": True,
            }
        },
    )
    # Markdown passes raw HTML through, so sanitize before rendering with |safe.
    html_content = nh3.clean(
        html_content,
        tags={
            "a", "p", "br", "hr", "em", "strong", "blockquote", "code", "pre",
            "span", "div", "img", "ul", "ol", "li", "h1", "h2", "h3", "h4",
            "h5", "h6", "sup", "sub", "del", "ins", "table", "thead", "tbody",
            "tr", "th", "td",
        },
        attributes={
            "*": {"class", "id"},
            "a": {"href", "title"},
            "img": {"src", "alt", "title"},
        },
        url_schemes={"http", "https", "mailto"},
        link_rel="noopener noreferrer",
    )
    html_content = add_image_attrs(html_content)

    # Get filename without extension for slug
    filename = os.path.basename(file_path)
    slug = os.path.splitext(filename)[0]

    date_iso = normalize_date(metadata.get("date", ""))

    return {
        "slug": slug,
        "title": metadata.get("title", slug.replace("-", " ").title()),
        "date": date_iso or metadata.get("date", ""),
        "date_iso": date_iso,
        "description": metadata.get("description", ""),
        "tags": metadata.get("tags", []),
        "content": html_content,
        "metadata": metadata,
    }


def get_all_blog_posts() -> List[Dict]:
    """Get all blog posts sorted by date (newest first)."""
    blog_dir = "app/blog"
    posts = []

    if not os.path.exists(blog_dir):
        return posts

    markdown_files = glob.glob(os.path.join(blog_dir, "*.md"))

    for file_path in markdown_files:
        try:
            post = parse_blog_post(file_path)
            posts.append(post)
        except Exception as e:
            logging.warning("Error parsing %s: %s", file_path, e)
            continue

    # Sort by date (newest first)
    posts.sort(key=lambda x: x.get("date", ""), reverse=True)
    return posts


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/consulting")
async def consulting(request: Request):
    return templates.TemplateResponse("consulting.html", {"request": request})


@app.get("/feed.xml")
async def feed():
    posts = get_all_blog_posts()[:20]
    root = ET.Element("rss", {"version": "2.0"})
    channel = ET.SubElement(root, "channel")
    ET.SubElement(channel, "title").text = "Ananya Pathak — ML Engineering Blog"
    ET.SubElement(channel, "link").text = "https://ananyapathak.xyz/blog"
    ET.SubElement(channel, "description").text = (
        "Notes on ML engineering, protein language models, and production data pipelines."
    )

    for post in posts:
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = post["title"]
        ET.SubElement(item, "link").text = f"https://ananyapathak.xyz/blog/{post['slug']}"
        ET.SubElement(item, "guid").text = f"https://ananyapathak.xyz/blog/{post['slug']}"
        if post["description"]:
            ET.SubElement(item, "description").text = post["description"]
        if post["date_iso"]:
            ET.SubElement(item, "pubDate").text = datetime.fromisoformat(
                post["date_iso"]
            ).strftime("%a, %d %b %Y 00:00:00 +0000")

    return Response(
        content=ET.tostring(root, encoding="unicode", xml_declaration=True),
        media_type="application/rss+xml",
    )


@app.get("/sitemap.xml")
async def sitemap():
    root = ET.Element("urlset", {"xmlns": "http://www.sitemaps.org/schemas/sitemap/0.9"})

    def add(loc, lastmod=None, changefreq=None, priority=None):
        url = ET.SubElement(root, "url")
        ET.SubElement(url, "loc").text = f"https://ananyapathak.xyz{loc}"
        if lastmod:
            ET.SubElement(url, "lastmod").text = lastmod
        if changefreq:
            ET.SubElement(url, "changefreq").text = changefreq
        if priority:
            ET.SubElement(url, "priority").text = priority

    add("/", changefreq="monthly", priority="1.0")
    add("/consulting", changefreq="monthly", priority="0.9")
    add("/blog", changefreq="weekly", priority="0.8")
    add("/feed.xml", changefreq="weekly", priority="0.3")
    for post in get_all_blog_posts():
        add(
            f"/blog/{post['slug']}",
            lastmod=post["date_iso"] or None,
            changefreq="monthly",
            priority="0.6",
        )

    return Response(
        content=ET.tostring(root, encoding="unicode", xml_declaration=True),
        media_type="application/xml",
    )


@app.get("/robots.txt")
async def robots():
    return FileResponse("app/static/robots.txt")


@app.get("/blog")
async def blog_index(request: Request):
    """Display list of all blog posts."""
    posts = get_all_blog_posts()
    return templates.TemplateResponse(
        "blog/index.html", {"request": request, "posts": posts}
    )


@app.get("/blog/{slug}")
async def blog_post(request: Request, slug: str):
    """Display individual blog post."""
    posts = get_all_blog_posts()

    # Find the post with matching slug
    post = None
    for p in posts:
        if p["slug"] == slug:
            post = p
            break

    if not post:
        raise HTTPException(status_code=404, detail="Blog post not found")

    return templates.TemplateResponse(
        "blog/post.html",
        {
            "request": request,
            "post": post,
            "related": get_related_posts(post, posts),
            "math": needs_math(post["content"]),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
