from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import markdown
import os
import re
from datetime import datetime
from typing import List, Dict
import glob

app = FastAPI()

# Configure templates
templates = Jinja2Templates(directory="app/templates")

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")


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
        markdown_content, extensions=["codehilite", "fenced_code"]
    )

    # Get filename without extension for slug
    filename = os.path.basename(file_path)
    slug = os.path.splitext(filename)[0]

    return {
        "slug": slug,
        "title": metadata.get("title", slug.replace("-", " ").title()),
        "date": metadata.get("date", ""),
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
            print(f"Error parsing {file_path}: {e}")
            continue

    # Sort by date (newest first)
    posts.sort(key=lambda x: x.get("date", ""), reverse=True)
    return posts


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/sitemap.xml")
async def sitemap():
    return FileResponse("app/static/sitemap.xml")


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
        "blog/post.html", {"request": request, "post": post}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
