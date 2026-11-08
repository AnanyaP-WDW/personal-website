"""Sanitizer checks for blog markdown. Run: python3 tests/test_blog_security.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import (  # noqa: E402
    add_image_attrs,
    get_related_posts,
    needs_math,
    parse_blog_post,
)


def render(markdown_body: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write("---\ntitle: T\n---\n" + markdown_body)
        path = f.name
    try:
        return parse_blog_post(path)["content"]
    finally:
        os.unlink(path)


def test_dangerous_html_is_stripped():
    html = render('<img src=x onerror="alert(1)">\n\n<a href="javascript:alert(1)">x</a>\n\n<script>alert(1)</script>')
    assert "onerror" not in html
    assert "javascript:" not in html
    assert "<script" not in html


def test_legit_markdown_survives():
    html = render(
        "## Heading\n\n[link](https://example.com) ![alt](/static/images/profile.png)\n\n"
        "```python\nprint(1 < 2)\n```\n\nMath \\(x^2\\) and `code`.\n"
    )
    assert "<h2" in html
    assert 'href="https://example.com"' in html
    assert 'src="/static/images/profile.png"' in html
    assert "<pre" in html
    assert "\\(x^2\\)" in html


def test_local_images_get_size_and_lazy_loading():
    html = add_image_attrs('<p><img alt="x" src="/static/images/profile.png"></p>')
    assert 'width="480"' in html and 'height="480"' in html
    assert 'loading="lazy"' in html
    # remote images are left alone apart from lazy loading
    remote = add_image_attrs('<img src="https://example.com/a.png">')
    assert "width=" not in remote and "loading=" in remote


def test_math_detection():
    assert needs_math("<p>inline \\(x^2\\)</p>")
    assert not needs_math("<p>plain text</p>")


def test_related_posts_share_tags_first():
    posts = [
        {"slug": "a", "tags": ["x"], "date": "2026-01-01"},
        {"slug": "b", "tags": ["x", "y"], "date": "2025-01-01"},
        {"slug": "c", "tags": [], "date": "2026-06-01"},
    ]
    related = get_related_posts(posts[0], posts, limit=2)
    assert [p["slug"] for p in related] == ["b", "c"]


if __name__ == "__main__":
    test_dangerous_html_is_stripped()
    test_legit_markdown_survives()
    test_local_images_get_size_and_lazy_loading()
    test_math_detection()
    test_related_posts_share_tags_first()
    print("ok")
