"""Sanitizer checks for blog markdown. Run: python3 tests/test_blog_security.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import parse_blog_post  # noqa: E402


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


if __name__ == "__main__":
    test_dangerous_html_is_stripped()
    test_legit_markdown_survives()
    print("ok")
