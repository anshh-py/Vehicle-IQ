"""Plain-text sanitization for untrusted OCR/provider output."""

from html.parser import HTMLParser
import re


class _TextExtractor(HTMLParser):
    BLOCK_TAGS = {"br", "p", "div", "tr", "td", "th", "li", "table", "section", "article"}
    HIDDEN_TAGS = {"script", "style", "iframe", "object", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden_depth = 0

    def handle_starttag(self, tag: str, _attrs) -> None:
        if tag.casefold() in self.HIDDEN_TAGS:
            self.hidden_depth += 1
        elif not self.hidden_depth and tag.casefold() in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized in self.HIDDEN_TAGS and self.hidden_depth:
            self.hidden_depth -= 1
        elif not self.hidden_depth and normalized in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth:
            self.parts.append(data)


def sanitize_ocr_text(value: str | None) -> str:
    """Return display-safe plain text; the untouched value remains evidence."""
    if not value:
        return ""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", value)
    parser = _TextExtractor()
    try:
        parser.feed(text)
        parser.close()
        text = "".join(parser.parts)
    except Exception:
        text = re.sub(r"<[^>]*>", " ", text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)[:20_000]
