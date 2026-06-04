import fitz
import re
import logging

logger = logging.getLogger(__name__)

_SKIP_HEADINGS = frozenset({
    "references", "bibliography", "acknowledgements", "acknowledgments",
    "appendix", "index", "about the author", "about the authors",
    "notes", "footnotes", "works cited",
})


class PDFExtractor:
    def extract(self, pdf_path: str) -> list[dict]:
        pages = []
        skipped = 0
        current_heading: str | None = None
        in_skip_section = False

        with fitz.open(pdf_path) as doc:
            for page in doc:
                page_heading = self._detect_heading(page)
                if page_heading:
                    current_heading = page_heading
                    if self._is_skip_heading(page_heading):
                        in_skip_section = True
                        logger.info(f"Skipping section: {page_heading!r}")

                if in_skip_section:
                    skipped += 1
                    continue

                text = self._clean(page.get_text())
                word_count = len(text.split())
                if word_count < 20:
                    skipped += 1
                    continue

                pages.append({
                    "page_number": page.number + 1,
                    "text": text,
                    "word_count": word_count,
                    "heading": current_heading,
                })

            logger.info(f"Extracted {len(pages)} pages, skipped {skipped}")
        return pages

    def _detect_heading(self, page) -> str | None:
        """Return the first heading on a page by detecting larger or bold font spans."""
        try:
            data = page.get_text("dict")
        except Exception:
            return None

        blocks = data.get("blocks", [])

        # Collect all font sizes to determine body (median) size
        all_sizes = [
            span["size"]
            for block in blocks if block.get("type") == 0
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if span.get("text", "").strip()
        ]
        if not all_sizes:
            return None

        body_size = sorted(all_sizes)[len(all_sizes) // 2]
        threshold = body_size * 1.1

        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                line_text = "".join(s.get("text", "") for s in spans).strip()
                if not line_text or len(line_text.split()) > 12:
                    continue
                max_size = max((s.get("size", 0) for s in spans), default=0)
                is_bold = any(s.get("flags", 0) & 16 for s in spans)
                if max_size >= threshold or is_bold:
                    return line_text

        return None

    @staticmethod
    def _is_skip_heading(heading: str) -> bool:
        normalized = heading.lower().strip().rstrip(".")
        return any(skip in normalized for skip in _SKIP_HEADINGS)

    def _clean(self, text: str) -> str:
        lines = text.splitlines()
        if len(lines) > 2:
            if len(lines[0]) < 60 and len(lines[0].split()) < 5:
                logger.debug(f"Stripped header line: {lines[0]!r}")
                lines = lines[1:]
            if lines and len(lines[-1]) < 60 and len(lines[-1].split()) < 5:
                logger.debug(f"Stripped footer line: {lines[-1]!r}")
                lines = lines[:-1]
        text = "\n".join(lines)
        text = re.sub(r"-\n(\w)", r"\1", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = self._strip_citation_lines(text)
        text = self._strip_short_blocks(text)
        return text.strip()

    def _strip_citation_lines(self, text: str) -> str:
        # Remove lines that look like bibliography entries:
        # [1] Author et al., or (Author, 2017) style lines
        lines = text.splitlines()
        kept = []
        for line in lines:
            stripped = line.strip()
            is_numbered_ref = bool(re.match(r"^\[\d+\]", stripped))
            is_author_year = bool(re.match(r"^[A-Z][a-zA-Z\s,\.\-]+\(\d{4}\)", stripped))
            if is_numbered_ref or is_author_year:
                logger.debug(f"Stripped citation line: {stripped[:60]!r}")
            else:
                kept.append(line)
        return "\n".join(kept)

    def _strip_short_blocks(self, text: str) -> str:
        blocks = text.split("\n\n")
        kept = []
        for block in blocks:
            word_count = len(block.split())
            if word_count < 25:
                logger.debug(f"Stripped short block ({word_count} words): {block[:60]!r}")
            else:
                kept.append(block)
        return "\n\n".join(kept)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    extractor = PDFExtractor()
    pages = extractor.extract(sys.argv[1])
    for p in pages[:3]:
        print(f"Page {p['page_number']} | Heading: {p['heading']!r}")
        print(p["text"][:200])
        print()
    print(f"Total pages: {len(pages)}")
