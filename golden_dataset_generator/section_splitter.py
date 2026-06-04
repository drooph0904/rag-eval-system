import logging
from config import SECTION_MAX_WORDS, SECTION_MIN_WORDS

logger = logging.getLogger(__name__)


class SectionSplitter:
    def split(self, pages: list[dict]) -> list[dict]:
        sections = []
        buffer_texts = []
        buffer_pages = []
        buffer_headings = []
        buffer_words = 0
        section_id = 1

        def flush():
            nonlocal section_id
            if not buffer_texts:
                return
            word_count = sum(len(t.split()) for t in buffer_texts)
            if word_count < SECTION_MIN_WORDS:
                return
            start, end = buffer_pages[0], buffer_pages[-1]
            source_pages = str(start) if start == end else f"{start}-{end}"
            sections.append({
                "section_id": section_id,
                "text": "\n\n".join(buffer_texts),
                "word_count": word_count,
                "source_pages": source_pages,
                "heading": buffer_headings[0] if buffer_headings else None,
            })
            section_id += 1

        for page in pages:
            pw = page["word_count"]

            if pw > SECTION_MAX_WORDS:
                flush()
                buffer_texts.clear()
                buffer_pages.clear()
                buffer_headings.clear()
                buffer_words = 0
                for chunk in page["text"].split("\n\n"):
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    cw = len(chunk.split())
                    if cw >= SECTION_MIN_WORDS:
                        sections.append({
                            "section_id": section_id,
                            "text": chunk,
                            "word_count": cw,
                            "source_pages": str(page["page_number"]),
                            "heading": page.get("heading"),
                        })
                        section_id += 1
                continue

            if buffer_words + pw > SECTION_MAX_WORDS:
                flush()
                buffer_texts.clear()
                buffer_pages.clear()
                buffer_headings.clear()
                buffer_words = 0

            buffer_texts.append(page["text"])
            buffer_pages.append(page["page_number"])
            buffer_headings.append(page.get("heading"))
            buffer_words += pw

        flush()

        avg = sum(s["word_count"] for s in sections) / max(len(sections), 1)
        logger.info(f"Created {len(sections)} sections, avg {avg:.0f} words/section")
        return sections


if __name__ == "__main__":
    import sys
    import logging as log
    from pdf_extractor import PDFExtractor
    log.basicConfig(level=log.INFO)
    pages = PDFExtractor().extract(sys.argv[1])
    sections = SectionSplitter().split(pages)
    print(f"Total sections: {len(sections)}")
    for s in sections[:3]:
        print(f"Section {s['section_id']} | Heading: {s['heading']!r} | Pages: {s['source_pages']}")
        print(s["text"][:150])
        print()
