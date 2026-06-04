from unittest.mock import MagicMock, patch
from pdf_extractor import PDFExtractor


def _make_page(text, number=0, heading=None):
    page = MagicMock()
    page.number = number

    def get_text(fmt=None):
        if fmt == "dict":
            # Use bold flag (16) for heading detection — more reliable in tests than
            # font size, which requires enough body spans to compute a correct median.
            spans_body = [{"text": text, "size": 12.0, "flags": 0}]
            blocks = [{"type": 0, "lines": [{"spans": spans_body}]}]
            if heading:
                spans_head = [{"text": heading, "size": 12.0, "flags": 16}]  # bold
                blocks.insert(0, {"type": 0, "lines": [{"spans": spans_head}]})
            return {"blocks": blocks}
        return text

    page.get_text.side_effect = get_text
    return page


def _mock_open(pages):
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=pages)
    cm.__exit__ = MagicMock(return_value=False)
    return patch("pdf_extractor.fitz.open", return_value=cm)


def test_extract_returns_list_of_dicts():
    long_text = "word " * 30
    with _mock_open([_make_page(long_text)]):
        result = PDFExtractor().extract("dummy.pdf")
    assert isinstance(result, list)
    assert len(result) == 1
    assert "page_number" in result[0]
    assert "text" in result[0]
    assert "word_count" in result[0]
    assert "heading" in result[0]


def test_extract_skips_pages_under_20_words():
    with _mock_open([_make_page("too short")]):
        result = PDFExtractor().extract("dummy.pdf")
    assert result == []


def test_extract_page_number_is_one_based():
    long_text = "word " * 30
    with _mock_open([_make_page(long_text, number=0)]):
        result = PDFExtractor().extract("dummy.pdf")
    assert result[0]["page_number"] == 1


def test_extract_strips_hyphenation():
    text = "word " * 15 + "con-\nnection " + "word " * 15
    with _mock_open([_make_page(text)]):
        result = PDFExtractor().extract("dummy.pdf")
    assert "con-\nnection" not in result[0]["text"]
    assert "connection" in result[0]["text"]


def test_extract_strips_short_first_and_last_lines():
    lines = ["Page 1"] + ["real content word " * 5] * 5 + ["footer"]
    text = "\n".join(lines)
    with _mock_open([_make_page(text)]):
        result = PDFExtractor().extract("dummy.pdf")
    if result:
        assert "Page 1" not in result[0]["text"]
        assert "footer" not in result[0]["text"]


def test_extract_collapses_excessive_newlines():
    text = "word " * 30 + "\n\n\n\n" + "word " * 30
    with _mock_open([_make_page(text)]):
        result = PDFExtractor().extract("dummy.pdf")
    assert "\n\n\n" not in result[0]["text"]


def test_extract_word_count_matches_text():
    long_text = "hello " * 25
    with _mock_open([_make_page(long_text)]):
        result = PDFExtractor().extract("dummy.pdf")
    assert result[0]["word_count"] == len(result[0]["text"].split())


def test_extract_strips_short_isolated_blocks():
    content_a = "word " * 30
    caption = "Figure 1: The model architecture diagram shown above."
    content_b = "word " * 30
    text = content_a.strip() + "\n\n" + caption + "\n\n" + content_b.strip()
    with _mock_open([_make_page(text)]):
        result = PDFExtractor().extract("dummy.pdf")
    assert result
    assert "Figure 1" not in result[0]["text"]
    assert "word" in result[0]["text"]


def test_extract_keeps_long_blocks():
    content = "word " * 30
    with _mock_open([_make_page(content)]):
        result = PDFExtractor().extract("dummy.pdf")
    assert result
    assert len(result[0]["text"].split()) >= 25


def test_extract_detects_heading_from_large_font():
    text = "word " * 30
    with _mock_open([_make_page(text, heading="3. Model Architecture")]):
        result = PDFExtractor().extract("dummy.pdf")
    assert result[0]["heading"] == "3. Model Architecture"


def test_extract_heading_is_none_when_no_large_font():
    text = "word " * 30
    with _mock_open([_make_page(text)]):
        result = PDFExtractor().extract("dummy.pdf")
    assert result[0]["heading"] is None


def test_extract_skips_references_section():
    content = "word " * 30
    ref_page = _make_page(content, heading="References")
    with _mock_open([ref_page]):
        result = PDFExtractor().extract("dummy.pdf")
    assert result == []


def test_extract_skips_acknowledgements_section():
    content = "word " * 30
    ack_page = _make_page(content, heading="Acknowledgements")
    with _mock_open([ack_page]):
        result = PDFExtractor().extract("dummy.pdf")
    assert result == []


def test_extract_strips_numbered_citation_lines():
    # [1] Author et al. style lines should be removed
    content = "word " * 30 + "\n[1] Vaswani et al., Attention Is All You Need, 2017.\n" + "word " * 10
    with _mock_open([_make_page(content)]):
        result = PDFExtractor().extract("dummy.pdf")
    if result:
        assert "[1]" not in result[0]["text"]


def test_extract_strips_author_year_citation_lines():
    # Author (Year) style lines should be removed
    content = "word " * 30 + "\nBahdanau, Cho and Bengio (2015) introduced attention.\n" + "word " * 10
    with _mock_open([_make_page(content)]):
        result = PDFExtractor().extract("dummy.pdf")
    if result:
        assert "Bahdanau" not in result[0]["text"]


def test_extract_does_not_skip_regular_sections():
    content = "word " * 30
    page = _make_page(content, heading="3. Methodology")
    with _mock_open([page]):
        result = PDFExtractor().extract("dummy.pdf")
    assert len(result) == 1
