from section_splitter import SectionSplitter


def _page(text, number=1):
    return {"page_number": number, "text": text, "word_count": len(text.split())}


def test_merges_short_pages_into_one_section():
    pages = [_page("word " * 100, 1), _page("word " * 100, 2)]
    sections = SectionSplitter().split(pages)
    assert len(sections) == 1
    assert sections[0]["word_count"] == 200


def test_starts_new_section_when_max_words_exceeded():
    # 500 words on page 1; adding page 2 (100 words) would exceed 500 → new section
    pages = [_page("word " * 500, 1), _page("word " * 100, 2)]
    sections = SectionSplitter().split(pages)
    assert len(sections) == 2


def test_skips_sections_below_min_words():
    pages = [_page("word " * 30, 1)]  # 30 < SECTION_MIN_WORDS (50)
    sections = SectionSplitter().split(pages)
    assert len(sections) == 0


def test_section_ids_start_at_1_and_increment():
    # 3 * 200 = 600 words → 2 sections (200+200=400 ≤ 500, then 200 alone)
    pages = [_page("word " * 200, i) for i in range(1, 4)]
    sections = SectionSplitter().split(pages)
    for i, s in enumerate(sections, 1):
        assert s["section_id"] == i


def test_source_pages_single_page():
    pages = [_page("word " * 200, 5)]
    sections = SectionSplitter().split(pages)
    assert sections[0]["source_pages"] == "5"


def test_source_pages_range():
    # 200 + 200 = 400 ≤ 500 → merge into one section spanning pages 3-4
    pages = [_page("word " * 200, 3), _page("word " * 200, 4)]
    sections = SectionSplitter().split(pages)
    assert sections[0]["source_pages"] == "3-4"


def test_oversized_single_page_splits_at_double_newline():
    chunk_a = "alpha " * 200   # 200 words
    chunk_b = "beta " * 200    # 200 words
    text = chunk_a.strip() + "\n\n" + chunk_b.strip()
    page = {"page_number": 1, "text": text, "word_count": 550}  # force word_count above 500
    sections = SectionSplitter().split([page])
    assert len(sections) == 2


def test_section_dict_has_required_keys():
    pages = [_page("word " * 200, 1)]
    sections = SectionSplitter().split(pages)
    assert set(sections[0].keys()) == {"section_id", "text", "word_count", "source_pages", "heading"}


def test_section_inherits_heading_from_first_page():
    page = _page("word " * 200, 1)
    page["heading"] = "2. Background"
    sections = SectionSplitter().split([page])
    assert sections[0]["heading"] == "2. Background"


def test_section_heading_none_when_no_heading():
    pages = [_page("word " * 200, 1)]
    sections = SectionSplitter().split(pages)
    assert sections[0]["heading"] is None
