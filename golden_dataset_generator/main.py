import sys
import os
import logging
from dotenv import load_dotenv
from config import LOG_LEVEL, LLM_MODEL
from pdf_extractor import PDFExtractor
from section_splitter import SectionSplitter
from question_generator import QuestionGenerator
from dataset_manager import DatasetManager
from multi_context_generator import MultiContextGenerator

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)


def run() -> None:
    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    if not pdf_path.lower().endswith(".pdf"):
        print("Error: File must have a .pdf extension.")
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set.")
        print("  1. Copy .env.example to .env")
        print("  2. Add your key: OPENAI_API_KEY=sk-...")
        sys.exit(1)

    pages = PDFExtractor().extract(pdf_path)
    sections = SectionSplitter().split(pages)

    generator = QuestionGenerator()
    all_questions: list[dict] = []
    for i, section in enumerate(sections, 1):
        print(f"Generating questions for section {i} of {len(sections)}...")
        all_questions.extend(generator.generate(section))

    mc_generator = MultiContextGenerator()
    mc_questions: list[dict] = []
    for i in range(len(sections) - 1):
        print(f"Generating multi-context questions for sections {sections[i]['section_id']} + {sections[i+1]['section_id']}...")
        mc_questions.extend(mc_generator.generate_for_pair(sections[i], sections[i + 1]))

    all_questions.extend(mc_questions)

    if not all_questions:
        print("Warning: No questions were generated across all sections.")

    out_path = DatasetManager().save(all_questions, pdf_path, len(sections), LLM_MODEL)

    by_type: dict[str, int] = {}
    for q in all_questions:
        t = q.get("question_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    print(f"\nDone! {len(all_questions)} questions generated.")
    print(f"By type: {by_type}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    run()
