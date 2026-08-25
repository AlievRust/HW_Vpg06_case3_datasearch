from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_is_not_imported_by_new_code() -> None:
    source_files = list((ROOT / "app").rglob("*.py"))
    content = "\n".join(path.read_text(encoding="utf-8") for path in source_files)
    assert "from legacy" not in content
    assert "import legacy" not in content


def test_mvp_files_are_configured() -> None:
    from app.config import Settings

    extensions = Settings.__dataclass_fields__["supported_extensions"].default_factory()
    assert extensions == {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}


def test_document_summary_is_one_sentence() -> None:
    from app.documents import DocumentService

    service = object.__new__(DocumentService)
    service.llm = type(
        "FakeLLM",
        (),
        {"generate": lambda self, prompt, **kwargs: "Первое предложение. Второе предложение."},
    )()
    assert service.summarize("file.txt", "content") == "Первое предложение."
