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


def test_yandex_embeddings_request_float_string_payload() -> None:
    from types import SimpleNamespace

    from app.llm import YandexClient

    calls = []

    class FakeEmbeddings:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])

    client = object.__new__(YandexClient)
    client.settings = SimpleNamespace(
        yandex_embedding_model="emb://folder/text-search-doc/latest",
        yandex_query_embedding_model="emb://folder/text-search-query/latest",
    )
    client.client = SimpleNamespace(embeddings=FakeEmbeddings())

    assert client.embed_documents(["document"])[0] == [0.1, 0.2]
    assert client.embed_query("query") == [0.1, 0.2]
    assert calls[0]["input"] == "document"
    assert calls[0]["encoding_format"] == "float"
    assert calls[1]["input"] == "query"
    assert calls[1]["encoding_format"] == "float"


def test_haystack_text_content_is_not_rendered_as_object_repr() -> None:
    from types import SimpleNamespace

    from app.assistant import _message_text

    text_content = SimpleNamespace(text="Привет! Я ассистент.")
    message = SimpleNamespace(content=[text_content])
    assert _message_text(message) == "Привет! Я ассистент."


def test_vlm_request_disables_reasoning_and_reads_text_block() -> None:
    from types import SimpleNamespace

    from app.llm import YandexClient

    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=[SimpleNamespace(text="Описание собаки.")])
                    )
                ]
            )

    client = object.__new__(YandexClient)
    client.settings = SimpleNamespace(
        yandex_vision_model="gpt://folder/vision",
        yandex_vision_reasoning_effort="none",
    )
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    assert client.describe_image(b"image") == "Описание собаки."
    assert calls[0]["reasoning_effort"] == "none"
    assert calls[0]["max_tokens"] == 700
