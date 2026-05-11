from src.generation.service import GenerationService, OllamaRequestError


def test_phi3_failure_returns_graceful_error() -> None:
    svc = GenerationService(
        primary_model=lambda _: (_ for _ in ()).throw(OllamaRequestError("fail")),
    )

    result = svc.generate("prompt")

    assert result.model_used == "phi3"
    assert result.reasoning_steps == []
    assert "phi3 could not be reached" in result.raw_text


def test_successful_phi3_response_parses_final_answer() -> None:
    svc = GenerationService(
        primary_model=lambda _: (
            "Reasoning Steps:\n"
            "1. Uses retrieved evidence.\n"
            "2. Gives disease-specific interpretation.\n"
            "3. Notes differential causes.\n"
            "4. Keeps a safety boundary.\n"
            "5. States evidence limits.\n"
            "Final Answer: Chest pain may arise from cardiac, respiratory, or gastrointestinal causes."
        ),
    )

    result = svc.generate("prompt")

    assert result.model_used == "phi3"
    assert len(result.reasoning_steps) == 5
    assert result.raw_text.startswith("Chest pain may arise")


def test_plain_phi3_response_is_used_as_final_answer() -> None:
    svc = GenerationService(
        primary_model=lambda _: (
            "Chest pain may arise from cardiac causes such as angina, respiratory "
            "conditions such as pneumonia, or gastrointestinal causes such as acid reflux."
        ),
    )

    result = svc.generate("prompt")

    assert result.model_used == "phi3"
    assert result.raw_text.startswith("Chest pain may arise")
    assert len(result.reasoning_steps) >= 1
    assert "invalid response" not in result.raw_text


def test_empty_phi3_response_is_treated_as_failure() -> None:
    svc = GenerationService(primary_model=lambda _: "")

    result = svc.generate("prompt")

    assert result.model_used == "phi3"
    assert "returned an empty response" in result.raw_text
    assert "could not be reached" not in result.raw_text
    assert "invalid response" not in result.raw_text
