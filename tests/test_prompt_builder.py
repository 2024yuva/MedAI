from src.models import Chunk, RetrievedContext
from src.prompt.builder import PromptBuilder


def test_prompt_builder_limits_context_chars() -> None:
    builder = PromptBuilder()
    contexts = [
        RetrievedContext(
            chunk=Chunk(id=str(i), text=("x" * 500), source_file=f"s{i}.pdf", page_number=i),
            similarity_score=0.9 - (i * 0.1),
            rank=i,
        )
        for i in range(1, 5)
    ]
    prompt = builder.build_cot_prompt("Q?", contexts, low_confidence=False)
    context_block = prompt.split("Context:\n", 1)[1]
    extracted = [line for line in context_block.splitlines() if line and not line.startswith("[")]
    assert all(len(line) <= 300 for line in extracted)
    assert sum(len(line) for line in extracted) <= 900
