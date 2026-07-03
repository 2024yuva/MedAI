from __future__ import annotations

from typing import List

from src.models import RetrievedContext


class PromptBuilder:
    MAX_CHUNK_CHARS = 300
    MAX_TOTAL_CONTEXT_CHARS = 900

    def _build_limited_context(self, context: List[RetrievedContext]) -> str:
        parts: list[str] = []
        used = 0
        for i, c in enumerate(context, start=1):
            if used >= self.MAX_TOTAL_CONTEXT_CHARS:
                break
            text = (c.chunk.text or "").strip()[: self.MAX_CHUNK_CHARS]
            remaining = self.MAX_TOTAL_CONTEXT_CHARS - used
            if remaining <= 0:
                break
            text = text[:remaining]
            if not text:
                continue
            used += len(text)
            parts.append(f"[{i}] Source={c.chunk.source_file} Page={c.chunk.page_number}\n{text}")
        return "\n\n".join(parts)

    def build_cot_prompt(self, question: str, context: List[RetrievedContext], low_confidence: bool = False) -> str:
        ctx = self._build_limited_context(context)
        insuff = "\nIf the context does not contain enough information, say so clearly." if low_confidence else ""
        return f"""System:
You are a helpful medical assistant. Answer in plain, simple language that anyone can understand.
Avoid medical jargon. If you must use a medical term, explain it in simple words right after.
Keep answers concise and focused. Use short paragraphs.
Only use information from the provided context.
Do not prescribe or recommend specific medications or dosages.
Always remind the user to consult a doctor for personal medical decisions.{insuff}

User question:
{question}

Context from medical references:
{ctx}

Answer in 1-2 short sentences directly and quickly. Do not include reasoning or long explanations:
"""
