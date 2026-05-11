from __future__ import annotations

import json
import logging
import re
import time
from urllib import error, request

from src.models import GenerationResult

LOGGER = logging.getLogger(__name__)
OLLAMA_GENERATE_PATH = "/api/generate"


class OllamaRequestError(RuntimeError):
    pass


class OllamaEmptyResponseError(RuntimeError):
    pass


def parse_reasoning_steps(text: str) -> list[str]:
    found = re.findall(r"^\s*\d+\.\s*(.+)$", text, flags=re.MULTILINE)
    return found[:5]


def fallback_reasoning_steps(answer: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    useful = [s.strip() for s in sentences if len(s.split()) >= 4]
    if useful:
        return useful[:5]
    return [
        "The response was generated from the retrieved context.",
        "The final answer should be interpreted with the cited evidence.",
        "Clinician assessment is appropriate if symptoms are severe or persistent.",
        "Medication or dosage decisions require a qualified clinician.",
        "The answer may be limited by the retrieved evidence.",
    ]


def parse_final_answer(text: str) -> str:
    match = re.search(r"Final Answer:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class GenerationService:
    def __init__(
        self,
        primary_model=None,
        ollama_base_url: str = "http://127.0.0.1:11434",
        primary_model_name: str = "phi3",
        configured_models: list[str] | None = None,
        request_timeout_s: int = 120,
    ) -> None:
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.primary_model_name = primary_model_name
        self.fallback_model_name = None
        self.configured_models = configured_models or ["phi3", "medgemma"]
        self.request_timeout_s = request_timeout_s
        self.primary_model = primary_model or self._build_ollama_model(self.primary_model_name)
        self.log_active_model()

    def log_active_model(self) -> None:
        LOGGER.warning("Generation model active: %s", self.primary_model_name)

    @property
    def generate_endpoint(self) -> str:
        return f"{self.ollama_base_url}{OLLAMA_GENERATE_PATH}"

    def _build_ollama_model(self, model_name: str):
        def _invoke(prompt: str) -> str:
            payload_obj = {"model": model_name, "prompt": prompt, "stream": False}
            payload = json.dumps(payload_obj).encode("utf-8")
            LOGGER.warning("Ollama endpoint: %s", self.generate_endpoint)
            LOGGER.warning("Ollama model: %s", model_name)
            req = request.Request(
                url=self.generate_endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=self.request_timeout_s) as resp:
                    LOGGER.warning("Ollama HTTP status: %s", resp.status)
                    body = resp.read().decode("utf-8")
            except error.HTTPError as exc:
                raw_error = exc.read().decode("utf-8", errors="replace")
                LOGGER.warning("Ollama HTTP status: %s", exc.code)
                LOGGER.warning("Ollama raw response: %s", raw_error)
                raise OllamaRequestError(f"Ollama HTTP request failed for {model_name}: {exc.code}") from exc
            except error.URLError as exc:
                LOGGER.warning("Ollama HTTP status: request_failed")
                LOGGER.warning("Ollama raw response: %s", exc)
                raise OllamaRequestError(f"Ollama request failed for {model_name}: {exc}") from exc
            LOGGER.warning("Ollama raw response: %s", body)
            try:
                parsed = json.loads(body)
                generated = parsed.get("response", "").strip()
            except json.JSONDecodeError:
                generated = body.strip()
            if not generated:
                raise OllamaEmptyResponseError(f"Ollama returned empty response for {model_name}")
            return generated

        return _invoke

    def health(self) -> dict:
        tags_req = request.Request(url=f"{self.ollama_base_url}/api/tags", method="GET")
        try:
            with request.urlopen(tags_req, timeout=10) as resp:
                body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            models = [m.get("name", "") for m in parsed.get("models", [])]
            return {
                "generationBackend": "ollama",
                "activeModel": self.primary_model_name,
                "fallbackModel": None,
                "configuredModels": self.configured_models,
                "ollamaReachable": True,
                "availableModels": self.configured_models,
                "installedModels": models,
                "primaryAvailable": any(name.startswith(self.primary_model_name) for name in models),
            }
        except Exception as exc:
            return {
                "generationBackend": "ollama",
                "activeModel": self.primary_model_name,
                "fallbackModel": None,
                "configuredModels": self.configured_models,
                "ollamaReachable": False,
                "error": str(exc),
                "availableModels": self.configured_models,
                "installedModels": [],
                "primaryAvailable": False,
            }

    def generate(self, prompt: str) -> GenerationResult:
        start = time.time()
        used = self.primary_model_name
        try:
            raw = self.primary_model(prompt)
            if not raw.strip():
                raise OllamaEmptyResponseError(f"Ollama returned empty response for {self.primary_model_name}")
        except OllamaRequestError:
            LOGGER.exception("%s generation failed", self.primary_model_name)
            message = (
                f"Generation is temporarily unavailable because {self.primary_model_name} could not be reached. "
                f"Please confirm Ollama is running and the {self.primary_model_name} model is installed."
            )
            return GenerationResult(
                raw_text=message,
                reasoning_steps=[],
                model_used=used,
                tokens_generated=len(message.split()),
                latency_ms=int((time.time() - start) * 1000),
            )
        except OllamaEmptyResponseError:
            LOGGER.exception("%s returned an empty response", self.primary_model_name)
            message = (
                f"Generation is temporarily unavailable because {self.primary_model_name} returned an empty response. "
                "Please retry the question."
            )
            return GenerationResult(
                raw_text=message,
                reasoning_steps=[],
                model_used=used,
                tokens_generated=len(message.split()),
                latency_ms=int((time.time() - start) * 1000),
            )
        except Exception as exc:
            LOGGER.exception("%s generation failed unexpectedly", self.primary_model_name)
            message = f"Generation is temporarily unavailable: {exc}"
            return GenerationResult(
                raw_text=message,
                reasoning_steps=[],
                model_used=used,
                tokens_generated=len(message.split()),
                latency_ms=int((time.time() - start) * 1000),
            )
        LOGGER.warning("Raw model output before parsing: %s", raw)
        steps = parse_reasoning_steps(raw)
        final_answer = parse_final_answer(raw)
        if not steps:
            steps = fallback_reasoning_steps(final_answer)
        return GenerationResult(
            raw_text=final_answer,
            reasoning_steps=steps,
            model_used=used,
            tokens_generated=len(final_answer.split()),
            latency_ms=int((time.time() - start) * 1000),
        )
