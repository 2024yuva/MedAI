from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import logging

import pdfplumber


logger = logging.getLogger(__name__)


@dataclass
class RawDocument:
    source_file: str
    pages: List[str]


def load_pdfs(data_dir: str) -> List[RawDocument]:
    folder = Path(data_dir)
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        logger.warning("No PDF documents found in %s", data_dir)
        return []
    docs: List[RawDocument] = []
    for pdf in pdfs:
        try:
            pages: List[str] = []
            with pdfplumber.open(str(pdf)) as p:
                for page in p.pages:
                    pages.append(page.extract_text() or "")
            docs.append(RawDocument(source_file=pdf.name, pages=pages))
        except Exception as exc:
            logger.error("Skipping unreadable PDF %s: %s", pdf.name, exc)
    return docs

