"""Extrai texto de arquivos de vários formatos."""
from __future__ import annotations

import logging
from pathlib import Path

import chardet

logger = logging.getLogger(__name__)


def extract_text(file_path: Path, mime_type: str | None = None) -> str | None:
    """Extrai texto de um arquivo. Retorna None se não conseguir.
    
    Suporta:
    - PDF (via pypdf)
    - Imagens PNG/JPG/JPEG/WEBP (via Tesseract OCR, português + inglês)
    - Arquivos texto/código (qualquer encoding detectável)
    """
    if not file_path.exists():
        return None
    
    suffix = file_path.suffix.lower()
    
    if suffix == ".pdf" or (mime_type and "pdf" in mime_type):
        return _extract_pdf(file_path)
    
    if suffix in (".png", ".jpg", ".jpeg", ".webp"):
        return _extract_image_ocr(file_path)
    
    return _extract_text_file(file_path)


def _extract_pdf(file_path: Path) -> str | None:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        text_parts = []
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
        text = "\n\n".join(text_parts).strip()
        return text if text else None
    except Exception as e:
        logger.error("Erro extraindo PDF %s: %s", file_path, e)
        return None


def _extract_image_ocr(file_path: Path) -> str | None:
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img, lang="por+eng")
        return text.strip() or None
    except Exception as e:
        logger.error("Erro fazendo OCR %s: %s", file_path, e)
        return None


def _extract_text_file(file_path: Path) -> str | None:
    try:
        raw = file_path.read_bytes()
        if len(raw) == 0:
            return None
        
        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "utf-8"
        
        try:
            return raw.decode(encoding, errors="replace")
        except Exception:
            return raw.decode("utf-8", errors="replace")
    except Exception as e:
        logger.error("Erro lendo texto %s: %s", file_path, e)
        return None


def estimate_tokens(text: str) -> int:
    """Estimativa grosseira: ~3 chars por token em pt-BR."""
    return max(1, len(text) // 3)
