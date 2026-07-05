"""
Text chunking module.
Splits extracted text into semantically meaningful chunks for embedding.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS


def chunk_text(text: str, max_chars: int = None, overlap: int = None) -> list[str]:
    """
    Split text into chunks suitable for embedding.

    Uses a line-aware strategy: accumulates lines into chunks up to max_chars,
    then starts a new chunk with overlap from the previous chunk's tail.
    Lines longer than max_chars are force-split so no chunk exceeds the limit.

    Args:
        text: Full text to chunk.
        max_chars: Maximum characters per chunk (default from config).
        overlap: Number of characters to overlap between chunks (default from config).

    Returns:
        List of text chunks.
    """
    if max_chars is None:
        max_chars = CHUNK_MAX_CHARS
    if overlap is None:
        overlap = CHUNK_OVERLAP_CHARS

    lines = text.split("\n")
    chunks = []
    current_chunk = ""

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        # FIX #19: Force-split overlong lines so no chunk exceeds max_chars
        segments = _split_long_line(line, max_chars) if len(line) >= max_chars else [line]

        for segment in segments:
            if len(current_chunk) + len(segment) + 1 < max_chars:
                current_chunk += segment + "\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                # Start new chunk with overlap from tail of previous
                if overlap > 0 and current_chunk:
                    # Clamp overlap so overlap + segment doesn't exceed max_chars
                    available = max(0, max_chars - len(segment) - 2)  # -2 for newlines
                    actual_overlap = min(overlap, available)
                    if actual_overlap > 0:
                        tail = current_chunk.strip()[-actual_overlap:]
                        current_chunk = tail + "\n" + segment + "\n"
                    else:
                        current_chunk = segment + "\n"
                else:
                    current_chunk = segment + "\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _split_long_line(line: str, max_chars: int) -> list[str]:
    """
    Force-split a single line that exceeds max_chars into smaller segments.
    Tries to split on word boundaries when possible.
    """
    segments = []
    while len(line) > max_chars:
        # Try to find a space near the boundary to split on
        split_pos = line.rfind(" ", 0, max_chars)
        if split_pos <= 0:
            split_pos = max_chars  # No space found, hard-split
        segments.append(line[:split_pos].strip())
        line = line[split_pos:].strip()
    if line:
        segments.append(line)
    return segments
