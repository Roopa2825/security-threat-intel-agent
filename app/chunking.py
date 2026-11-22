"""Split long text into smaller overlapping chunks (no external dependencies)."""


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into chunks of at most ~`size` characters, keeping paragraphs
    together where possible. Each new chunk starts with the last `overlap`
    characters of the previous one, so ideas at a boundary are not lost."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if current else ""
            current = f"{tail}\n\n{para}" if tail else para
    if current:
        chunks.append(current)
    return chunks
