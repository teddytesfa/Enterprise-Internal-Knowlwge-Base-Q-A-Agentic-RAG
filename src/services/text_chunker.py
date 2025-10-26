from typing import List

class TextChunker:
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split_words(self, text: str) -> List[str]:
        return text.split()

    def chunk_text(self, text: str) -> List[str]:
        words = self._split_words(text)
        if not words:
            return []
        chunks = []
        step = self.chunk_size - self.overlap
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            if chunk_words:
                chunks.append(" ".join(chunk_words))
            if i + self.chunk_size >= len(words):
                break
        return chunks
