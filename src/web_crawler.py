"""
Enterprise Web Content Extractor & DOM Chunking Module.
Parses HTML structures, strips boilerplate tags (scripts, styles, nav, footer),
and chunks extracted text into overlapping retrieval passages.
"""

import re
from typing import List, Dict, Any


class WebCrawlerAndChunker:
    """
    HTML DOM text extraction and fixed-window character chunking engine.
    """

    def __init__(self, chunk_size=400, chunk_overlap=80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def crawl_and_extract_text(self, html_or_url_content: str, source_url: str = "https://docs.enterprise-ai.org") -> Dict[str, Any]:
        """
        Strips HTML tags and extracts clean text passages.
        """
        # Strip script, style, and navigation tags
        clean_text = re.sub(r'<script.*?>.*?</script>', '', html_or_url_content, flags=re.DOTALL | re.IGNORECASE)
        clean_text = re.sub(r'<style.*?>.*?</style>', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
        clean_text = re.sub(r'<nav.*?>.*?</nav>', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
        clean_text = re.sub(r'<footer.*?>.*?</footer>', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove remaining HTML tags
        clean_text = re.sub(r'<.*?>', ' ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        chunks = self._recursive_chunk(clean_text, source_url)

        return {
            "source_url": source_url,
            "raw_character_count": len(clean_text),
            "num_chunks_extracted": len(chunks),
            "chunks": chunks
        }

    def _recursive_chunk(self, text: str, source_url: str) -> List[Dict[str, Any]]:
        """
        Splits text into overlapping word windows.
        """
        words = text.split()
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(words):
            end = min(start + self.chunk_size // 5, len(words))
            chunk_text = " ".join(words[start:end])
            
            chunks.append({
                "chunk_id": f"{source_url}#chunk_{chunk_id}",
                "source_url": source_url,
                "text": chunk_text,
                "token_count": len(chunk_text.split())
            })
            
            chunk_id += 1
            if end == len(words):
                break
            start += max(1, (self.chunk_size - self.chunk_overlap) // 5)

        return chunks
