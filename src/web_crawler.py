"""
Enterprise Web Content Extractor & DOM Chunking Module.
Uses BeautifulSoup (bs4) to parse HTML structures, strip boilerplate DOM tags
(scripts, styles, nav, footer, header), discover recursive hyperlinks, and chunk extracted text into overlapping retrieval passages.
"""

import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup


class WebCrawlerAndChunker:
    """
    HTML DOM text extraction and fixed-window character chunking engine powered by BeautifulSoup.
    """

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80, parser: str = "html.parser"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parser = parser

    def crawl_and_extract_text(self, html_or_url_content: str, source_url: str = "https://docs.enterprise-ai.org") -> Dict[str, Any]:
        """
        Uses BeautifulSoup to strip boilerplate HTML tags, discover links, and extract clean text passages.
        """
        soup = BeautifulSoup(html_or_url_content, self.parser)

        # Discover internal and external hyperlinks for recursive crawling
        discovered_links = [
            a.get("href") for a in soup.find_all("a", href=True)
            if not a.get("href", "").startswith("#")
        ]

        # Strip non-informative DOM tags (scripts, styles, navigation, footers, headers)
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
            tag.decompose()

        # Extract normalized whitespace text
        clean_text = soup.get_text(separator=" ", strip=True)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        chunks = self._recursive_chunk(clean_text, source_url)

        return {
            "source_url": source_url,
            "raw_character_count": len(clean_text),
            "num_chunks_extracted": len(chunks),
            "discovered_links": discovered_links,
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
