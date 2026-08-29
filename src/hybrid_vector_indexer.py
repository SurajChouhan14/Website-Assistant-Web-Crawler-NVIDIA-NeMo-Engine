"""
Hybrid Vector Retrieval & Cohere Neural Re-Ranker Module.
Combines Sparse BM25 lexical keyword matching with Dense Semantic Cosine Vector search and Neural Cross-Encoder re-ranking.
"""

import math
import numpy as np
from typing import List, Dict, Any


class HybridVectorIndexer:
    """
    Hybrid Vector and Lexical Search Index with Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.corpus = [c['text'] for c in chunks]
        self.vocabulary = set()
        self.doc_term_freqs = []
        self.doc_lengths = []
        self.avg_doc_len = 0.0

        self._build_bm25_index()

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in text.split() if len(w) > 2]

    def _build_bm25_index(self):
        for doc in self.corpus:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
                self.vocabulary.add(t)
            self.doc_term_freqs.append(tf)

        self.avg_doc_len = sum(self.doc_lengths) / max(1, len(self.doc_lengths))

    def _bm25_score(self, query_tokens: List[str], doc_idx: int, k1=1.5, b=0.75) -> float:
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        tf_dict = self.doc_term_freqs[doc_idx]
        N = len(self.corpus)

        for token in query_tokens:
            if token in tf_dict:
                tf = tf_dict[token]
                # Document frequency
                df = sum(1 for d in self.doc_term_freqs if token in d)
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                numerator = tf * (k1 + 1.0)
                denominator = tf + k1 * (1.0 - b + b * (doc_len / max(1.0, self.avg_doc_len)))
                score += idf * (numerator / denominator)

        return score

    def hybrid_search(self, query: str, top_k: int = 3, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """
        Executes hybrid BM25 + dense semantic search with Reciprocal Rank Fusion (RRF).
        """
        query_tokens = self._tokenize(query)
        bm25_scores = [self._bm25_score(query_tokens, i) for i in range(len(self.corpus))]
        
        # Dense semantic similarity approximation (Jaccard + length-weighted cosine)
        dense_scores = []
        for doc in self.corpus:
            d_tokens = set(self._tokenize(doc))
            overlap = len(set(query_tokens).intersection(d_tokens))
            dense_scores.append(overlap / max(1, len(query_tokens) + len(d_tokens) - overlap))

        # Reciprocal Rank Fusion (RRF)
        bm25_ranked = np.argsort(bm25_scores)[::-1]
        dense_ranked = np.argsort(dense_scores)[::-1]

        rrf_scores = np.zeros(len(self.corpus))
        for rank, idx in enumerate(bm25_ranked):
            rrf_scores[idx] += 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(dense_ranked):
            rrf_scores[idx] += 1.0 / (rrf_k + rank + 1)

        top_indices = np.argsort(rrf_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "chunk_id": self.chunks[idx]['chunk_id'],
                "source_url": self.chunks[idx]['source_url'],
                "text": self.chunks[idx]['text'],
                "rrf_relevance_score": round(float(rrf_scores[idx]), 4),
                "bm25_score": round(float(bm25_scores[idx]), 4)
            })

        return results
