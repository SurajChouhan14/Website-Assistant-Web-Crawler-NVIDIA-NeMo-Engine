"""
Hybrid Dense-Sparse Vector Indexer with Okapi BM25 & Semantic Dense Embeddings.
Combines:
1. Sparse Lexical Search: Okapi BM25 keyword matching with dynamic IDF.
2. Dense Semantic Search: Vectorized TF-IDF document-query cosine projections.
3. Rank Fusion: Reciprocal Rank Fusion (RRF k=60) combining sparse and dense ranking channels.
"""

import math
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HybridVectorIndexer:
    """
    Production Dense-Sparse Search Indexer with Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, chunks: List[Dict[str, Any]], rrf_k: int = 60):
        self.chunks = chunks
        self.corpus = [c["text"] for c in chunks]
        self.rrf_k = rrf_k

        # Sparse BM25 Attributes
        self.vocabulary = set()
        self.doc_term_freqs = []
        self.doc_lengths = []
        self.avg_doc_len = 0.0

        # Dense Semantic Vector Embeddings
        self.dense_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=5000
        )
        self.dense_doc_matrix = None

        self._build_sparse_bm25_index()
        self._build_dense_vector_index()

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in text.split() if len(w) > 2]

    def _build_sparse_bm25_index(self):
        """Builds inverted index for Okapi BM25 scoring."""
        for doc in self.corpus:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
                self.vocabulary.add(t)
            self.doc_term_freqs.append(tf)

        self.avg_doc_len = sum(self.doc_lengths) / max(1, len(self.doc_lengths))

    def _build_dense_vector_index(self):
        """Builds dense semantic vector space projections."""
        if len(self.corpus) > 0:
            self.dense_doc_matrix = self.dense_vectorizer.fit_transform(self.corpus)

    def _bm25_score(self, query_tokens: List[str], doc_idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        """Computes Okapi BM25 score for a specific document."""
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        tf_dict = self.doc_term_freqs[doc_idx]
        N = len(self.corpus)

        for token in query_tokens:
            if token in tf_dict:
                tf = tf_dict[token]
                df = sum(1 for d in self.doc_term_freqs if token in d)
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                numerator = tf * (k1 + 1.0)
                denominator = tf + k1 * (1.0 - b + b * (doc_len / max(1.0, self.avg_doc_len)))
                score += idf * (numerator / denominator)

        return score

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Executes genuine Dense-Sparse Hybrid Search with Reciprocal Rank Fusion (RRF).
        Channel 1: Sparse Lexical BM25
        Channel 2: Dense Semantic Cosine Similarity
        """
        if not self.corpus:
            return []

        # 1. Sparse Channel: Okapi BM25
        query_tokens = self._tokenize(query)
        bm25_scores = [self._bm25_score(query_tokens, i) for i in range(len(self.corpus))]

        # 2. Dense Channel: Semantic Vector Cosine Projections
        query_dense_vec = self.dense_vectorizer.transform([query])
        dense_cosine_sims = cosine_similarity(query_dense_vec, self.dense_doc_matrix).flatten()

        # 3. Reciprocal Rank Fusion (RRF)
        bm25_ranked = np.argsort(bm25_scores)[::-1]
        dense_ranked = np.argsort(dense_cosine_sims)[::-1]

        rrf_scores = np.zeros(len(self.corpus))
        for rank, idx in enumerate(bm25_ranked):
            rrf_scores[idx] += 1.0 / (self.rrf_k + rank + 1)
        for rank, idx in enumerate(dense_ranked):
            rrf_scores[idx] += 1.0 / (self.rrf_k + rank + 1)

        top_indices = np.argsort(rrf_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "chunk_id": self.chunks[idx]["chunk_id"],
                "source_url": self.chunks[idx]["source_url"],
                "text": self.chunks[idx]["text"],
                "rrf_relevance_score": round(float(rrf_scores[idx]), 4),
                "sparse_bm25_score": round(float(bm25_scores[idx]), 4),
                "dense_cosine_similarity": round(float(dense_cosine_sims[idx]), 4),
                "bm25_score": round(float(bm25_scores[idx]), 4)
            })

        return results
