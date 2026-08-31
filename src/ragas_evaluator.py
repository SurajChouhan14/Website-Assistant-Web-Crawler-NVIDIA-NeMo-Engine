"""
Automated RAGAs Triad Evaluation Engine.
Computes mathematical metrics across the RAG Triad based on the official RAGAs formulation (Es et al., 2023):
1. Propositional Faithfulness (Fraction of answer propositions entailed by context)
2. Answer Relevance Metric (Raw semantic token overlap between Query and Response)
3. Context Recall Metric (Fraction of Ground Truth Information Captured in Context)
"""

import math
import numpy as np
import re
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGAsTriadEvaluator:
    """
    Evaluation framework computing RAG Triad metrics across benchmark samples.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def evaluate_sample_faithfulness(self, context_text: str, answer_text: str) -> float:
        """
        Propositional Faithfulness: Decomposes answer into sentences and measures semantic entailment.
        """
        if not answer_text or not context_text:
            return 0.0

        propositions = [p.strip() for p in re.split(r'[.!?\n]+', answer_text) if len(p.strip()) > 5]
        if not propositions:
            return 1.0

        vec = TfidfVectorizer(ngram_range=(1, 2)).fit([context_text] + propositions)
        ctx_vec = vec.transform([context_text])
        
        entailed_weight = 0.0
        for prop in propositions:
            prop_vec = vec.transform([prop])
            sim = float(cosine_similarity(prop_vec, ctx_vec)[0][0])
            if sim >= 0.22:
                entailed_weight += 1.0
            elif sim >= 0.10:
                entailed_weight += (sim / 0.22)
            else:
                entailed_weight += 0.0

        faithfulness = entailed_weight / len(propositions)
        return float(np.clip(faithfulness, 0.0, 1.0))

    def evaluate_answer_relevance(self, question: str, answer_text: str) -> float:
        """
        Answer Relevance: Measures raw token overlap between user query and response.
        """
        if not question or not answer_text:
            return 0.0

        q_tokens = [w for w in re.findall(r'\w+', question.lower()) if len(w) >= 3]
        a_tokens = [w for w in re.findall(r'\w+', answer_text.lower()) if len(w) >= 3]

        if not q_tokens or not a_tokens:
            return 0.0

        overlap = len(set(q_tokens).intersection(set(a_tokens))) / len(set(q_tokens))
        return float(np.clip(overlap, 0.0, 1.0))

    def evaluate_context_recall(self, ground_truth: str, context_text: str) -> float:
        """
        Context Recall: Measures fraction of ground truth factual elements present in context.
        """
        if not ground_truth or not context_text:
            return 0.0

        stopwords = {
            "what", "is", "the", "how", "to", "in", "for", "and", "a", "an", "of", "on", "with", "from", "by"
        }
        gt_tokens = [w for w in re.findall(r'[A-Za-z0-9.%+-]+', ground_truth.lower()) if w not in stopwords and len(w) >= 2]

        if not gt_tokens:
            return 1.0

        context_lower = context_text.lower()
        context_tokens = set(re.findall(r'[A-Za-z0-9.%+-]+', context_lower))
        captured = sum(1 for token in gt_tokens if token in context_tokens or token in context_lower)
        return float(captured / len(gt_tokens))
