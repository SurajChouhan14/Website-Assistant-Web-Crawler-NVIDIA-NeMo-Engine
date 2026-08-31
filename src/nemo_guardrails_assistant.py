"""
Deterministic Security Guardrails & Enterprise Website Assistant Engine.
Implements NeMo Guardrails-inspired architectural rails:
1. Input Rails: Prompt injection and out-of-domain interception via deterministic regex patterns
2. Output Rails: Multi-pattern enterprise PII redaction (SSN, credit cards, internal emails, API keys)
3. Colang Companion: Ships config/rails.co definitions for enterprise NeMo deployment
"""

import os
import re
from typing import Dict, Any, List, Optional


class NVIDIANeMoWebsiteAssistant:
    """
    Website Assistant with deterministic input jailbreak and output PII sanitization guardrails.
    """

    def __init__(self, indexer, config_dir: Optional[str] = None):
        self.indexer = indexer
        if config_dir is None:
            config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
        self.config_dir = config_dir

        self.colang_rules = self._load_colang_definitions()
        
        # Input Guardrail Injection Rules
        self.injection_patterns = [
            r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
            r"system\s+prompt\s+override",
            r"you\s+are\s+now\s+(DAN|unfiltered|jailbroken|an\s+unconstrained\s+AI)",
            r"bypass\s+(security|safety|guardrails)",
            r"drop\s+table|delete\s+from|rm\s+-rf|sudo",
            r"dump\s+internal\s+passwords",
            r"print\s+(your\s+)?(initial|system)\s+prompt"
        ]

        # Output Guardrail PII & Secret Redaction Rules
        self.pii_rules = [
            ("SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
            ("CREDIT_CARD", r"\b(?:\d{4}[ -]?){3}\d{4}\b"),
            ("API_KEY", r"\b(?:sk|ghp|pk)_[a-zA-Z0-9_-]{20,}\b"),
            ("INTERNAL_EMAIL", r"\b[A-Za-z0-9._%+-]+@internal-enterprise\.[A-Za-z]{2,}\b")
        ]

    def _load_colang_definitions(self) -> Dict[str, Any]:
        """Loads Colang (.co) and YAML (.yml) definition files from config directory."""
        colang_path = os.path.join(self.config_dir, "rails.co")
        config_path = os.path.join(self.config_dir, "config.yml")
        rules = {"colang_loaded": False, "config_loaded": False}

        if os.path.exists(colang_path):
            with open(colang_path, 'r', encoding='utf-8') as f:
                rules["colang_source"] = f.read()
                rules["colang_loaded"] = True

        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                rules["config_source"] = f.read()
                rules["config_loaded"] = True

        return rules

    def validate_input_rail(self, query: str) -> Dict[str, Any]:
        """
        Input Guardrail: Intercepts prompt injections and off-domain queries.
        """
        for pattern in self.injection_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return {
                    "passed": False,
                    "rail_name": "Input_Jailbreak_Shield",
                    "reason": f"Adversarial prompt injection pattern detected: {pattern}",
                    "intercept_message": "[GUARDRAIL BLOCKED]: Request intercepted. Prompt injection or system override detected."
                }

        # Check off-topic query
        off_topic_patterns = [r"how\s+to\s+bake", r"recipe\s+for", r"who\s+won\s+the\s+world\s+cup"]
        for pattern in off_topic_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return {
                    "passed": False,
                    "rail_name": "Input_Domain_Shield",
                    "reason": "Query is outside enterprise documentation domain",
                    "intercept_message": "[GUARDRAIL BLOCKED]: Query falls outside enterprise documentation scope. Please ask questions related to platform architecture."
                }

        return {"passed": True, "rail_name": "Input_Jailbreak_Shield", "reason": None}

    def sanitize_output_rail(self, response_text: str) -> Dict[str, Any]:
        """
        Output Guardrail: Identifies and redacts sensitive PII, emails, and API keys.
        """
        sanitized = response_text
        redactions = []

        for pii_type, pattern in self.pii_rules:
            matches = re.findall(pattern, sanitized)
            if matches:
                redactions.extend([(pii_type, m) for m in matches])
                sanitized = re.sub(pattern, f"[REDACTED_{pii_type}]", sanitized)

        return {
            "output_rail_sanitized": True,
            "sanitized_response": sanitized,
            "num_redactions": len(redactions),
            "redacted_items": redactions
        }

    def answer_query(self, user_query: str, top_k: int = 2) -> Dict[str, Any]:
        """
        Executes Input Rails -> Hybrid BM25 Search -> Grounded Synthesis -> Output Sanitization.
        """
        # 1. Execute Input Rails
        input_rail_res = self.validate_input_rail(user_query)
        if not input_rail_res["passed"]:
            return {
                "user_query": user_query,
                "guardrail_status": "INTERCEPTED_BY_INPUT_RAIL",
                "rail_intercepted": input_rail_res["rail_name"],
                "final_answer": input_rail_res["intercept_message"],
                "retrieved_passages": [],
                "telemetry": {
                    "input_rail_passed": False,
                    "output_rail_passed": None,
                    "regex_guardrails_enforced": True,
                    "colang_config_loaded": bool(self.colang_rules.get("colang_loaded", False))
                }
            }

        # 2. Hybrid Retrieval (BM25 + Token Overlap RRF)
        retrieved_docs = self.indexer.hybrid_search(user_query, top_k=top_k)

        # 3. Grounded Extractive Template Synthesis
        if not retrieved_docs:
            raw_response = "I could not find relevant documentation on the platform to answer your question."
        else:
            raw_response = (
                f"Based on the official website documentation:\n"
                f"{retrieved_docs[0]['text']}\n\n"
                f"For further details, refer to {retrieved_docs[0]['source_url']}."
            )

        # 4. Execute Output Rails (PII Sanitization)
        output_rail_res = self.sanitize_output_rail(raw_response)

        return {
            "user_query": user_query,
            "guardrail_status": "PASSED_ALL_GUARDRAILS",
            "rail_intercepted": None,
            "final_answer": output_rail_res["sanitized_response"],
            "retrieved_passages": retrieved_docs,
            "telemetry": {
                "input_rail_passed": True,
                "output_rail_passed": True,
                "pii_redacted_count": output_rail_res["num_redactions"],
                "regex_guardrails_enforced": True,
                "colang_config_loaded": bool(self.colang_rules.get("colang_loaded", False))
            }
        }
