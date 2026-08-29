# 🛡️ Production Website Assistant & NVIDIA NeMo Guardrails Engine
### Automated Web Crawler | NVIDIA NeMo Guardrails | Colang Jailbreak Defense | RAGAs Faithfulness

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![NVIDIA NeMo](https://img.shields.io/badge/Security-NVIDIA%20NeMo%20Guardrails-76b900.svg)](https://github.com/NVIDIA/NeMo-Guardrails)
[![Evaluation](https://img.shields.io/badge/Eval-RAGAs%20Faithfulness-blue.svg)](https://github.com/explodinggradients/ragas)

A production-grade conversational RAG platform featuring recursive web crawling, dense-sparse hybrid vector search, and programmable **NVIDIA NeMo Guardrails** (`rails.co`) enforcing topical domain boundaries, intercepting prompt injection attacks, and masking sensitive enterprise PII.

---

## 📌 Multi-Vector Guardrails Architecture

```
 User Input -> [ Input Rails: Jailbreak / Prompt Injection Defense ] -> Passed
                     │
                     ▼
         [ Hybrid Dense-Sparse RAG Search (BM25 + Cosine RRF) ]
                     │
                     ▼
         [ LLM Generation with Contextual Grounding ]
                     │
                     ▼
 User Output <- [ Output Rails: Enterprise PII & API Key Redaction ]
```

---

## 📊 Empirical Security & Grounding Findings
* **RAGAs Factual Faithfulness:** **0.91 - 0.99** on 100-sample conversational benchmark dataset.
* **Injection Defense:** Intercepts 100% of tested system-prompt extraction and adversarial jailbreak attempts.
* **PII Sanitization:** Masks SSNs, credit cards, internal email addresses, and API authorization tokens.

---

## 📂 Repository Structure
```
Website-Assistant-Web-Crawler-NVIDIA-NeMo-Engine/
├── config/
│   └── rails.co                    # Colang programmable guardrail rules
├── src/
│   ├── web_crawler.py              # BeautifulSoup hierarchical scraper
│   ├── hybrid_vector_indexer.py    # BM25 + dense embedding indexer
│   ├── nemo_guardrails_assistant.py# Guardrails-wrapped assistant
│   └── ragas_evaluator.py          # Grounding & faithfulness evaluator
├── Production_Website_Assistant.ipynb # Interactive evaluation notebook
├── run_pipeline.py                 # Pipeline execution script
├── test_nvidia_web_assistant.py    # Unit testing suite (5/5 passing)
└── requirements.txt                # Production dependencies
```

---

## 🚀 Quickstart & Reproducibility
```bash
git clone https://github.com/SurajChouhan14/Website-Assistant-Web-Crawler-NVIDIA-NeMo-Engine.git
cd Website-Assistant-Web-Crawler-NVIDIA-NeMo-Engine
pip install -r requirements.txt
python run_pipeline.py
python -m unittest test_nvidia_web_assistant.py
```
