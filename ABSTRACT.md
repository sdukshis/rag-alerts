# Enhancing Observability Alerts with RAG

## Abstract

The process of handling alerts from monitoring systems includes stages such as detection, triage, investigation, and response. The most important metric of the entire process is incident response time, which is composed of the durations of the individual stages. Machine learning technologies can help reduce time at different stages.

In this workshop, we focus on the investigation stage, where a triggered alert and observed metrics must be analyzed to understand what has happened. During investigation, SRE specialists often need to gather additional information, find similar past alerts, and prepare a response plan. Depending on system complexity and specialist experience, this stage can take a significant portion of the overall response time.

Retrieval-Augmented Generation (RAG) allows us to enrich an alert with additional context at creation time, reducing investigation effort and accelerating incident response.

We will review libraries and tools in Python and cloud-based LLMs to build a system for enriching monitoring alerts.

## Workshop Plan

1. Introduction to RAG
2. Environment setup
3. Knowledge base of past incidents
4. Building a RAG pipeline
5. Building document and query embeddings
6. Reranking
7. Prompt engineering
8. Alert enrichment
9. Conclusions and possible future development

## Tools Used

- Python
- OpenAI
- Hugging Face
- Grafana

