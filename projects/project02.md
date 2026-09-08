---
title: Small Project 2 - Non-Autoregressive Architectures & Agentic Systems
---

# Small Project 2: Non-Autoregressive Architectures & Agentic Systems

## 1. Context & Objectives
While autoregressive generative AI (LLMs) dominates contemporary discussions, emerging paradigms address fundamental limits of autoregression (hallucinations, exposure bias, high latency, lack of grounded world dynamics):
* **Joint Embedding Predictive Architectures (JEPA):** Learning representations by predicting in latent feature space rather than input/pixel space (LeCun).
* **World Models:** Learning compact latent transition dynamics for planning and control (Ha & Schmidhuber, Dreamer).
* **Agentic AI Systems:** Autonomous goal-directed loops combining tool use, memory, and reflection.

Students will choose one of two tracks:
* **Track A (Representation & Dynamics):** Implement and evaluate a simplified JEPA or World Model architecture on a continuous control or visual prediction benchmark.
* **Track B (Agentic AI Systems):** Build a multi-step autonomous agent with tool execution, verification loops, and self-correction, evaluating failure modes and robustness against prompt perturbation.

## 2. Deliverables
1. **Source Code & Implementation:** Reproducible repository with environment configuration and automated tests.
2. **Technical Report (PDF, max 6 pages):**
   - Architectural formulation and design choices.
   - Quantitative evaluation against baseline models.
   - Ablation analysis (e.g. latent space dimension, tool failure recovery rate).
3. **Live Demonstration:** Brief 5-minute presentation in Class 14.

## 3. Important Dates & Submission
* **Release:** Week 9 (November 12-13, 2026)
* **Submission Deadline:** Class 14 (December 17-18, 2026)
* **Weight:** 25% of final course grade.
