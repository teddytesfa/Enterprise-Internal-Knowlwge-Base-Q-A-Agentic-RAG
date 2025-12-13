# Eval Driven Development (EDD) Implementation Strategy

## 1. Executive Summary

This document outlines the strategy for implementing Evaluation Driven Development (EDD) for the Enterprise Internal Knowledge Base Q&A Agentic RAG. The goal is to systematically measure and improve the agent's performance by moving away from ad-hoc testing to a robust, data-driven evaluation pipeline. This approach directly addresses the real-world problems of poor search relevance, information overload, and outdated/conflicting documentation.

## 2. Core Methodology: The "Golden Dataset"

The foundation of our EDD strategy is the creation of a "Golden Dataset"—a curated collection of inputs (questions) and expected outputs (answers/citations) that represent real-world usage.

### 2.1 Components
1.  **Personas:** Archetypal users with specific goals, technical levels, and frustration points.
2.  **Scenarios:** Specific situations or failure modes (e.g., "Conflict Resolution", "Specific Process Search").
3.  **Synthetic Data:** A large set of generated questions and answers mapped to the personas and scenarios.

### 2.2 Iteration Loop
1.  **Build/Update Golden Dataset:** Define new scenarios or refine existing ones.
2.  **Run Evals:** execute the RAG agent against the dataset.
3.  **Analyze Failures:** Identify patterns (e.g., "Agent prefers outdated PDF over new Wiki").
4.  **Refine Agent:** detailed engineering (prompt tuning, retrieval configuration) to fix specific failure modes.
5.  **Repeat.**

## 3. Personas & Scenarios Definition

### 3.1 Personas
We will define personas to ensure coverage across different user types. initial personas:
*   **The Onboarding New Hire:** Needs high-level process info, easily confused by jargon.
*   **The Senior Engineer:** Needs specific technical details, commands, and accurate configurations.
*   **The Product Manager:** Needs project status, decision logs, and high-level timelines.

### 3.2 Failure Mode Scenarios (The "Why")
We are targeting these specific business problems:
*   **Poor Search Relevance:** The agent returns "something" but not the *specific* thing asked for.
*   **Information Overload:** The agent dumps too much context without synthesizing a clear answer.
*   **Conflict Resolution:** The agent fails to distinguish between an old v1 process and a new v2 process.

## 4. Implementation Steps

### Phase 1: Foundation (Current)
*   Define Personas in `data/edd/personas.json`
*   Define Scenarios in `data/edd/scenarios.json`
*   Generate initial Synthetic Questions in `data/edd/synthetic_questions.json`

### Phase 2: Instrumentation (Next)
*   Build an evaluation script `src/eval/run_eval.py` to:
    *   Load the synthetic questions.
    *   Query the RAG agent.
    *   Compare results (Hybrid approach: **LLM-as-a-judge** for scale + **Manual Labeling** for ground-truth calibration).
    *   Output a report.

### Phase 3: Automation & CI (Portfolio Showcase)
*   **Lightweight CI Demonstration:** Instead of a complex, costly pipeline, we will add a simple GitHub Action workflow file (`.github/workflows/eval_demo.yml`) as a **reference implementation**.
    *   This serves as "educational content" for portfolio viewers, demonstrating *how* to set up automated EDD without incurring high runtime costs.
    *   It will run a "dry run" or a tiny subset of tests to prove the concept.

## 5. Metrics
*   **Relevance Score (1-5):** How effectively did the answer address the core question?
*   **Citation Accuracy (Binary):** Did the agent cite the correct "Source of Truth"?
*   **Synthesis Quality (1-5):** Was the answer a coherent summary or a copy-paste dump?
