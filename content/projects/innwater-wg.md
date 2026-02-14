---
title: "InnWater – AI-Augmented Water Governance Platform"
date: 2025-10-01
summary: "AI-powered decision-support system for water governance assessment and policy optimization within the WEFE nexus."
tags: [ "nlp", "rag", "water-governance" ]
tech: [ "Python", "FastAPI", "Angular", "RAG", "LLM", "Qdrant" ]
cover:
  image: "/projects/innwater-wg/innwater-cover-page.png"
repo: "https://github.com/Applied-Artificial-Intelligence-Eurecat/water-governance-diagnosis-tool"
demo: "https://innwater.eurecatprojects.com/innwater/"
---

## Overview

InnWater is a Horizon Europe project promoting social innovation in multi-level water governance within the WEFE nexus (Water–Energy–Food–Ecosystem). I contributed as **AI Engineer/System Architect** at Eurecat, designing the **Water Governance Diagnosis Tool** and **AI Assistant** integrated into the platform.

---

## 🔍 Water Governance Diagnosis Tool

{{< figure
src="/projects/innwater-wg/innwater-results-and-assessment.png"
alt="Data"
caption="Water Governance Diagnosis Results and assessment."
width="90%"
align="center" >}}

Digitized the **OECD Water Governance Principles** into a structured analytics pipeline:

* Interactive questionnaire for governance assessment
* Quantitative scoring engine with classification:
  * Governance Gap (< 1.75)
  * Moderate Governance (< 2.70)
  * Strong Governance (> 2.70)
* Results dashboard with interpretability layer
* Integration with CGE economic simulation model

The tool converts qualitative governance inputs into reproducible, policy-oriented outputs.

---

## 🤖 AI Assistant – RAG-based Support

{{< figure
src="/projects/innwater-wg/innwater-ai-chat.png"
alt="Data"
caption="Water Governance Assistant."
width="90%"
align="center" >}}

Designed and implemented a **Retrieval-Augmented Generation (RAG)** architecture to support governance navigation:

**Architecture:**
* Query treatment module
* Hierarchical retrieval over indexed project deliverables
* Embedding-based semantic search
* LLM response generation grounded in validated documents
* Logging & evaluation (golden dataset benchmarking)

**Capabilities:**
* Explains OECD governance principles
* Interprets governance scores
* Suggests policy improvements
* Connects governance gaps with economic scenarios
* Multilingual interaction

---

## ⚖ Ethical & Trustworthy AI

Implemented safeguards aligned with EU AI Act, GDPR, and ALTAI framework:

* Citation-grounded generation
* Bias validation datasets
* Toxicity classifier
* Multilingual fairness evaluation
* Transparent AI disclaimers

---

## Impact

* Operationalized governance theory into AI-supported digital infrastructure
* Enabled non-expert stakeholders to interpret complex water governance data
* Bridged governance assessment with economic modeling
* Demonstrated responsible AI deployment in public-sector decision support
