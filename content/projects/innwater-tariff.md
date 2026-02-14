---
title: "InnWater – Water Tariff Dashboard"
date: 2026-02-02
summary: "AI-augmented economic simulation platform for sustainable and equitable water tariff design within the WEFE nexus."
tags: [ "rag", "water-governance" ]
tech: [ "Python", "FastAPI", "Angular", "PostgreSQL", "RAG", "LLM", "Docker" ]
cover:
  image: "/projects/innwater-tariff/innwater-tariff-simulation.png"
demo: "https://innwater.eurecatprojects.com/msm/"
repo: "https://innwater.eurecatprojects.com/msm/"
---

## Overview

The **InnWater Water Tariff Dashboard** is an AI-augmented decision-support system designed to simulate, analyze, and
optimize water pricing structures. Developed within the Horizon Europe *InnWater* project, it supports sustainable,
equitable, and economically efficient tariff design in multi-level water governance contexts.

I contributed as **AI Engineer/System Architect** at Eurecat, integrating the AI Assistant layer and supporting backend
architecture for economic simulation workflows.

---

## 💧 Water Tariff Simulation & Assessment

{{< figure
src="/projects/innwater-tariff/innwater-tariff-dashboard.png"
alt="Water Tariff Simulation"
caption="Water Tariff Dashboard."
width="80%"
align="center" >}}

The dashboard enables structured tariff analysis through:

* Simulation of alternative pricing schemes:
    * Flat rates
    * Increasing block tariffs
    * Progressive consumption models
* Comparison between sanitation and non-sanitation subscribers
* Comparison between poor and non-poor groups
* Environmental and resource cost internalization

The tool converts tariff design into a reproducible, scenario-based analytical workflow.

## 🤖 AI Assistant – RAG-based Economic Interpretation

{{< figure
src="/projects/innwater-tariff/innwater-tariff-diagram.png"
alt="Innwater AI Assistant Diagram"
caption="Innwater AI Assistant Diagram"
width="80%"
align="center" >}}

The Water Tariff Dashboard integrates a **Retrieval-Augmented Generation (RAG) Agent**  architecture, shared across the
InnWater platform.

### Architecture

* Query treatment module
* Hierarchical retrieval over indexed project deliverables
* Embedding-based semantic search
* LLM-grounded response generation
* Logging & evaluation via golden dataset benchmarking

### Capabilities

* Interprets tariff simulation outputs
* Explains affordability indicators in policy terms
* Highlights trade-offs between equity and cost recovery
* Suggests scenario adjustments
* Connects tariff outcomes with governance gaps and CGE model results
* Multilingual interaction

The AI layer transforms the dashboard from a numerical simulator into an **AI-assisted policy analysis tool**.

---

## 🏗 System Architecture

### Frontend

* Angular 15
* Bootstrap 5
* Chart.js & D3.js
* Interactive scenario comparison

### Backend

* FastAPI (Python)
* PostgreSQL
* SQLAlchemy ORM
* Pandas & NumPy for economic modeling

### AI Layer

* RAG pipeline
* Semantic embeddings
* LLM response generation
* Bias and robustness evaluation

### Deployment

* Docker-based containerization
* Integrated within the InnWater Governance Platform

---

## Impact

* Operationalized water tariff theory into an AI-supported economic decision tool
* Enabled evidence-based pricing design for water utilities
* Bridged tariff modeling with governance and macroeconomic simulation
* Demonstrated responsible AI deployment in public-sector economic policy  

