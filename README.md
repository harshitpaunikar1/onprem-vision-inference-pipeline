# On-Prem Vision Inference Pipeline

> **Domain:** Manufacturing

## Overview

Manual review of incoming images slows operations and increases sensitive data exposure. Teams need consistent, structured outputs integrating cleanly with existing systems, yet current processes are subjective and hard to scale. Strict privacy constraints limit third-party AI use. When outputs are not standardized, downstream analytics break and decisions get delayed. Without automation and on-prem controls, costs rise with growing volume, response times slip, and compliance risk increases.

## Approach

- Mapped image types, user goals, edge cases, system interfaces through discovery workshops
- Hosted Gemma 3 and Llama 3.1 on secure on-prem servers for low-latency, private inference
- Built prompt pipeline with task-specific templates and guardrails to steer extraction
- Normalized outputs into predefined JSON schema; enforced validation rules and fallbacks
- Exposed ingestion via API gateway that buffered, authenticated, pushed to database
- Enabled no-code prompt configuration, A/B tests, monitoring; iterated weekly with test sets

## Skills & Technologies

- Gemma 3
- Llama 3.1
- Prompt Engineering
- Computer Vision Inference
- JSON Schema Design
- API Gateway Integration
- Python
- Data Privacy & Security
