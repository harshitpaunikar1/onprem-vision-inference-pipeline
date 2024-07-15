# On-Prem Vision Inference Pipeline Diagrams

Generated on 2026-04-26T04:29:37Z from README narrative plus project blueprint requirements.

## On-prem vision pipeline architecture

```mermaid
flowchart TD
    N1["Step 1\nMapped image types, user goals, edge cases, system interfaces through discovery wo"]
    N2["Step 2\nHosted Gemma 3 and Llama 3.1 on secure on-prem servers for low-latency, private in"]
    N1 --> N2
    N3["Step 3\nBuilt prompt pipeline with task-specific templates and guardrails to steer extract"]
    N2 --> N3
    N4["Step 4\nNormalized outputs into predefined JSON schema; enforced validation rules and fall"]
    N3 --> N4
    N5["Step 5\nExposed ingestion via API gateway that buffered, authenticated, pushed to database"]
    N4 --> N5
```

## Prompt → JSON extraction flow

```mermaid
flowchart LR
    N1["Inputs\nImages or camera frames entering the inference workflow"]
    N2["Decision Layer\nPrompt → JSON extraction flow"]
    N1 --> N2
    N3["User Surface\nAPI-facing integration surface described in the README"]
    N2 --> N3
    N4["Business Outcome\nInference or response latency"]
    N3 --> N4
```

## Evidence Gap Map

```mermaid
flowchart LR
    N1["Present\nREADME, diagrams.md, local SVG assets"]
    N2["Missing\nSource code, screenshots, raw datasets"]
    N1 --> N2
    N3["Next Task\nReplace inferred notes with checked-in artifacts"]
    N2 --> N3
```
