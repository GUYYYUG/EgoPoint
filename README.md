# EgoPoint-Bench: Benchmarking and Enhancing Referential Reasoning in Egocentric Vision



## 1. Introduction

EgoPoint-Bench is a comprehensive Question-Answering (QA) benchmark designed to evaluate and enhance multimodal pointing reasoning in egocentric views. Unlike existing datasets that rely on third-person views or explicit text descriptions, EgoPoint-Bench focuses on the **spatial semantics of pointing**, addressing the critical issue of "Referential Hallucination" in current Multimodal Large Language Models (MLLMs).

Our benchmark comprises **11,729 high-fidelity samples** (10,567 simulation and 1,162 real-world), bridging the gap between simulated physics-based supervision and real-world application.

## 2. Dataset Overview & Statistics

EgoPoint-Bench is constructed using a dual-source strategy: a scalable simulation pipeline (**Point-Sim**) and rigorous **Real-world** collection.

![egopoint-overview](./paper_img/egopoint-overview.pdf)

> *Overview of the EgoPoint-Bench construction pipeline, including Point-Sim framework, real-world data collection, and capability dimensions.*

### 2.1 Data Scale and Distribution

| **Source** | **Subset**       | **Train** | **Val** | **Test**  | **Total**  | **Avg. QA Len** |
| ---------- | ---------------- | --------- | ------- | --------- | ---------- | --------------- |
| **Sim**    | HM3D             | 3,227     | 365     | 718       | 4,310      | 10.12           |
|            | HSSD             | 1,964     | 214     | 605       | 2,783      | 8.68            |
|            | AI2-THOR         | 1,982     | 220     | 606       | 2,808      | 10.22           |
|            | ReplicaCAD       | 601       | 65      | -         | 666        | 8.67            |
| **Real**   | MLVision Capture | -         | -       | **1,162** | **1,162**  | **11.02**       |
| **Total**  |                  | **7,774** | **864** | **2,485** | **11,729** | **9.81**        |

### 2.2 Capability Taxonomy (5 Dimensions)

To ensure comprehensive evaluation, we structured the benchmark across five core capability dimensions:

1. **Basic Perception (BP):** Identifies fundamental attributes (category, color, texture) aligned with gestures.
2. **Function & State (FS):** Infers semantic properties (e.g., edibility, operability) and states.
3. **Spatial Context (SC):** Perceives egocentric spatial relationships and reachability.
4. **OCR:** Extracts textual information from pointed targets (brands, slogans).
5. **Adversarial Resilience (AR):** Evaluates reliability against counterfactuals and void references (e.g., pointing at empty space).

### 2.3 Hierarchical Deixis Levels

We introduce a hierarchical taxonomy to cover the full spectrum of referential ambiguity:

- **L1 (Explicit Action):** Explicitly describes the gesture (e.g., *"the object I am pointing at"*).
- **L2 (Visual Locative):** Implies spatial proximity (e.g., *"that thing over there"*).
- **L3 (Implicit Pronoun):** Relies purely on visual context (e.g., *"this"*).

![Figure 2c: Distribution of Deixis Levels](./paper_img/deixis_level.pdf)

> *Distribution of Deixis Levels across Sim and Real datasets.*

### 2.4 Question Types

To balance ecological validity with objective benchmarking, we employ a hybrid format:

- **Multiple Choice (SCQ):** For rigorous automated evaluation.
- **True/False (TF):** For rapid discriminative testing.
- **Open-Ended (OQ):** To reflect natural human inquiry (evaluated via LLM-as-a-Judge).

![Figure 9: Option distribution](./paper_img/answer_statistics.png)

> *Distribution of answer types in the training set, ensuring no answer bias.*

## 3. Motivation: The Problem of "Referential Hallucination"

Current state-of-the-art MLLMs often fail to precisely ground the spatial semantics of pointing. Instead of tracing the precise geometric projection of the pointing finger, models frequently fixate on:

- **Proximal Distraction:** Objects immediately adjacent to the hand.

- **Object Saliency:** Visually prominent entities regardless of the pointing ray.

![Figure 9: error samples](./paper_img/error_lora.pdf)

We term this phenomenon **"Referential Hallucination"**.

## 3. Repository Structure

This repository contains the necessary code and a mini-dataset sample to replicate our evaluation pipeline and inspect data quality.

Plaintext

```
.
├── eval_qwen3vl.py                 # Main evaluation script for Qwen3-VL models
├── requirements.txt                # Python environment dependencies
├── mini_dataset_img/               # Sample images (Real & Sim) for verification
│   ├── real_001.jpg
│   ├── sim_001.jpg
│   └── ...
├── mini_dataset.json               # Metadata for sample images (Q, A, Dimensions)
├── evaluation_qwen_results.json    # Baseline results on the mini-dataset (Qwen3-VL-8B)
└── evaluation_qwen_lora_results.json # Results after LoRA fine-tuning on Sim data
```

## 4. Getting Started

### 4.1 Environment Setup

The code is tested on **Python 3.10** with **NVIDIA A100** GPUs.

Bash

```
conda create -n egopoint python=3.10
conda activate egopoint
pip install -r requirements.txt
```

### 4.2 Data Format

The dataset uses a JSON format that encapsulates all spatial and semantic annotations. Example from `mini_dataset.json`:

JSON

```
{
      "image_id": "real_9",
      "question": "What best describes the object I am pointing at?",
      "options": [
          "A. Ground",
          "B. Black bricks",
          "C. Wolf",
          "D. Glass"
      ],
      "answer": "C. Wolf",
      "dimension": "Basic Perception",
      "deixis_level": "L1",
      "type": "Multiple_Choice"
  },
```

### 4.3 Running Evaluation

To run the evaluation on the provided mini-dataset using the `Qwen3-VL` model:

Bash

```
python eval_qwen3vl.py
```

*Note: The script `eval_qwen3vl.py` is configured to automatically load the model, process the `mini_dataset.json`, and output the accuracy statistics for Multiple Choice and True/False questions.*

## 5. Evaluation Results (Sample)

We provide sample outputs in `evaluation_qwen_lora_results.json` demonstrating the effectiveness of our **Sim-to-Real** transfer.

------

**Note:** The full dataset (11k+ samples) and full training scripts will be released upon acceptance. This repository serves as a functional demonstration of the data format and evaluation protocol.
