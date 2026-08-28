# 🧠 LLMs on a Budget: Say HOLA

### Efficient Instruction Tuning with Hierarchical Sparsity

<p align="center">

**A lightweight framework for improving the efficiency and cross-domain robustness of small language models under resource constraints.**

<br>

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Research-LLM%20Efficiency-purple.svg)](#)
[![Status](https://img.shields.io/badge/Status-Research-orange.svg)](#)

</p>

---

## 📌 Overview

**HOLA** is a lightweight instruction-tuning framework designed for deploying language models in **resource-constrained environments**.

The project investigates how **hierarchical sparsity, low-rank adaptation, low-bit representations, and adaptive retrieval** can be combined to reduce the computational and memory requirements of language models while preserving their reasoning capabilities.

The framework is evaluated across reasoning benchmarks including **GSM8K** and **ARC-Challenge**, with experiments spanning both model performance and hardware efficiency.

### ✨ Key idea

> **Can small language models become more efficient and robust without requiring large-scale computational resources?**

HOLA explores this question through a combination of:

* 🧩 **Hierarchical Sparsity**
* 🔧 **Parameter-efficient adaptation**
* 🧠 **Cross-domain representation learning**
* ⚡ **Low-bit inference**
* 🔎 **Adaptive retrieval**
* 💾 **Memory-aware optimization**
* 🖥️ **Edge-device benchmarking**

---

## 🏗️ HOLA Framework

<p align="center">
  <img src="docs/figures/hola_architecture.png" width="900">
</p>

**Figure 1.** Overview of the HOLA framework. The pipeline combines hierarchical sparsity, lightweight adaptation, low-bit representations, and adaptive retrieval to improve the efficiency and robustness of language models.

### 🔄 Pipeline

```text
                 ┌─────────────────────┐
                 │   Pre-trained LLM   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Lightweight         │
                 │ Instruction Tuning  │
                 └──────────┬──────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────────┐
        │   HSD    │  │   Lo-Bi  │  │ AdaComp-RAG  │
        │ Sparsity │  │ Low-Bit   │  │ Adaptive RAG │
        └─────┬────┘  └─────┬────┘  └──────┬───────┘
              │             │               │
              └─────────────┼───────────────┘
                            ▼
                 ┌─────────────────────┐
                 │  Efficient HOLA LLM │
                 └──────────┬──────────┘
                            │
              ┌─────────────┼──────────────┐
              ▼             ▼              ▼
         ┌─────────┐   ┌─────────┐   ┌───────────┐
         │ GSM8K   │   │   ARC   │   │ Edge HW   │
         └─────────┘   └─────────┘   └───────────┘
```

---

# 🚀 Highlights

| Capability                 | Description                                                        |
| -------------------------- | ------------------------------------------------------------------ |
| 🧠 Hierarchical Sparsity   | Introduces structured sparsity into lightweight adaptation modules |
| 🔧 Parameter Efficiency    | Reduces the number of trainable parameters                         |
| 🌐 Cross-Domain Robustness | Evaluates transfer between mathematical and science reasoning      |
| ⚡ Low-Bit Inference        | Reduces model memory requirements                                  |
| 🔎 Adaptive Retrieval      | Uses retrieval selectively to improve efficiency                   |
| 💾 Memory Awareness        | Measures memory consumption during training/inference              |
| ⏱️ Latency Analysis        | Benchmarks inference latency across hardware                       |
| 🖥️ Edge Deployment        | Designed for constrained platforms                                 |

---

# 🔬 Research Contributions

The repository investigates four complementary components.

### 1. Hierarchical Sparsity

HOLA introduces structured sparsity into lightweight adaptation modules to reduce redundant computation while retaining important representations.

### 2. HSD

The **Hierarchical Selective Distillation (HSD)** component transfers useful intermediate representations across domains and improves generalization.

### 3. Lo-Bi

**Lo-Bi** introduces low-bit representations to reduce model memory footprint and make inference more suitable for resource-constrained hardware.

### 4. AdaComp-RAG

**AdaComp-RAG** combines adaptive compression with retrieval-augmented generation to balance reasoning quality, memory usage, and inference latency.

---

# 📊 Benchmarks

HOLA is evaluated on two reasoning benchmarks.

## GSM8K

**GSM8K** contains grade-school mathematical word problems requiring multi-step reasoning.

**Metric:** Exact Match Accuracy (EMA)

## ARC-Challenge

**ARC-Challenge** contains challenging grade-school science questions designed to evaluate reasoning and knowledge.

**Metric:** Multiple Choice Accuracy (MCA)

---

# 📚 Datasets

| Dataset           | Task                   | Metric                   | Hugging Face                                   |
| ----------------- | ---------------------- | ------------------------ | ---------------------------------------------- |
| **GSM8K**         | Mathematical reasoning | Exact Match Accuracy     | [GSM8K](https://huggingface.co/datasets/gsm8k) |
| **ARC-Challenge** | Science reasoning      | Multiple Choice Accuracy | [ARC](https://huggingface.co/datasets/ai2_arc) |

### Loading the datasets

```python
from datasets import load_dataset

# GSM8K
gsm8k = load_dataset("gsm8k", "main")

# ARC-Challenge
arc = load_dataset("ai2_arc", "ARC-Challenge")
```

---

# 🤖 Models

The experiments consider language models spanning different parameter scales and architectures.

| Model               | Approx. Scale | Purpose                   |
| ------------------- | ------------: | ------------------------- |
| GPT-2               |         124M+ | Lightweight baseline      |
| TinyLlama           |          1.1B | Small-scale LLM           |
| Phi-1.5             |          1.3B | Efficient reasoning model |
| Gemma-2B            |            2B | Compact instruction model |
| LLaMA-family models |           3B+ | Mid-scale evaluation      |
| Phi-3.5 Mini        |         ~3.8B | Efficient reasoning       |
| Mistral-7B          |            7B | High-capacity comparison  |

> **Note:** Model identifiers and checkpoints should be kept consistent with the exact experimental configuration reported in the accompanying paper.

---

# 🏆 Results

## Overall Performance

<p align="center">
  <img src="docs/figures/performance_comparison.png" width="850">
</p>

**Figure 2.** Comparison of baseline models and HOLA-enhanced models on GSM8K and ARC-Challenge.

According to the reported experiments, HOLA provides substantial gains for lightweight models while maintaining competitive performance at larger model scales.

### Reported results

* **GPT-2**

  * GSM8K: **+15.6%** Exact Match Accuracy
  * ARC: **+14.3%** Multiple Choice Accuracy

* **Mistral-7B**

  * GSM8K: **83.4% EMA**
  * ARC: **66.9% MCA**

---

# 🌐 Cross-Domain Generalization

One of the main goals of HOLA is to investigate whether representations learned for one reasoning domain can transfer effectively to another.

<p align="center">
  <img src="docs/figures/cross_domain_heatmap.png" width="800">
</p>

**Figure 3.** Cross-domain transfer performance between mathematical and science reasoning tasks.

Reported transfer results include:

| Transfer Direction | Reported Performance |
| ------------------ | -------------------: |
| ARC → GSM8K        |            **68.5%** |
| GSM8K → ARC        |            **78.7%** |

These experiments are intended to measure whether HOLA can preserve useful representations when the target domain differs from the training domain.

---

# 🧪 Ablation Study

To understand the contribution of individual components, HOLA is evaluated with selected components removed.

<p align="center">
  <img src="docs/figures/ablation_study.png" width="850">
</p>

**Figure 4.** Ablation analysis of HOLA components.

One reported experiment shows:

```text
Full HOLA        → 89.2% EMA
Without HSD      → 85.1% EMA
```

This suggests that the HSD component contributes meaningfully to the overall performance of the framework.

Additional ablations investigate the effect of removing:

* HSD
* Lo-Bi
* AdaComp-RAG
* Hierarchical sparsity

---

# 💻 Hardware-Aware Evaluation

HOLA is designed with deployment constraints in mind.

The repository includes experiments targeting:

| Hardware        | Deployment Scenario          |
| --------------- | ---------------------------- |
| 🟢 Jetson Nano  | Edge AI                      |
| 🟢 Raspberry Pi | Highly constrained inference |
| 🔵 Intel i7     | Consumer CPU                 |
| 🟣 NVIDIA A100  | High-performance reference   |

---

## ⚡ Efficiency Results

<p align="center">
  <img src="docs/figures/hardware_efficiency.png" width="850">
</p>

**Figure 5.** Memory and latency comparison across hardware platforms.

Reported experiments demonstrate:

* Up to **~800 MB memory savings** on constrained hardware
* Approximately **50 ms latency reduction** in selected configurations

> Hardware results can vary depending on batch size, quantization configuration, framework version, and measurement methodology.

---

# 🧠 Representation Analysis

HOLA's representation behavior is also investigated using dimensionality-reduction techniques.

<p align="center">
  <img src="docs/figures/tsne_domains.png" width="800">
</p>

**Figure 6.** t-SNE visualization of representations from different reasoning domains.

The visualization provides qualitative evidence of domain structure within the learned representations and helps analyze how HOLA separates or aligns representations across tasks.

---

# 📈 Visualizations

The repository provides scripts for generating:

* 📊 Model performance comparisons
* 🔥 Cross-domain transfer heatmaps
* 📉 Ablation plots
* 📈 Ranking-shift plots
* 🧠 t-SNE representations
* 💾 Memory usage comparisons
* ⏱️ Latency comparisons
* 🖥️ Hardware efficiency plots

Example:

<p align="center">
  <img src="docs/figures/ranking_shift.png" width="800">
</p>

---

# 📦 Requirements

The main dependencies include:

```text
Python >= 3.9
PyTorch
Transformers
Datasets
PEFT
Accelerate
Scikit-learn
NumPy
Pandas
Matplotlib
Seaborn
```







# 🌍 Why HOLA?

Large language models can achieve impressive reasoning performance, but deploying them outside large data centers remains challenging because of:

* High memory requirements
* Expensive inference
* Large model footprints
* Limited edge-device resources
* Increasing latency on constrained hardware

HOLA investigates whether **efficient adaptation + structured sparsity + low-bit inference + adaptive retrieval** can provide a practical alternative.

### The goal is simple:

```text
                    Large LLM
                       │
                       │
             High compute / memory
                       │
                       ▼
              ┌─────────────────┐
              │    HOLA         │
              │                 │
              │ Sparse          │
              │ Low-bit         │
              │ Adaptive        │
              │ Retrieval       │
              └────────┬────────┘
                       │
                       ▼
             Efficient Small LLM
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Accuracy      Memory       Latency
```

