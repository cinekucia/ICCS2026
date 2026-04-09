# LLM Essay Scoring Under Holistic and Analytic Rubrics

[![GitHub](https://img.shields.io/badge/GitHub-cinekucia%2FICCS2026-black)](https://github.com/cinekucia/ICCS2026)

This repository contains the code and materials for our **ICCS 2026 conference paper** on **automated essay scoring with large language models (LLMs)** across **holistic** and **analytic** scoring settings.

## Overview

This project investigates how instruction-tuned LLMs perform as essay scorers under different evaluation regimes. The repository focuses on:

- **holistic** versus **analytic** essay scoring
- comparisons across multiple **instruction-tuned open-weight LLMs**
- the effect of different **prompting strategies**

## Datasets

The experiments use three essay-scoring datasets:

- **ASAP 2.0** — holistic essay scoring
- **ELLIPSE** — analytic scoring with multiple traits
- **DREsS** — analytic rubric-based scoring

Together, these datasets make it possible to compare overall essay-quality scoring with fine-grained trait-level evaluation.

## Evaluated LLMs

The repository covers the following instruction-tuned open-weight LLMs:

- **Meta Llama-3.1-8B-Instruct**
- **Meta Llama-3.1-70B-Instruct**
- **Meta Llama-3.1-405B-Instruct**
- **GPT-OSS-20B**
- **GPT-OSS-120B**

## Prompt Files

The prompt templates used in the experiments are available in:

- `guidelines_prompt.txt`
- `keywords_system_prompt.txt`

These correspond to two main prompting strategies:

- **Guidelines prompts** — detailed rubric descriptions and scoring instructions
- **Keywords prompts** — shorter trait-focused prompts

## Repository Link

Repository: **[github.com/cinekucia/ICCS2026](https://github.com/cinekucia/ICCS2026)**
