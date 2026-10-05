# Amharic LLM LoRA Fine-Tuning

A reproducible notebook for parameter-efficient fine-tuning of a causal language model on an Amharic sentence corpus using **LoRA/PEFT**.

## Overview

This project fine-tunes `Qwen/Qwen3-0.6B` with LoRA on the Hugging Face **Amharic Sentences Corpus V1.0**:

- **Dataset:** `a3xrfgb/amharic-sentences-corpus`
- **Base model:** `Qwen/Qwen3-0.6B`
- **Training objective:** causal language-model training / text continuation
- **Parameter-efficient method:** LoRA via PEFT and TRL
- **Optional low-memory mode:** 4-bit QLoRA-style loading with bitsandbytes
- **Default sequence length:** 1024 tokens
- **Default epochs:** 1
- **LoRA rank:** 16
- **LoRA alpha:** 32
- **LoRA dropout:** 0.05

> **Important:** The notebook trains on raw sentences. It is continued language-model training, not instruction tuning. It does not by itself create a question-answering/chat model.

## Repository structure

```text
amharic-llm-lora-finetuning/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── notebooks/
│   └── amharic_llm_lora_finetuning.ipynb
├── results/
│   └── README.md
└── images/
    └── README.md
```

## Open in Google Colab

After pushing this repository to GitHub, replace `YOUR_USERNAME` below with your GitHub username:

```text
https://colab.research.google.com/github/YOUR_USERNAME/amharic-llm-lora-finetuning/blob/main/notebooks/amharic_llm_lora_finetuning.ipynb
```

You can place this badge in the repository README:

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/amharic-llm-lora-finetuning/blob/main/notebooks/amharic_llm_lora_finetuning.ipynb)
```

## Installation

Python 3.x with a CUDA-capable GPU is recommended for training.

```bash
pip install -r requirements.txt
```

For 4-bit loading, install `bitsandbytes` in a compatible CUDA environment:

```bash
pip install -U bitsandbytes
```

The notebook also installs/upgrades its Hugging Face dependencies directly.

## Quick start

1. Open `notebooks/amharic_llm_lora_finetuning.ipynb`.
2. Run the installation and import cells.
3. Start with a smoke test by setting:

```python
MAX_TRAIN_SAMPLES = 10_000
MAX_EVAL_SAMPLES = 1_000
```

4. Confirm that loading, preprocessing, tokenization, model creation, trainer creation, and generation work.
5. For a full experiment, set:

```python
MAX_TRAIN_SAMPLES = None
```

6. Run training.
7. Evaluate the model using held-out loss/perplexity.
8. Save the LoRA adapter to:

```text
./amharic-qwen3-0.6b-lora/adapter
```

## Main configuration

The notebook exposes the principal experiment settings in one configuration cell:

```python
MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_LENGTH = 1024
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8

USE_4BIT = False
```

For limited GPU memory, the notebook recommends reducing the per-device batch size, reducing sequence length, increasing gradient accumulation, or enabling 4-bit loading.

## Data processing

The notebook:

1. Loads the training split from the Hugging Face Hub.
2. Automatically detects a suitable text column.
3. Normalizes whitespace and removes null characters.
4. Removes empty/invalid examples.
5. Optionally removes exact duplicate sentences.
6. Creates a reproducible train/evaluation split using seed `42`.
7. Normalizes the selected field to a standard `text` column.

The deduplication implementation uses the Hugging Face `Dataset.unique()` API rather than the pandas-style `drop_duplicates()` method.

## LoRA configuration

The default LoRA configuration targets:

```text
q_proj
k_proj
v_proj
o_proj
```

with:

```text
r = 16
alpha = 32
dropout = 0.05
bias = none
task = CAUSAL_LM
```

The base model is frozen while the LoRA adapter parameters are trained.

## Evaluation

The notebook evaluates the held-out set using language-model evaluation loss and derives perplexity as:

```python
perplexity = exp(eval_loss)
```

For a research-grade evaluation, perplexity should not be the only metric. A separate held-out Amharic benchmark should be developed for the intended downstream tasks.

## Generation

The notebook includes Amharic prompts for qualitative text generation and saves the LoRA adapter for later reloading.

Example prompts included in the notebook:

```text
ኢትዮጵያ በአፍሪካ
ዛሬ በአዲስ አበባ
የአማርኛ ቋንቋ
```

Generated text should be reported as an experimental output rather than as a guaranteed quality benchmark.

## Outputs

The notebook saves the trained adapter and tokenizer under:

```text
./amharic-qwen3-0.6b-lora/adapter
```

The repository intentionally does **not** include trained model weights or checkpoints. These can be large and should normally be stored in an appropriate model repository or release artifact.

## Reproducibility

The default seed is:

```python
SEED = 42
```

The final notebook also prints the main configuration so that an experiment can be documented and reproduced.

## Hugging Face resources

- Dataset: https://huggingface.co/datasets/a3xrfgb/amharic-sentences-corpus
- Qwen3-0.6B: https://huggingface.co/Qwen/Qwen3-0.6B
- Qwen3-0.6B-Base: https://huggingface.co/Qwen/Qwen3-0.6B-Base
- TRL SFTTrainer: https://huggingface.co/docs/trl/sft_trainer
- TRL PEFT integration: https://huggingface.co/docs/trl/peft_integration

## Current results

No numerical training results are claimed in this repository until an experiment is actually run and its evaluation outputs are recorded.

Recommended reporting fields:

| Metric | Value |
|---|---:|
| Base model | Qwen/Qwen3-0.6B |
| Dataset | Amharic Sentences Corpus V1.0 |
| Training samples | To be reported |
| Evaluation samples | To be reported |
| Epochs | 1 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Sequence length | 1024 |
| Evaluation loss | To be reported |
| Perplexity | To be reported |
| GPU | To be reported |
| Training time | To be reported |

## Citation

If you use this repository in academic work, cite the repository and the underlying dataset/model resources used in your experiment.

## License

This repository is released under the MIT License. The dataset and base model remain subject to their respective licenses and terms; users should review those licenses before redistribution or commercial use.
