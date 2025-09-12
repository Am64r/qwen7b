# Mathematical Reasoning LLM Fine-tuning

Fine-tuning large language models on mathematical reasoning problems using parameter-efficient techniques and memory optimization.

## Project Overview

This project fine-tunes transformer models on 87K+ mathematical problems from AIME, AIMO, and DeepScaleR datasets. The focus is on memory-efficient training using LoRA adapters and gradient checkpointing techniques.

## Key Results

- 60% reduction in GPU memory usage through LoRA adapters and gradient checkpointing
- 2x training speedup using Unsloth optimization framework  
- Successful validation loss convergence across 3 training epochs
- Models deployed to HuggingFace Hub for production use

## Models Fine-tuned

- DeepSeek-R1-Distill-Qwen-7B
- Qwen2.5-Math-7B

Published models:
- theamrelhady/finetuned-math-deepseek-r1-distill-qwen-7b-v3
- theamrelhady/finetuned-qwen-2.5-math-7b-v2

## Technical Approach

Parameter-Efficient Fine-Tuning using LoRA adapters with rank-8 configuration. Training used supervised fine-tuning with BFloat16 precision, AdamW optimizer, and batch size of 64 across H100 hardware.

Memory optimization through gradient checkpointing and 4-bit quantization during inference.

## Files

- `ds-r1-distill-qwen-7b.ipynb` - Fine-tuning notebook for DeepSeek-R1-Distill model with evaluation on AIME problems
- `ds-ri-distill-qwen-7b-v2.ipynb` - Fine-tuning notebook for Qwen2.5-Math model with combined dataset training

## Requirements

Core libraries: PyTorch, Hugging Face Transformers, Unsloth, PEFT, TRL, datasets, accelerate

Hardware: High-memory GPU (H100 recommended), 32GB+ system RAM
