#!/bin/sh

# Download hugging face model for llama.cpp to serve
# Needs hf_transfer installed:
#  pip install "huggingface_hub[hf_transfer]"

set -eu

# Suggested models:

# Qwen 13B finetuned with DeepSeek R1
# MODEL_REPO=bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF
# MODEL_FILENAME=DeepSeek-R1-Distill-Qwen-14B-Q5_K_M.gguf

# 3.1 8B
# MODEL_REPO="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
# MODEL_FILENAME="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# 3.2 3B
# MODEL_REPO="bartowski/Llama-3.2-3B-Instruct-GGUF"
# MODEL_FILENAME="Llama-3.2-3B-Instruct-Q4_K_M.gguf"


export MODEL_REPO="${MODEL_REPO:-bartowski/Meta-Llama-3.1-8B-Instruct-GGUF}"
export MODEL_FILENAME="${MODEL_FILENAME:-Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"

echo "Downloading model:"
echo $MODEL_REPO
echo $MODEL_FILENAME

HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download $MODEL_REPO $MODEL_FILENAME --local-dir ./models
