# This is for running locally without docker or on unsupported platforms for docker (e.g. macOS)
# Download binary here: https://github.com/ggml-org/llama.cpp/releases, then unzip to match directory structure

MODEL_FILENAME="${MODEL_FILENAME:-Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
echo "Running with model at ./${MODEL_FILENAME}"

./llama-server --model ../../models/$MODEL_FILENAME --ctx-size 16384 --parallel 4 --n-gpu-layers 33
