# This is for running with docker, which may be easier for CUDA-enabled environments
# Download binaries here: https://github.com/ggml-org/llama.cpp/releases

MODEL_FILENAME="${MODEL_FILENAME:-Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
echo "Running with model at ./${MODEL_FILENAME}"

docker run --gpus all -v ./models:/models -p 8080:8080 ghcr.io/ggml-org/llama.cpp:server -m /models/$MODEL_FILENAME --host 0.0.0.0 --port 8080 --ctx-size 16384 --parallel 8 --n-gpu-layers 20
