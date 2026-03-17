#!/bin/bash
# scripts/start_vllm.sh
# Starts the vLLM server on the H200 GPU with Qwen 2.5 Instruct.

echo "🚀 Starting vLLM Server on H200..."

# Command provided by user for H200 deployment
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192
