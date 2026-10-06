#!/bin/bash
echo "Starting Llama 3.1 8B AI Server..."
nohup /tmp/llama_bin/build/bin/llama-server -m "Source_Code/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" -c 4096 --port 8081 --host 127.0.0.1 > llama_server.log 2>&1 &

echo "Starting FastAPI Backend..."
nohup python3 Source_Code/nutrix_ai/api.py > backend.log 2>&1 &

echo "Starting Flutter Web Frontend..."
cd ../new_feature_project && nohup python3 server_no_cache.py > frontend.log 2>&1 &

echo "Pipeline started successfully!"
echo "AI Server:   http://localhost:8081"
echo "Backend API: http://localhost:8000"
echo "Frontend:    http://localhost:3000"
