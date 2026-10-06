#!/bin/bash
echo "Stopping Pipeline..."
pkill -f llama-server
pkill -f "api.py"
pkill -f "server_no_cache.py"
echo "Pipeline stopped successfully!"
