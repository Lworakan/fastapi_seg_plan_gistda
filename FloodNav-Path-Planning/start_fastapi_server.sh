#!/bin/bash

# FastAPI Path Planning Server Startup Script

echo "Starting FastAPI Path Planning Server..."
echo "========================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements_fastapi.txt

# Start the server
echo "Starting FastAPI server with uvicorn..."
echo "Server will be available at: http://localhost:8000"
echo "API documentation at: http://localhost:8000/docs"
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn fastapi_path_planning:app --host 0.0.0.0 --port 8000 --reload