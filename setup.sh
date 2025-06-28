#!/bin/bash

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv vidtranscribeVenv

# Activate virtual environment
echo "Activating virtual environment..."
source vidtranscribeVenv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env file from template if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit the .env file with your configuration"
fi

# Create temporary directory
echo "Creating temporary directory..."
mkdir -p /tmp/vidTranscriber

# Create output directory
echo "Creating output directory..."
mkdir -p /tmp/vidTranscriberOutput

# Run the GPU setup checker
echo "Checking GPU and CUDA compatibility..."
python check_gpu_setup.py

echo ""
echo "Setup complete! Run the server with:"
echo "source vidtranscribeVenv/bin/activate"
echo "cd vidTranscriber"
echo "python run.py"
echo ""
echo "Access the API documentation at http://localhost:8000/docs" 