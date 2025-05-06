#!/usr/bin/env python3
"""
vidTranscriber server launcher script.
This script starts the vidTranscriber API service.
"""

import os
import sys
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get configuration from environment
PORT = int(os.getenv("PORT", 8509))
HOST = os.getenv("HOST", "0.0.0.0")
RELOAD = os.getenv("RELOAD", "false").lower() == "true"
WORKERS = int(os.getenv("WORKERS", 1))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

# Get CUDA device info
try:
    import torch
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "None"
        print(f"GPU Device: {0}")
    else:
        print("CUDA not available, using CPU")
except ImportError:
    print("PyTorch not installed, unable to check CUDA availability")

# Print startup information
print(f"Starting Video Transcriber API on {HOST}:{PORT}")
model_name = os.getenv("MODEL_NAME", "turbo")
compute_type = os.getenv("COMPUTE_TYPE", "float16")
print(f"Compute Type: {compute_type}")
print(f"Model: {model_name}")
print(f"Workers: {WORKERS}")

# Start the server
if __name__ == "__main__":
    try:
        uvicorn.run(
            "app.main:app",
            host=HOST,
            port=PORT,
            reload=RELOAD,
            workers=WORKERS,
            log_level=LOG_LEVEL
        )
    except KeyboardInterrupt:
        print("\nShutting down Video Transcriber API...")
        sys.exit(0) 