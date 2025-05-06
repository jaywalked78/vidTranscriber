#!/usr/bin/env python3
"""
vidTranscriber server launcher script.
This script starts the vidTranscriber API service.
"""

import os
import sys
import signal
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

# Setup proper signal handling for cleaner shutdown
if not RELOAD:  # Custom signal handling breaks reload mode
    # Define a cleaner shutdown handler
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    original_sigterm_handler = signal.getsignal(signal.SIGTERM)
    
    def signal_handler(sig, frame):
        print("\n🛑 Shutdown signal received. Gracefully shutting down...")
        # Call original handlers if they're callable
        if callable(original_sigint_handler) and sig == signal.SIGINT:
            original_sigint_handler(sig, frame)
        if callable(original_sigterm_handler) and sig == signal.SIGTERM:
            original_sigterm_handler(sig, frame)
        sys.exit(0)
    
    # Set up signal handling before starting server
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# Get CUDA device info
try:
    import torch
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "None"
        print(f"GPU Device: {0} ({device_name})")
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