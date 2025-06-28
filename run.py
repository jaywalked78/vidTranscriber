import uvicorn
import os
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Transcriber API Server")
    parser.add_argument(
        "--host", 
        type=str, 
        default=os.getenv("HOST", "0.0.0.0"), 
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=int(os.getenv("PORT", 8000)), 
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=1, 
        help="Number of worker processes"
    )
    
    args = parser.parse_args()
    
    print(f"Starting Video Transcriber API on {args.host}:{args.port}")
    print(f"GPU Device: {os.getenv('CUDA_VISIBLE_DEVICES', '0')}")
    print(f"Compute Type: {os.getenv('COMPUTE_TYPE', 'float16')}")
    print(f"Model: {os.getenv('MODEL_NAME', 'turbo')}")
    print(f"Workers: {args.workers}")
    
    uvicorn.run(
        "app.main:app", 
        host=args.host, 
        port=args.port, 
        reload=args.reload,
        workers=args.workers
    ) 