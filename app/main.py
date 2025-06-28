from fastapi import FastAPI, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import os
import signal
import sys
import asyncio
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from app.api.routes import transcription
from app.utils.rate_limiter import RateLimiter
from app.core.transcriber import transcription_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create startup and shutdown handlers
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic - initialize resources
    logger.info("Starting up Video Transcription API")
    
    # Yield control to FastAPI
    yield
    
    # Shutdown logic - clean up resources
    logger.info("Shutting down Video Transcription API")
    
    # Cleanup will be handled by signal handlers in transcription_service

# Create FastAPI app with lifespan manager
app = FastAPI(
    title="Video Transcription API",
    description="Transcribe videos to text using faster-whisper and GPU acceleration",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create rate limiter
rate_limiter = RateLimiter(
    limit=int(os.getenv("REQUEST_LIMIT_PER_MINUTE", 10)),
    window=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", 60))
)

# Rate limiting middleware
@app.middleware("http")
async def rate_limiting_middleware(request: Request, call_next):
    client_ip = request.client.host
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again later."}
        )
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Include routers
app.include_router(transcription.router)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "device": transcription_service.device
    }

if __name__ == "__main__":
    import uvicorn
    
    # Configure Uvicorn to handle SIGTERM gracefully
    # (Signal handling for application itself is in transcription_service)
    uvicorn.run(
        "app.main:app", 
        host=os.getenv("HOST", "0.0.0.0"), 
        port=int(os.getenv("PORT", 8000)), 
        reload=False,  # Disable reload in production
        log_level=os.getenv("LOG_LEVEL", "info")
    ) 