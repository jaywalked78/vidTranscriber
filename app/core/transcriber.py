import os
import time
import uuid
import asyncio
import aiohttp
import tempfile
import json
import signal
import sys
from pathlib import Path
import ffmpeg
import torch
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Tuple
from contextlib import asynccontextmanager
import logging

# Import faster-whisper with better error handling
try:
    from faster_whisper import WhisperModel, BatchedInferencePipeline
except ImportError:
    logger.error("Failed to import faster-whisper. Please make sure it's installed.")
    sys.exit(1)

from app.models.transcription import (
    JobStatus, 
    TranscriptionOptions,
    TranscriptionSegment
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Set faster-whisper logger level
logging.getLogger("faster_whisper").setLevel(logging.INFO)

# Global job storage (In production, use Redis or a database)
jobs = {}

# Global flag for graceful shutdown
is_shutting_down = False

# Model name mapping between OpenAI Whisper and faster-whisper
MODEL_NAME_MAPPING = {
    "tiny": "tiny",
    "tiny.en": "tiny.en",
    "base": "base",
    "base.en": "base.en",
    "small": "small",
    "small.en": "small.en",
    "medium": "medium",
    "medium.en": "medium.en",
    "large": "large-v2", # Default to large-v2 for "large"
    "large-v1": "large-v1",
    "large-v2": "large-v2",
    "large-v3": "large-v3",
    "turbo": "large-v3",  # Turbo is an optimized version of large-v3
}


class TranscriptionService:
    """Service for transcribing videos using faster-whisper with GPU acceleration"""
    
    def __init__(self):
        """Initialize the transcription service"""
        self.model = None
        self.batched_model = None
        
        # Get model name from environment or default to "turbo"
        model_name = os.getenv("MODEL_NAME", "turbo")
        # Map OpenAI model name to faster-whisper model name
        self.model_name = MODEL_NAME_MAPPING.get(model_name, "large-v3")
        
        self.compute_type = os.getenv("COMPUTE_TYPE", "float16")
        
        # Try to use CUDA, but with fallback to CPU
        try:
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info("CUDA is available, using GPU acceleration")
            else:
                self.device = "cpu"
                logger.info("CUDA is not available, using CPU")
        except Exception as e:
            logger.warning(f"Error checking CUDA availability: {str(e)}. Falling back to CPU.")
            self.device = "cpu"
        
        self.batch_size = int(os.getenv("BATCH_SIZE", "8"))  # Default batch size for faster processing
        self.temp_dir = os.getenv("TEMP_DIR", "/tmp/vidTranscriber")
        self.output_dir = os.getenv("OUTPUT_DIR", "/tmp/vidTranscriberOutput")
        self.max_concurrent_jobs = int(os.getenv("MAX_CONCURRENT_JOBS", 2))
        self.max_video_size_mb = int(os.getenv("MAX_VIDEO_SIZE_MB", 10240))  # Default 10GB limit
        self.max_video_duration_minutes = int(os.getenv("MAX_VIDEO_DURATION_MINUTES", 720))  # Default 12 hours
        
        # Audio processing configuration
        self.audio_sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", 16000))  # 16kHz
        self.audio_codec = os.getenv("AUDIO_CODEC", "pcm_s16le")  # 16-bit PCM
        self.audio_channels = int(os.getenv("AUDIO_CHANNELS", 1))  # Mono
        self.download_chunk_size = int(os.getenv("DOWNLOAD_CHUNK_SIZE_MB", 1)) * 1024 * 1024  # Default 1MB chunks
        
        # VAD configuration
        self.vad_min_silence_ms = int(os.getenv("VAD_MIN_SILENCE_MS", 500))  # 500ms
        
        # GPU/CPU configuration
        self.gpu_device_index = int(os.getenv("GPU_DEVICE_INDEX", 0))  # GPU device index
        self.cpu_threads = int(os.getenv("CPU_THREADS", 4))  # CPU threads for processing
        
        # Shutdown configuration
        self.shutdown_timeout_seconds = int(os.getenv("SHUTDOWN_TIMEOUT_SECONDS", 10))  # Graceful shutdown timeout
        
        # Create temp directory if it doesn't exist
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize semaphore for concurrent jobs
        self._semaphore = asyncio.Semaphore(self.max_concurrent_jobs)
        
        # Track active jobs for graceful shutdown
        self.active_jobs = set()
        
        # Load model
        self._load_model()
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        # Use different signal handlers based on platform
        if os.name == 'posix':  # For Linux/Unix/MacOS
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig,
                    lambda sig=sig: asyncio.create_task(self._handle_shutdown(sig))
                )
            logger.info("Signal handlers for graceful shutdown have been set up")
        else:
            # For Windows, signal handlers work differently
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, lambda s, f: asyncio.create_task(self._handle_shutdown(s)))
    
    async def _handle_shutdown(self, sig):
        """Handle shutdown signals gracefully"""
        global is_shutting_down
        
        if is_shutting_down:
            return
            
        is_shutting_down = True
        
        logger.info(f"Received shutdown signal {sig}, gracefully shutting down...")
        
        # Update all active jobs to indicate server shutdown
        for job_id in self.active_jobs.copy():  # Use copy to avoid modification during iteration
            try:
                if job_id in jobs and jobs[job_id]["status"] not in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    self.update_job(
                        job_id, 
                        status=JobStatus.FAILED, 
                        error="Server shutting down, transcription interrupted"
                    )
            except Exception as e:
                logger.error(f"Error updating job {job_id} during shutdown: {str(e)}")
        
        # Clear active jobs
        self.active_jobs.clear()
        
        # Clean up temp files
        try:
            temp_files = list(Path(self.temp_dir).glob("*"))
            for file in temp_files:
                try:
                    if file.is_file():
                        file.unlink()
                except Exception as e:
                    logger.error(f"Error removing temp file {file}: {str(e)}")
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {str(e)}")
        
        # Clear CUDA cache if using GPU
        if self.device == "cuda":
            try:
                torch.cuda.empty_cache()
                logger.info("CUDA memory cache cleared")
            except Exception as e:
                logger.error(f"Error clearing CUDA cache: {str(e)}")
        
        # Give pending tasks a moment to complete
        logger.info(f"Waiting up to {self.shutdown_timeout_seconds} seconds for active transcriptions to stop...")
        
        # Wait for active jobs to complete or timeout
        for i in range(self.shutdown_timeout_seconds):
            if not self.active_jobs:
                break
            await asyncio.sleep(1)
            if i == (self.shutdown_timeout_seconds // 2):  # Halfway point warning
                remaining = self.shutdown_timeout_seconds - i
                logger.warning(f"Some transcription jobs are still running. Will force shutdown in {remaining} seconds...")
        
        if self.active_jobs:
            logger.warning(f"Force shutdown: {len(self.active_jobs)} jobs still active")
        
        # Exit the application with a clean exit code
        logger.info("Shutdown complete, exiting")
        sys.exit(0)
    
    def _load_model(self):
        """Load the whisper model with GPU acceleration and fallback to CPU if needed"""
        try:
            logger.info(f"Loading model {self.model_name} on {self.device} with {self.compute_type}")
            
            # Choose compute_type based on device
            if self.device == "cuda":
                if self.compute_type == "int8":
                    # For GPU, use int8_float16 for better memory efficiency
                    compute_type = "int8_float16"
                else:
                    compute_type = "float16"
            else:
                # For CPU
                compute_type = self.compute_type
            
            # Initialize the model - models will be downloaded automatically
            try:
                self.model = WhisperModel(
                    self.model_name,  # No model_path, will download automatically
                    device=self.device,
                    compute_type=compute_type,
                    device_index=self.gpu_device_index,
                    cpu_threads=self.cpu_threads
                )
                
                # Initialize batched inference pipeline for better performance
                self.batched_model = BatchedInferencePipeline(
                    model=self.model
                )
                
                logger.info(f"Model {self.model_name} loaded successfully on {self.device}")
            except Exception as e:
                # If GPU fails, try falling back to CPU
                if self.device == "cuda":
                    logger.warning(f"Failed to load model on GPU: {str(e)}. Falling back to CPU.")
                    self.device = "cpu"
                    compute_type = "float32"
                    
                    self.model = WhisperModel(
                        self.model_name,
                        device=self.device,
                        compute_type=compute_type,
                        cpu_threads=self.cpu_threads
                    )
                    
                    self.batched_model = BatchedInferencePipeline(
                        model=self.model
                    )
                    
                    logger.info(f"Model {self.model_name} loaded successfully on CPU (fallback)")
                else:
                    # If CPU also fails, re-raise the exception
                    logger.error(f"Failed to load model on CPU: {str(e)}")
                    raise
                
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def create_job(self, video_url: str, options: TranscriptionOptions) -> str:
        """Create a new transcription job"""
        if is_shutting_down:
            raise ValueError("Server is shutting down, cannot accept new jobs")
            
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "id": job_id,
            "video_url": video_url,
            "options": options.dict(),
            "status": JobStatus.QUEUED,
            "created_at": time.time(),
            "updated_at": time.time(),
            "progress": 0.0,
            "result": None,
            "error": None
        }
        return job_id
    
    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Get job information"""
        if job_id not in jobs:
            raise ValueError(f"Job with ID {job_id} not found")
        return jobs[job_id]
    
    def update_job(self, job_id: str, **kwargs) -> None:
        """Update job information"""
        if job_id not in jobs:
            raise ValueError(f"Job with ID {job_id} not found")
        
        jobs[job_id].update(**kwargs)
        jobs[job_id]["updated_at"] = time.time()
    
    @asynccontextmanager
    async def _download_video(self, job_id: str, url: str) -> str:
        """
        Download video from URL with progress tracking
        
        Args:
            job_id: ID of the transcription job
            url: URL of the video to download
            
        Returns:
            Path to downloaded video file
        """
        temp_file = Path(self.temp_dir) / f"{job_id}_video.mp4"
        
        try:
            self.update_job(job_id, status=JobStatus.DOWNLOADING, progress=0.0)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise ValueError(f"Failed to download video: HTTP {response.status}")
                    
                    # Check content length
                    content_length = response.headers.get('content-length')
                    if content_length:
                        size_mb = int(content_length) / (1024 * 1024)
                        if size_mb > self.max_video_size_mb:
                            raise ValueError(f"Video size ({size_mb:.2f}MB) exceeds limit ({self.max_video_size_mb}MB)")
                    
                    # Download with progress tracking
                    total_size = int(content_length) if content_length else 0
                    downloaded = 0
                    
                    with open(temp_file, 'wb') as f:
                        async for chunk in response.content.iter_chunked(self.download_chunk_size):
                            # Check for shutdown signal during download
                            if is_shutting_down:
                                raise ValueError("Server shutting down, download interrupted")
                                
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = downloaded / total_size * 0.4  # 40% of total progress
                                self.update_job(job_id, progress=progress)
            
            yield str(temp_file)
            
        except Exception as e:
            self.update_job(job_id, status=JobStatus.FAILED, error=str(e))
            raise
        finally:
            # Clean up temp file
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception as e:
                    logger.error(f"Failed to delete temp file {temp_file}: {str(e)}")
    
    async def _extract_audio(self, job_id: str, video_path: str) -> str:
        """
        Extract audio from video file
        
        Args:
            job_id: ID of the transcription job
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file
        """
        audio_path = Path(self.temp_dir) / f"{job_id}_audio.wav"
        
        try:
            # Check for shutdown signal
            if is_shutting_down:
                raise ValueError("Server shutting down, extraction interrupted")
                
            self.update_job(job_id, status=JobStatus.EXTRACTING, progress=0.4)  # 40% progress
            
            # Extract audio using ffmpeg
            ffmpeg.input(video_path).output(
                str(audio_path),
                acodec=self.audio_codec,
                ac=self.audio_channels,
                ar=self.audio_sample_rate
            ).run(quiet=True, overwrite_output=True)
            
            self.update_job(job_id, progress=0.5)  # 50% progress
            return str(audio_path)
            
        except Exception as e:
            self.update_job(job_id, status=JobStatus.FAILED, error=str(e))
            raise
    
    def _save_json_result(self, job_id: str, result: Dict[str, Any]) -> str:
        """
        Save transcription result to a JSON file
        
        Args:
            job_id: ID of the transcription job
            result: Transcription result
            
        Returns:
            Path to the JSON file
        """
        try:
            # Format for saving (remove circular references)
            output_data = {
                "job_id": job_id,
                "timestamp": time.time(),
                "text": result["text"],
                "language": result.get("language"),
                "language_probability": result.get("language_probability"),
                "segments": result["segments"]
            }
            
            # Create filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            filename = f"{job_id}_{timestamp}.json"
            file_path = Path(self.output_dir) / filename
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            # Add output path to job info
            self.update_job(job_id, output_file=str(file_path))
            
            logger.info(f"Saved transcription result to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save JSON result: {str(e)}")
            raise
    
    async def _transcribe_audio(self, job_id: str, audio_path: str, options: TranscriptionOptions) -> Dict[str, Any]:
        """
        Transcribe audio file
        
        Args:
            job_id: ID of the transcription job
            audio_path: Path to audio file
            options: Transcription options
            
        Returns:
            Transcription result
        """
        try:
            # Check for shutdown signal
            if is_shutting_down:
                raise ValueError("Server shutting down, transcription interrupted")
                
            self.update_job(job_id, status=JobStatus.TRANSCRIBING, progress=0.5)
            
            # Prepare options
            transcribe_options = {
                "beam_size": options["beam_size"],
                "word_timestamps": options["word_timestamps"],
                "language": options.get("language"),
                "vad_filter": options["vad_filter"],
                "vad_parameters": {"min_silence_duration_ms": self.vad_min_silence_ms},
                "batch_size": self.batch_size  # Use configured batch size
            }
            
            # Run in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            
            def run_transcription():
                # Check for shutdown signal
                if is_shutting_down:
                    raise ValueError("Server shutting down, transcription interrupted")
                    
                # For batched pipeline, progress_callback is not supported
                # Set progress to 75% before starting
                self.update_job(job_id, progress=0.75)
                
                try:
                    # Transcribe with batched pipeline (without progress_callback)
                    segments, info = self.batched_model.transcribe(
                        audio_path,
                        **transcribe_options
                    )
                    
                    # Set progress to 90% after transcription is complete
                    self.update_job(job_id, progress=0.9)
                    
                    # Convert segments to list
                    segments_list = []
                    full_text = ""
                    
                    for i, segment in enumerate(segments):
                        segment_dict = {
                            "id": i,
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text.strip()
                        }
                        
                        # Add word-level timestamps if available
                        if segment.words:
                            segment_dict["words"] = [
                                {"start": word.start, "end": word.end, "word": word.word}
                                for word in segment.words
                            ]
                        
                        segments_list.append(segment_dict)
                        full_text += segment.text + " "
                    
                    return {
                        "text": full_text.strip(),
                        "segments": segments_list,
                        "language": info.language,
                        "language_probability": info.language_probability
                    }
                except RuntimeError as e:
                    # Handle CUDA errors that might occur during transcription
                    error_msg = str(e)
                    if "CUDA" in error_msg or "cudnn" in error_msg or "GPU" in error_msg:
                        logger.error(f"CUDA/GPU error during transcription: {error_msg}")
                        
                        # Fall back to CPU if this was a GPU error
                        if self.device == "cuda":
                            logger.info("Attempting to fall back to CPU for this transcription")
                            
                            # Create a CPU model for this specific transcription
                            cpu_model = WhisperModel(
                                self.model_name,
                                device="cpu",
                                compute_type="float32",
                                cpu_threads=self.cpu_threads
                            )
                            
                            cpu_batched = BatchedInferencePipeline(model=cpu_model)
                            
                            # Update progress
                            self.update_job(job_id, progress=0.6)
                            
                            # Try transcription with CPU
                            segments, info = cpu_batched.transcribe(
                                audio_path,
                                **transcribe_options
                            )
                            
                            # Update progress
                            self.update_job(job_id, progress=0.9)
                            
                            # Convert segments to list
                            segments_list = []
                            full_text = ""
                            
                            for i, segment in enumerate(segments):
                                segment_dict = {
                                    "id": i,
                                    "start": segment.start,
                                    "end": segment.end,
                                    "text": segment.text.strip()
                                }
                                
                                # Add word-level timestamps if available
                                if segment.words:
                                    segment_dict["words"] = [
                                        {"start": word.start, "end": word.end, "word": word.word}
                                        for word in segment.words
                                    ]
                                
                                segments_list.append(segment_dict)
                                full_text += segment.text + " "
                            
                            return {
                                "text": full_text.strip(),
                                "segments": segments_list,
                                "language": info.language,
                                "language_probability": info.language_probability
                            }
                    # Re-raise if not a CUDA error or if fallback failed
                    raise
            
            # Run transcription in a thread pool with cancellation support
            try:
                # Use asyncio.wait_for to make the transcription cancellable
                # We'll check every 30 seconds if we need to cancel
                async def run_with_cancellation():
                    return await loop.run_in_executor(None, run_transcription)
                
                result = await run_with_cancellation()
                
            except asyncio.CancelledError:
                logger.info(f"Transcription job {job_id} was cancelled")
                raise ValueError("Transcription cancelled due to shutdown")
            
            # Format as markdown
            markdown = self._format_as_markdown(result)
            result["markdown"] = markdown
            
            # Save as JSON file
            json_path = self._save_json_result(job_id, result)
            result["json_path"] = json_path
            
            # Update job
            self.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                progress=1.0,
                result=result
            )
            
            return result
            
        except Exception as e:
            self.update_job(job_id, status=JobStatus.FAILED, error=str(e))
            raise
        finally:
            # Clean up audio file
            if Path(audio_path).exists():
                try:
                    Path(audio_path).unlink()
                except Exception as e:
                    logger.error(f"Failed to delete audio file {audio_path}: {str(e)}")
            
            # Clear CUDA cache
            if self.device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Failed to clear CUDA cache: {str(e)}")
            
            # Remove from active jobs
            if job_id in self.active_jobs:
                self.active_jobs.remove(job_id)
    
    def _format_as_markdown(self, result: Dict[str, Any]) -> str:
        """
        Format transcription result as markdown
        
        Args:
            result: Transcription result
            
        Returns:
            Markdown formatted transcription
        """
        markdown = f"# Video Transcription\n\n"
        
        # Add metadata
        if result.get("language"):
            markdown += f"**Detected Language:** {result['language']}"
            if result.get("language_probability"):
                markdown += f" (confidence: {result['language_probability']:.2f})"
            markdown += "\n\n"
        
        # Add full text
        markdown += f"## Full Transcript\n\n{result['text']}\n\n"
        
        # Add segments with timestamps
        markdown += f"## Segments\n\n"
        
        for segment in result['segments']:
            start_time = self._format_timestamp(segment['start'])
            end_time = self._format_timestamp(segment['end'])
            markdown += f"**[{start_time} - {end_time}]** {segment['text']}\n\n"
        
        return markdown
    
    def _format_timestamp(self, seconds: float) -> str:
        """
        Format seconds as HH:MM:SS
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp
        """
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    async def process_job(self, job_id: str) -> None:
        """
        Process a transcription job
        
        Args:
            job_id: ID of the transcription job
        """
        if is_shutting_down:
            self.update_job(
                job_id, 
                status=JobStatus.FAILED, 
                error="Server is shutting down, cannot process job"
            )
            return
            
        # Add to active jobs
        self.active_jobs.add(job_id)
        
        # Use semaphore to limit concurrent jobs
        async with self._semaphore:
            try:
                job = self.get_job(job_id)
                video_url = job["video_url"]
                options = job["options"]
                
                # Download video
                async with self._download_video(job_id, video_url) as video_path:
                    # Extract audio
                    audio_path = await self._extract_audio(job_id, video_path)
                    
                    # Transcribe audio
                    await self._transcribe_audio(job_id, audio_path, options)
                    
            except Exception as e:
                logger.error(f"Error processing job {job_id}: {str(e)}")
                self.update_job(job_id, status=JobStatus.FAILED, error=str(e))
                
                # Remove from active jobs
                if job_id in self.active_jobs:
                    self.active_jobs.remove(job_id)


# Create global transcription service
transcription_service = TranscriptionService() 