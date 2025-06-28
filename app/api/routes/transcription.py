from fastapi import APIRouter, HTTPException, BackgroundTasks, status, Depends, Response
from typing import Dict, Any, Optional, Union
import os
import asyncio
from pathlib import Path
import time

from app.core.transcriber import transcription_service
from app.models.transcription import (
    TranscriptionRequest,
    TranscriptionJobResponse,
    TranscriptionResult,
    TranscriptionOptions,
    JobStatus
)

router = APIRouter(prefix="/transcribe", tags=["transcription"])


@router.post(
    "/url", 
    status_code=status.HTTP_200_OK, 
    response_model=Union[TranscriptionResult, TranscriptionJobResponse]
)
async def transcribe_from_url(
    request: TranscriptionRequest,
    background_tasks: BackgroundTasks,
    wait_for_result: bool = True,
    max_wait_time: Optional[float] = None,  # No timeout by default - wait indefinitely
    client_timeout: Optional[float] = None  # Optional client-specific timeout (for n8n, etc.)
) -> Dict[str, Any]:
    """
    Start a new transcription job from a video URL.
    
    If wait_for_result is True (default), the function will wait for the job to complete
    and return the full transcription result. By default, it waits indefinitely.
    
    Timeout options:
    - client_timeout: Timeout for external clients (n8n, etc.) in seconds
    - max_wait_time: General server timeout in seconds
    - If either timeout is exceeded, returns job ID for polling
    
    If wait_for_result is False, it will immediately return a job ID and process in the background.
    """
    try:
        # Create transcription options
        options = TranscriptionOptions(
            language=request.language,
            beam_size=request.beam_size,
            vad_filter=request.vad_filter,
            word_timestamps=request.word_timestamps
        )
        
        # Create job
        job_id = transcription_service.create_job(request.video_url, options)
        
        # Start processing the job
        task = asyncio.create_task(transcription_service.process_job(job_id))
        
        # Check if we should wait for the result
        if wait_for_result:
            try:
                # Wait for job to complete with optional timeout
                start_time = time.time()
                while True:
                    # Check timeout (use client_timeout if provided, otherwise max_wait_time)
                    effective_timeout = client_timeout or max_wait_time
                    if effective_timeout is not None and time.time() - start_time > effective_timeout:
                        # If timeout exceeded, return the job ID for polling
                        job = transcription_service.get_job(job_id)
                        timeout_type = "client" if client_timeout else "server"
                        return {
                            "job_id": job_id,
                            "status": job["status"],
                            "progress": job["progress"],
                            "error": job.get("error"),
                            "output_file": job.get("output_file"),
                            "message": f"Job still processing ({timeout_type} timeout reached), please poll for results"
                        }
                    
                    # Check job status
                    job = transcription_service.get_job(job_id)
                    if job["status"] == JobStatus.COMPLETED:
                        # If job is complete, return the full result
                        result = job["result"]
                        return {
                            "job_id": job_id,
                            "status": job["status"],
                            "language": result.get("language"),
                            "text": result["text"],
                            "segments": result["segments"],
                            "markdown": result["markdown"],
                            "json_path": result.get("json_path")
                        }
                    elif job["status"] == JobStatus.FAILED:
                        # If job failed, raise an error
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Transcription failed: {job.get('error', 'Unknown error')}"
                        )
                    
                    # Wait a short time before checking again
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                # If the task is cancelled, ensure the job continues in the background
                background_tasks.add_task(lambda: None)  # Dummy task to keep event loop running
                return {
                    "job_id": job_id,
                    "status": JobStatus.QUEUED,
                    "progress": 0.0
                }
        else:
            # If not waiting, add the task to background_tasks and return job ID
            background_tasks.add_task(lambda: None)  # Ensure the task continues running
            return {
                "job_id": job_id,
                "status": JobStatus.QUEUED,
                "progress": 0.0
            }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start transcription job: {str(e)}"
        )


@router.get("/{job_id}", response_model=TranscriptionJobResponse)
async def get_job_status(job_id: str) -> Dict[str, Any]:
    """
    Get the status of a transcription job.
    """
    try:
        job = transcription_service.get_job(job_id)
        
        return {
            "job_id": job_id,
            "status": job["status"],
            "progress": job["progress"],
            "error": job.get("error"),
            "output_file": job.get("output_file")
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}"
        )


@router.get("/{job_id}/result", response_model=TranscriptionResult)
async def get_transcription_result(job_id: str) -> Dict[str, Any]:
    """
    Get the result of a completed transcription job.
    """
    try:
        job = transcription_service.get_job(job_id)
        
        if job["status"] != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job is not completed (current status: {job['status']})"
            )
        
        if not job.get("result"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Transcription result not found"
            )
        
        result = job["result"]
        
        return {
            "job_id": job_id,
            "status": job["status"],
            "language": result.get("language"),
            "text": result["text"],
            "segments": result["segments"],
            "markdown": result["markdown"],
            "json_path": result.get("json_path")
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get transcription result: {str(e)}"
        )


@router.get("/{job_id}/json", response_model=None)
async def get_transcription_json(job_id: str) -> Response:
    """
    Get the raw JSON file for a completed transcription.
    """
    try:
        job = transcription_service.get_job(job_id)
        
        if job["status"] != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job is not completed (current status: {job['status']})"
            )
        
        output_file = job.get("output_file")
        if not output_file or not Path(output_file).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="JSON file not found"
            )
        
        # Read the content of the JSON file
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Return the JSON content
        return Response(
            content=content, 
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={Path(output_file).name}"
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get JSON file: {str(e)}"
        ) 