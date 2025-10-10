"""
FastAPI Application for Gemini Invoice Extraction
Provides REST API endpoints for invoice extraction using Gemini-Flash VLM
"""
import os
import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
from loguru import logger
import zipfile

# Import extractor components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.gemini_extractor.pipeline import GeminiInvoiceExtractor, BatchProcessor
from src.gemini_extractor.config import load_config
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Gemini Invoice Extraction API",
    description="Extract invoice information from images using Google Gemini-Flash VLM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
try:
    config = load_config()
    logger.info("✅ Configuration loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load configuration: {e}")
    config = None

# Initialize extractor
extractor = None
batch_processor = None

# Directories for file storage
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
TEMP_DIR = Path("temp")

# Create directories
for directory in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# In-memory job storage (use Redis/DB in production)
jobs_db: Dict[str, Dict[str, Any]] = {}


# ===================================================================
# Pydantic Models
# ===================================================================

class ExtractionRequest(BaseModel):
    """Request model for extraction"""
    prompt: Optional[str] = Field(
        default="Extract invoice data to JSON",
        description="Custom prompt for extraction"
    )
    return_raw: Optional[bool] = Field(
        default=False,
        description="Return raw response without validation"
    )
    save_result: Optional[bool] = Field(
        default=True,
        description="Save extraction result to file"
    )


class ProductInfo(BaseModel):
    """Product information"""
    PRODUCT: str = Field(..., description="Product name")
    NUM: int = Field(..., description="Quantity")
    VALUE: float = Field(..., description="Total value")


class InvoiceData(BaseModel):
    """Invoice data model"""
    SELLER: str = Field(..., description="Seller name")
    TIMESTAMP: str = Field(..., description="Transaction timestamp")
    PRODUCTS: List[ProductInfo] = Field(..., description="List of products")
    TOTAL_COST: float = Field(..., description="Total cost")
    ADDRESS: Optional[str] = Field(None, description="Seller address")
    TAX_CODE: Optional[str] = Field(None, description="Tax code")


class ExtractionResponse(BaseModel):
    """Response model for extraction"""
    success: bool
    message: str
    data: Optional[InvoiceData] = None
    job_id: Optional[str] = None
    processing_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class BatchExtractionResponse(BaseModel):
    """Response for batch extraction"""
    success: bool
    message: str
    job_id: str
    total_files: int
    status_url: str


class JobStatus(BaseModel):
    """Job status model"""
    job_id: str
    status: str  # pending, processing, completed, failed
    total_files: int
    processed: int
    successful: int
    failed: int
    created_at: str
    completed_at: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None
    errors: Optional[List[Dict[str, Any]]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    extractor_initialized: bool
    config_loaded: bool
    model_version: Optional[str] = None


# ===================================================================
# Startup & Shutdown Events
# ===================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize extractor on startup"""
    global extractor, batch_processor
    
    logger.info("🚀 Starting Gemini Invoice Extraction API...")
    
    try:
        if config is None:
            raise ValueError("Configuration not loaded")
        
        # Validate API key is set
        if not config.validate():
            raise ValueError("Configuration validation failed")
        
        # Initialize extractor
        extractor = GeminiInvoiceExtractor(config)
        batch_processor = BatchProcessor(extractor)
        
        logger.info("✅ Extractor initialized successfully")
        logger.info(f"📋 Model: {config.get('api.model_version')}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize extractor: {e}")
        logger.warning("⚠️ API will start but extraction endpoints will not work")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down API...")
    
    # Clean up temporary files
    try:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        TEMP_DIR.mkdir(exist_ok=True)
        logger.info("🧹 Cleaned up temporary files")
    except Exception as e:
        logger.error(f"Failed to clean up temp files: {e}")


# ===================================================================
# Helper Functions
# ===================================================================

def save_upload_file(upload_file: UploadFile) -> Path:
    """
    Save uploaded file to disk
    
    Args:
        upload_file: FastAPI UploadFile object
        
    Returns:
        Path to saved file
    """
    # Generate unique filename
    file_ext = Path(upload_file.filename).suffix
    filename = f"{uuid.uuid4()}{file_ext}"
    file_path = UPLOAD_DIR / filename
    
    # Save file
    with open(file_path, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    
    return file_path


def validate_image_file(file: UploadFile) -> None:
    """
    Validate uploaded image file
    
    Args:
        file: Uploaded file
        
    Raises:
        HTTPException: If file is invalid
    """
    # Check file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Check file size (max 10MB)
    max_size = config.get('preprocessing.max_file_size_mb', 10) * 1024 * 1024
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {max_size / 1024 / 1024:.1f}MB"
        )


def create_job(total_files: int) -> str:
    """
    Create a new processing job
    
    Args:
        total_files: Number of files to process
        
    Returns:
        Job ID
    """
    job_id = str(uuid.uuid4())
    
    jobs_db[job_id] = {
        'job_id': job_id,
        'status': 'pending',
        'total_files': total_files,
        'processed': 0,
        'successful': 0,
        'failed': 0,
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
        'results': [],
        'errors': []
    }
    
    return job_id


def update_job(job_id: str, **kwargs):
    """Update job status"""
    if job_id in jobs_db:
        jobs_db[job_id].update(kwargs)


# ===================================================================
# API Endpoints
# ===================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "extractor_initialized": extractor is not None,
        "config_loaded": config is not None,
        "model_version": config.get('api.model_version') if config else None
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if extractor is not None else "degraded",
        "version": "1.0.0",
        "extractor_initialized": extractor is not None,
        "config_loaded": config is not None,
        "model_version": config.get('api.model_version') if config else None
    }


@app.post("/extract", response_model=ExtractionResponse)
async def extract_invoice(
    file: UploadFile = File(..., description="Invoice image file"),
    model_name: Optional[str] = Form(default="gemini-flash-latest", description="Model name to use"),
    prompt: Optional[str] = Form(default="Extract invoice data to JSON"),
    return_raw: Optional[bool] = Form(default=False),
    save_result: Optional[bool] = Form(default=True)
):
    """
    Extract invoice information from a single image
    
    Args:
        file: Invoice image file (JPG, PNG, BMP)
        prompt: Custom extraction prompt
        return_raw: Return raw response without validation
        save_result: Save result to output directory
        
    Returns:
        Extracted invoice data
    """
    # if config.get('api.model_version') != model_name:
    #     # Reload extractor with new model
    extractor = GeminiInvoiceExtractor(config, model_name)
    
    if extractor is None:
        raise HTTPException(
            status_code=503,
            detail="Extractor not initialized. Check server logs."
        )

    start_time = datetime.now()
    file_path = None
    
    try:
        # Validate file
        validate_image_file(file)
        
        # Save uploaded file
        file_path = save_upload_file(file)
        logger.info(f"📁 Processing file: {file.filename}")
        
        # Extract invoice data
        result = extractor.extract(
            image_path=file_path,
            prompt=prompt,
            return_raw=return_raw
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Save result if requested
        if save_result:
            output_file = OUTPUT_DIR / f"{job_id}.json"
            extractor.save_result(result, output_file)
        
        logger.info(f"✅ Extraction successful for {file.filename} (job_id: {job_id})")
        
        return {
            "success": True,
            "message": "Extraction completed successfully",
            "data": result,
            "job_id": job_id,
            "processing_time": processing_time,
            "metadata": {
                "filename": file.filename,
                "file_size": file_path.stat().st_size,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Extraction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}"
        )
    
    finally:
        # Clean up uploaded file
        if file_path and file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")


@app.post("/extract/batch", response_model=BatchExtractionResponse)
async def extract_batch(
    files: List[UploadFile] = File(..., description="Multiple invoice images"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    prompt: Optional[str] = Form(default="Extract invoice data to JSON"),
    continue_on_error: Optional[bool] = Form(default=True)
):
    """
    Extract invoice information from multiple images in batch
    
    Args:
        files: List of invoice image files
        prompt: Custom extraction prompt
        continue_on_error: Continue processing if one file fails
        
    Returns:
        Job information for batch processing
    """
    if batch_processor is None:
        raise HTTPException(
            status_code=503,
            detail="Batch processor not initialized"
        )
    
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )
    
    # Create job
    job_id = create_job(len(files))
    
    # Save all files
    saved_files = []
    for file in files:
        try:
            validate_image_file(file)
            file_path = save_upload_file(file)
            saved_files.append((file.filename, file_path))
        except Exception as e:
            logger.error(f"Failed to save file {file.filename}: {e}")
            if not continue_on_error:
                raise HTTPException(status_code=400, detail=str(e))
    
    # Process batch in background
    background_tasks.add_task(
        process_batch_job,
        job_id=job_id,
        files=saved_files,
        prompt=prompt,
        continue_on_error=continue_on_error
    )
    
    logger.info(f"🚀 Batch job created: {job_id} ({len(saved_files)} files)")
    
    return {
        "success": True,
        "message": "Batch processing started",
        "job_id": job_id,
        "total_files": len(saved_files),
        "status_url": f"/jobs/{job_id}"
    }


async def process_batch_job(
    job_id: str,
    files: List[tuple],
    prompt: str,
    continue_on_error: bool
):
    """
    Background task to process batch extraction
    
    Args:
        job_id: Job identifier
        files: List of (filename, filepath) tuples
        prompt: Extraction prompt
        continue_on_error: Continue on errors
    """
    update_job(job_id, status='processing')
    
    results = []
    errors = []
    successful = 0
    failed = 0
    
    for i, (filename, file_path) in enumerate(files, 1):
        try:
            logger.info(f"Processing {i}/{len(files)}: {filename}")
            
            # Extract
            data = extractor.extract(file_path, prompt=prompt)
            
            # Save result
            result_file = OUTPUT_DIR / f"{job_id}_{i}_{Path(filename).stem}.json"
            extractor.save_result(data, result_file)
            
            results.append({
                'filename': filename,
                'success': True,
                'data': data,
                'result_file': str(result_file)
            })
            successful += 1
            
        except Exception as e:
            logger.error(f"Failed to process {filename}: {str(e)}")
            errors.append({
                'filename': filename,
                'error': str(e)
            })
            failed += 1
            
            if not continue_on_error:
                break
        
        finally:
            # Clean up uploaded file
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {e}")
            
            # Update progress
            update_job(
                job_id,
                processed=i,
                successful=successful,
                failed=failed
            )
    
    # Mark job as completed
    update_job(
        job_id,
        status='completed',
        completed_at=datetime.now().isoformat(),
        results=results,
        errors=errors
    )
    
    logger.info(f"✅ Batch job completed: {job_id} ({successful}/{len(files)} successful)")


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """
    Get status of a batch processing job
    
    Args:
        job_id: Job identifier
        
    Returns:
        Job status information
    """
    if job_id not in jobs_db:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}"
        )
    
    return jobs_db[job_id]


@app.get("/jobs/{job_id}/results")
async def get_job_results(job_id: str):
    """
    Get detailed results of a completed job
    
    Args:
        job_id: Job identifier
        
    Returns:
        JSON with all extraction results
    """
    if job_id not in jobs_db:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}"
        )
    
    job = jobs_db[job_id]
    
    if job['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed yet. Status: {job['status']}"
        )
    
    return JSONResponse(content=job['results'])


@app.get("/jobs/{job_id}/download")
async def download_job_results(job_id: str):
    """
    Download all results as a ZIP file
    
    Args:
        job_id: Job identifier
        
    Returns:
        ZIP file with all JSON results
    """
    if job_id not in jobs_db:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}"
        )
    
    job = jobs_db[job_id]
    
    if job['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed yet. Status: {job['status']}"
        )
    
    # Create ZIP file
    zip_path = TEMP_DIR / f"{job_id}_results.zip"
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for result in job['results']:
                if result.get('success') and 'result_file' in result:
                    result_file = Path(result['result_file'])
                    if result_file.exists():
                        zipf.write(result_file, result_file.name)
        
        return FileResponse(
            path=zip_path,
            media_type='application/zip',
            filename=f"job_{job_id}_results.zip"
        )
        
    except Exception as e:
        logger.error(f"Failed to create ZIP: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create download: {str(e)}"
        )


@app.get("/jobs")
async def list_jobs(
    limit: int = Query(default=10, ge=1, le=100),
    status: Optional[str] = Query(default=None)
):
    """
    List all jobs
    
    Args:
        limit: Maximum number of jobs to return
        status: Filter by status (pending, processing, completed, failed)
        
    Returns:
        List of jobs
    """
    jobs = list(jobs_db.values())
    
    # Filter by status
    if status:
        jobs = [j for j in jobs if j['status'] == status]
    
    # Sort by creation time (newest first)
    jobs.sort(key=lambda x: x['created_at'], reverse=True)
    
    # Limit results
    jobs = jobs[:limit]
    
    return {
        "total": len(jobs_db),
        "returned": len(jobs),
        "jobs": jobs
    }


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and its results
    
    Args:
        job_id: Job identifier
        
    Returns:
        Success message
    """
    if job_id not in jobs_db:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}"
        )
    
    job = jobs_db[job_id]
    
    # Delete result files
    deleted_files = 0
    if 'results' in job:
        for result in job['results']:
            if 'result_file' in result:
                result_file = Path(result['result_file'])
                try:
                    if result_file.exists():
                        result_file.unlink()
                        deleted_files += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {result_file}: {e}")
    
    # Remove from database
    del jobs_db[job_id]
    
    logger.info(f"🗑️ Deleted job {job_id} and {deleted_files} result files")
    
    return {
        "success": True,
        "message": f"Job {job_id} deleted",
        "files_deleted": deleted_files
    }


@app.get("/config")
async def get_configuration():
    """
    Get current API configuration (sensitive data excluded)
    
    Returns:
        Configuration dictionary
    """
    if config is None:
        raise HTTPException(
            status_code=503,
            detail="Configuration not loaded"
        )
    
    # Return safe config (exclude API key)
    safe_config = config.to_dict()
    if 'api' in safe_config and 'api_key' in safe_config['api']:
        safe_config['api']['api_key'] = '***HIDDEN***'
    
    return safe_config


@app.post("/config/reload")
async def reload_configuration():
    """
    Reload configuration from file
    
    Returns:
        Success message
    """
    global config, extractor, batch_processor
    
    try:
        config.reload()
        
        # Reinitialize extractor with new config
        extractor = GeminiInvoiceExtractor(config, config.get('api.model_version'))
        batch_processor = BatchProcessor(extractor)
        
        logger.info("🔄 Configuration reloaded and extractor reinitialized")
        
        return {
            "success": True,
            "message": "Configuration reloaded successfully",
            "model_version": config.get('api.model_version')
        }
        
    except Exception as e:
        logger.error(f"Failed to reload config: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload configuration: {str(e)}"
        )


# ===================================================================
# Main Entry Point
# ===================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Get host and port from config or environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    logger.info(f"🚀 Starting API server at http://{host}:{port}")
    logger.info(f"📚 API Documentation: http://{host}:{port}/docs")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
