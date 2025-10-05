"""
WorldAway - Exoplanet Seeker Backend
FastAPI application for exoplanet classification
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import uvicorn
import os
import uuid
import json
from datetime import datetime
import logging

from utils.preprocessing import preprocess_row, preprocess_dataframe
from utils.explainability import get_shap_values, get_top_features
from utils.file_handler import read_input_file, write_output_file
from utils.model_loader import ModelLoader
from utils.validators import validate_file_size, validate_file_type, validate_row_data
from config import settings

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# initialize FastAPI app
app = FastAPI(
    title="WorldAway - Exoplanet Seeker API",
    description="ML-powered exoplanet classification system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# initialize model loader (singleton)
model_loader = ModelLoader(settings.MODEL_PATH)

# global model variable
model = None

# in-memory job storage (use Redis in production)
jobs_storage = {}


# pydantic models
class PredictionRequest(BaseModel):
    koi_period: float = Field(..., description="Orbital period (days)")
    koi_time0bk: float = Field(..., description="Transit epoch (BKJD)")
    koi_impact: float = Field(..., description="Impact parameter")
    koi_duration: float = Field(..., description="Transit duration (hours)")
    koi_depth: float = Field(..., description="Transit depth (ppm)")
    koi_prad: float = Field(..., description="Planetary radius (Earth radii)")
    koi_teq: float = Field(..., description="Equilibrium temperature (K)")
    koi_insol: float = Field(..., description="Insolation flux (Earth flux)")
    koi_model_snr: float = Field(..., description="Transit signal-to-noise")
    koi_steff: float = Field(..., description="Stellar effective temperature (K)")
    koi_slogg: float = Field(..., description="Stellar surface gravity (log10(cm/s²))")
    koi_srad: float = Field(..., description="Stellar radius (solar radii)")
    koi_kepmag: float = Field(..., description="Kepler magnitude")


class PredictionResponse(BaseModel):
    predicted_label: str
    predicted_probs: Dict[str, float]
    top_features: List[List]


class JobStatus(BaseModel):
    job_id: str
    status: str  # "processing", "completed", "failed"
    progress: float
    total_rows: Optional[int] = None
    processed_rows: Optional[int] = None
    download_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model
    logger.info("Loading exoplanet classification model...")
    try:
        model_loader.load_model()
        model = model_loader.get_model()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

    # create storage directories
    os.makedirs(settings.JOBS_DIR, exist_ok=True)
    os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "WorldAway - Exoplanet Seeker API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loader.model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/predict/row", response_model=PredictionResponse)
async def predict_single_row(request: PredictionRequest):
    try:
       
        label_map = {0: 'FALSE_POSITIVE', 1: 'CANDIDATE', 2: 'CONFIRMED'}
        
        
        row_dict = request.dict()
        validate_row_data(row_dict)
        processed_row = preprocess_row(row_dict)
        
        
        prediction = model.predict([processed_row])[0]
        prediction = label_map.get(int(prediction), str(prediction))
        
        
        probs = model.predict_proba([processed_row])
        probs_dict = {label_map.get(int(i), str(i)): float(prob) for i, prob in enumerate(probs[0])}
        
        # top features
        feature_names = settings.FEATURE_NAMES
        feature_importance = dict(zip(feature_names, processed_row))
        top_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        return PredictionResponse(
            predicted_label=prediction,
            predicted_probs=probs_dict,
            top_features=top_features
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/file")
async def predict_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload Excel/CSV file for batch prediction
    
    Returns job_id for tracking progress
    """
    try:
        # validate file
        validate_file_type(file.filename)
        
        # create job
        job_id = str(uuid.uuid4())
        upload_path = os.path.join(settings.UPLOADS_DIR, f"{job_id}_{file.filename}")
        
        # save uploaded file
        with open(upload_path, "wb") as f:
            content = await file.read()
            validate_file_size(len(content))
            f.write(content)
        
        # initialize job status
        jobs_storage[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "progress": 0.0,
            "total_rows": None,
            "processed_rows": 0,
            "download_url": None,
            "error_message": None,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "input_file": upload_path
        }
        
        # process file in background
        background_tasks.add_task(process_file_job, job_id, upload_path, file.filename)
        
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "File upload successful. Processing started.",
            "status_endpoint": f"/api/job/{job_id}/status"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        raise HTTPException(status_code=500, detail="File upload failed")


async def process_file_job(job_id: str, input_path: str, original_filename: str):
    """
    Background task to process uploaded file
    """
    try:
        # read input file
        df = read_input_file(input_path)
        total_rows = len(df)
        
        # update job with total rows
        jobs_storage[job_id]["total_rows"] = total_rows
        
        # check row limit
        if total_rows > settings.MAX_ROWS_PER_FILE:
            raise ValueError(f"File contains {total_rows} rows. Maximum allowed is {settings.MAX_ROWS_PER_FILE}")
        
        # preprocess dataframe
        df_processed = preprocess_dataframe(df)
        
        # get model
        model = model_loader.get_model()
        
        # process in batches
        batch_size = settings.BATCH_SIZE
        predictions = []
        prediction_map = {0: 'FALSE_POSITIVE', 1: 'CANDIDATE', 2: 'CONFIRMED'}
        
        probabilities_list = []
        
        for i in range(0, len(df_processed), batch_size):
            batch = df_processed[i:i + batch_size]
            
            # predict
            batch_predictions = model.predict(batch)
            batch_predictions = [prediction_map.get(p, p) for p in batch_predictions]
            batch_probabilities = model.predict_proba(batch)
            
            predictions.extend(batch_predictions)
            probabilities_list.extend(batch_probabilities)
            
            # update progress
            processed_rows = min(i + batch_size, total_rows)
            progress = (processed_rows / total_rows) * 100
            jobs_storage[job_id]["processed_rows"] = processed_rows
            jobs_storage[job_id]["progress"] = round(progress, 2)
        
        # add predictions to original dataframe
        df["predicted_label"] = predictions
        
        # add probability columns
        # class_names = model.classes_
        class_names = ['FALSE_POSITIVE', 'CANDIDATE', 'CONFIRMED']
        for idx, class_name in enumerate(class_names):
            df[f"predicted_prob_{class_name}"] = [probs[idx] for probs in probabilities_list]
        
        # add row IDs
        df["row_id"] = range(1, len(df) + 1)
        
        # save output file
        output_filename = f"output_{original_filename}"
        output_path = os.path.join(settings.JOBS_DIR, f"{job_id}_{output_filename}")
        write_output_file(df, output_path, original_filename)
        
        # update job status
        jobs_storage[job_id].update({
            "status": "completed",
            "progress": 100.0,
            "download_url": f"/api/job/{job_id}/download",
            "completed_at": datetime.now().isoformat()
        })
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        jobs_storage[job_id].update({
            "status": "failed",
            "error_message": str(e),
            "completed_at": datetime.now().isoformat()
        })


@app.get("/api/job/{job_id}/status", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status of a batch processing job"""
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = jobs_storage[job_id]
    return JobStatus(**job_data)


@app.get("/api/job/{job_id}/download")
async def download_results(job_id: str):
    """Download processed results file"""
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = jobs_storage[job_id]
    
    if job_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    # find output file
    output_files = [f for f in os.listdir(settings.JOBS_DIR) if f.startswith(job_id)]
    if not output_files:
        raise HTTPException(status_code=404, detail="Output file not found")
    
    output_path = os.path.join(settings.JOBS_DIR, output_files[0])
    
    return FileResponse(
        output_path,
        media_type="text/csv",
        filename=output_files[0].replace(f"{job_id}_", "").replace('.xlsx', '.csv').replace('.xls', '.csv')
    )


@app.get("/api/job/{job_id}/results")
async def get_job_results(job_id: str, page: int = 1, page_size: int = 50):
    """Get paginated results for a completed job"""
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = jobs_storage[job_id]
    
    if job_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    # read output file
    output_files = [f for f in os.listdir(settings.JOBS_DIR) if f.startswith(job_id)]
    if not output_files:
        raise HTTPException(status_code=404, detail="Results not found")
    
    output_path = os.path.join(settings.JOBS_DIR, output_files[0])
    df = read_input_file(output_path)
    
    # paginate
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    results = df.iloc[start_idx:end_idx].to_dict('records')
    
    return {
        "total_rows": len(df),
        "page": page,
        "page_size": page_size,
        "total_pages": (len(df) + page_size - 1) // page_size,
        "results": results
    }


@app.post("/api/explain")
async def explain_prediction(request: PredictionRequest):
    """
    Get SHAP-based explanation for a prediction
    """
    try:
        # preprocess
        row_dict = request.dict()
        processed_row = preprocess_row(row_dict)
        
        # get model
        model = model_loader.get_model()
        
        # get SHAP values
        shap_values = get_shap_values(model, [processed_row])
        
        # get prediction
        prediction = model.predict([processed_row])[0]
        prediction_map = {0: 'FALSE_POSITIVE', 1: 'CANDIDATE', 2: 'CONFIRMED'}
        prediction = prediction_map.get(prediction, prediction)
        probabilities = model.predict_proba([processed_row])[0]
        class_names = model.classes_
        
        # format SHAP values for each class
        shap_explanations = {}
        for idx, class_name in enumerate(class_names):
            feature_contributions = [
                {
                    "feature": feature_name,
                    "value": float(processed_row[i]),
                    "contribution": float(shap_values[0][i, idx]) if len(shap_values[0].shape) > 1 else float(shap_values[0][i])
                }
                for i, feature_name in enumerate(settings.FEATURE_NAMES)
            ]
            # sort by absolute contribution
            feature_contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
            shap_explanations[class_name] = feature_contributions
        
        return {
            "predicted_label": prediction,
            "predicted_probs": {class_name: float(prob) for class_name, prob in zip(class_names, probabilities)},
            "shap_explanations": shap_explanations,
            "base_value": float(shap_values[0].base_values[0]) if hasattr(shap_values[0], 'base_values') else 0.0
        }
        
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Explanation generation failed")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )