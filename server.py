import io
import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import tempfile
import shutil
import logging
from typing import Optional

# --- Configuration ---
MODEL_PATH = "models/best.pt"
TEMP_DIR = "temp_processing"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Thermal Strike API",
    description="API for processing thermal images and videos for human detection.",
    version="1.0.0"
)

# --- CORS Middleware ---
# This allows your React frontend (running on a different port) to communicate with this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading ---
try:
    logger.info(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    # If the model can't be loaded, the server is not useful.
    # You might want to handle this more gracefully.
    raise

# --- Helper Functions ---

def process_image(contents: bytes) -> bytes:
    """Processes a single image."""
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    results = model(img)
    annotated_img = results[0].plot()

    _, encoded_img = cv2.imencode(".jpg", annotated_img)
    return encoded_img.tobytes()

def process_video(video_path: str) -> str:
    """Processes a video file and returns the path to the annotated video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video file.")

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_filename = f"processed_{os.path.basename(video_path)}"
    output_path = os.path.join(TEMP_DIR, output_filename)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame)
        annotated_frame = results[0].plot()
        
        out.write(annotated_frame)

    cap.release()
    out.release()
    return output_path

# --- API Endpoints ---

@app.post("/api/process")
async def process_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Receives an image or video file, processes it with the YOLO model,
    and returns the annotated result.
    """
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    content_type = file.content_type
    logger.info(f"Received file: {file.filename} with content type: {content_type}")

    if content_type.startswith("image/"):
        contents = await file.read()
        processed_image_bytes = process_image(contents)
        return StreamingResponse(io.BytesIO(processed_image_bytes), media_type="image/jpeg")

    elif content_type.startswith("video/"):
        # Save video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, dir=TEMP_DIR, suffix=".mp4") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_video_path = tmp.name
        
        try:
            processed_video_path = process_video(temp_video_path)
            # Use a background task to clean up the processed file after sending it
            if background_tasks:
                background_tasks.add_task(os.remove, processed_video_path)
            response = FileResponse(processed_video_path, media_type="video/mp4", filename=os.path.basename(processed_video_path))
            # The background task will run after the response is sent.
            return response
        finally:
            # Clean up temporary files
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload an image or video.")

@app.get("/")
def read_root():
    return {"message": "Thermal Strike API is online."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)