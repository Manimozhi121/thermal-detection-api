# VigilSense API — Thermal Human Detection Backend

FastAPI backend powering the **VigilSense** system for real-time human detection in thermal imagery.  
Serves a fine-tuned **YOLOv8** model optimized for edge deployment.

## Tech Stack
**FastAPI** • **YOLOv8** • **Python 3.8+**

---

## To Start and run the server
```bash
git clone https://github.com/Manimozhi121/thermal-detection-api.git
cd thermal-detection-api-main
pip install -r requirements.txt

# Run the server
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```
## Usage
You can send inference requests from the frontend web app or directly via Bash:
```bash
curl -X POST "http://localhost:8000/predict/" \
  -F "file=@path/to/thermal_image.jpg"
```
