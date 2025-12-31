from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from src.routers import api
from src.services.prediction_lstm import LSTMPredictionService
from src.services.prediction_gcn import GCNPredictionService

app = FastAPI(title="Sign Language Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load LSTM-GRU model (không cần thay đổi)
try:
    model_path_lstm = os.path.join('Model', 'lstm.pth')
    label_map_path = os.path.join('Model', 'label_map.json')
    prediction_service_lstm = LSTMPredictionService(model_path_lstm, label_map_path)
    api.set_prediction_service_lstm(prediction_service_lstm)
    print("✓ LSTM model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load LSTM model: {e}")

# Load tất cả 10 GCN models dựa trên topic
try:
    gcn_models_dir = os.path.join('Model', 'gcn')
    prediction_service_gcn = GCNPredictionService(gcn_models_dir)
    api.set_prediction_service_gcn(prediction_service_gcn)
    print("✓ All GCN models loaded successfully")
except Exception as e:
    print(f"✗ Failed to load GCN models: {e}")

app.include_router(api.router)

@app.get("/")
async def read_index():
    return FileResponse('../frontend/index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)