from fastapi import APIRouter
from fastapi.responses import JSONResponse
from src.schemas.request import FullSequence, WordItem
from typing import List
import pandas as pd
import os

router = APIRouter()

prediction_service_lstm = None
prediction_service_gcn = None

def set_prediction_service_lstm(service):
    global prediction_service_lstm
    prediction_service_lstm = service

def set_prediction_service_gcn(service):
    global prediction_service_gcn
    prediction_service_gcn = service

@router.get("/api/words", response_model=List[WordItem])
async def get_words():
    df = pd.read_csv('Model/data.csv')
    # Chỉ lấy 3 từ đầu tiên cho demo (có thể thay đổi hoặc xóa để lấy tất cả)
    selected_ids = [142, 143, 144, 145, 146, 147]
    df_filtered = df[df['ID'].isin(selected_ids)]
    
    words = []
    for _, row in df_filtered.iterrows():
        words.append({
            "id": int(row['ID']),
            "label": row['LABEL'],
            "video_path": f"/api/videos/{row['VIDEO']}",
            "topic_id": row['topic_id']  # Thêm topic_id từ CSV
        })
    return words

@router.get("/api/videos/{video_name}")
async def get_video(video_name: str):
    from fastapi.responses import FileResponse
    video_path = os.path.join('Model', 'Videos', video_name)
    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4")
    return JSONResponse({"error": "Video not found"}, status_code=404)

@router.post("/api/predict-lstm")
async def predict_lstm(data: FullSequence):
    if not data.sequence:
        return JSONResponse({"error": "Empty sequence"}, status_code=400)
    
    if not prediction_service_lstm:
        return JSONResponse({"error": "Prediction service not available"}, status_code=500)
    
    predicted_label, confidence, predicted_id, top_preds = prediction_service_lstm.predict_sequence(data.sequence)
    
    # Kiểm tra nếu không có prediction (trả về None)
    if predicted_label is None:
        return JSONResponse({"error": "Cannot extract keypoints from sequence"}, status_code=400)
    
    response = {
        "prediction_result": predicted_label,
        "prediction_id": predicted_id,
        "confidence": float(confidence),
        "top_predictions": top_preds
    }
    return JSONResponse(response)

@router.post("/api/predict-gcn")
async def predict_gcn(data: FullSequence):
    if not data.sequence:
        return JSONResponse({"error": "Empty sequence"}, status_code=400)
    
    if not prediction_service_gcn:
        return JSONResponse({"error": "GCN Prediction service not available"}, status_code=500)
    
    # Kiểm tra topic_id có được cung cấp không
    if not data.topic_id:
        return JSONResponse({"error": "topic_id is required"}, status_code=400)
    
    # Gọi predict_sequence với topic_id
    predicted_label, confidence, predicted_id, top_preds = prediction_service_gcn.predict_sequence(
        data.sequence, 
        data.topic_id
    )
    
    # Kiểm tra nếu không có prediction (trả về None)
    if predicted_label is None:
        return JSONResponse({"error": "Cannot extract keypoints from sequence or invalid topic_id"}, status_code=400)
    
    response = {
        "prediction_result": predicted_label,
        "prediction_id": predicted_id,
        "confidence": float(confidence),
        "top_predictions": top_preds
    }
    return JSONResponse(response)