import torch
import torch.nn.functional as F
import json
import os
import numpy as np
from src.models.lstm_model import SignLanguageModel
from src.utils.skeleton import extract_keypoints_from_results, interpolate_sequence

class LSTMPredictionService:
    def __init__(self, model_path, label_map_path):
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.num_frames = 40  # Sequence length (PHẢI KHỚP VỚI checkpoint_extraction.py target_len=40)
        self.input_size = 195  # 65 landmarks x 3 coordinates
        
        with open(label_map_path, 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)
        
        self.id_to_label = {v: k for k, v in self.label_map.items()}
        self.num_classes = len(self.label_map)
        
        # Load LSTM-GRU model giống hệt train.py
        self.model = SignLanguageModel(input_size=self.input_size, num_classes=self.num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def predict_sequence(self, sequence_data):
        # Trích xuất keypoints từ mỗi frame
        keypoints_list = []
        for frame_data in sequence_data:
            results = frame_data.results
            keypoints = extract_keypoints_from_results(results)
            keypoints_list.append(keypoints)
        
        if len(keypoints_list) == 0:
            return None, 0.0, None, []
        
        # Interpolate về 40 frames
        interpolated = interpolate_sequence(keypoints_list, target_len=self.num_frames)
        if interpolated is None:
            return None, 0.0, None, []
        
        # Reshape sequence giống train.py:
        # Shape: (40, 65, 3) -> (40, 195)
        # Flatten mỗi frame thành vector 195 features
        sequence_flattened = interpolated.reshape(interpolated.shape[0], -1)
        
        # Chuyển thành tensor và thêm batch dimension
        # Shape: (1, 40, 195) - batch_size=1, seq_len=40, features=195
        x = torch.tensor(sequence_flattened, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Dự đoán
        with torch.no_grad():
            output = self.model(x)
            probabilities = F.softmax(output, dim=1)
            
            top_k = min(3, len(probabilities[0]))
            top_probs, top_indices = torch.topk(probabilities[0], top_k)
            
            predicted_id = top_indices[0].item()
            confidence = top_probs[0].item()
            predicted_label = self.id_to_label.get(predicted_id, "Unknown")
            
            top_predictions = []
            for i in range(top_k):
                idx = top_indices[i].item()
                prob = top_probs[i].item()
                label = self.id_to_label.get(idx, "Unknown")
                top_predictions.append({
                    "id": idx,
                    "word": label,
                    "probability": float(prob)
                })
        
        return predicted_label, confidence, predicted_id, top_predictions