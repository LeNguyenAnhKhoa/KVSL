import torch
import torch.nn.functional as F
import json
import numpy as np
import os
from torch_geometric.data import Data, Batch
from src.models.gcn_model import STGCN
from src.utils.skeleton import extract_keypoints_from_results, interpolate_sequence

def create_fully_connected_edge_index(num_nodes):
    """Tạo edge_index fully connected cho GCN"""
    rows, cols = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    return torch.tensor([rows, cols], dtype=torch.long)

class GCNPredictionService:
    def __init__(self, models_dir):
        """Load tất cả 10 mô hình GCN cho các topic khác nhau.
        
        Args:
            models_dir: Đường dẫn đến thư mục chứa các mô hình .pth và .json
        """
        self.device = torch.device('cpu')
        self.num_frames = 60  # Đã thay đổi từ 40 thành 60 frames
        self.num_nodes = 65  # 23 pose + 0 face + 21 left hand + 21 right hand
        self.num_features_per_node = 3  # x, y, z
        
        # Tạo edge index fully connected (dùng chung cho tất cả các models)
        self.edge_index = create_fully_connected_edge_index(self.num_nodes)
        
        # Dictionary chứa tất cả các models và label_maps theo topic_id
        self.models = {}
        self.label_maps = {}
        self.id_to_labels = {}
        
        # Danh sách các topic_id cần load
        self.topic_ids = [
            'cau_hoi', 'chu_cai', 'do_vat', 'gia_dinh', 'hanh_dong',
            'nghe_nghiep', 'so_dem', 'thoi_tiet', 'tinh_cach', 'tu_thong_dung'
        ]
        
        # Load tất cả các models
        for topic_id in self.topic_ids:
            model_path = os.path.join(models_dir, f'{topic_id}.pth')
            label_map_path = os.path.join(os.path.dirname(models_dir), f'{topic_id}.json')
            
            if not os.path.exists(model_path):
                print(f"Warning: Model not found: {model_path}")
                continue
            if not os.path.exists(label_map_path):
                print(f"Warning: Label map not found: {label_map_path}")
                continue
            
            # Load label map
            with open(label_map_path, 'r', encoding='utf-8') as f:
                label_map = json.load(f)
            
            self.label_maps[topic_id] = label_map
            self.id_to_labels[topic_id] = {v: k for k, v in label_map.items()}
            num_classes = len(label_map)
            
            # Load model
            num_node_features = self.num_frames * self.num_features_per_node  # 60 * 3 = 180
            model = STGCN(num_node_features=num_node_features, num_classes=num_classes).to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            
            self.models[topic_id] = model
            print(f"✓ Loaded GCN model for topic '{topic_id}' with {num_classes} classes")
    
    def predict_sequence(self, sequence_data, topic_id):
        """Dự đoán cử chỉ từ sequence data sử dụng model của topic tương ứng.
        
        Args:
            sequence_data: Danh sách các frame chứa keypoints
            topic_id: ID của topic (vd: 'cau_hoi', 'chu_cai', ...)
            
        Returns:
            predicted_label, confidence, predicted_id, top_predictions
        """
        # Kiểm tra topic_id có hợp lệ không
        if topic_id not in self.models:
            print(f"Error: Topic '{topic_id}' not found. Available topics: {list(self.models.keys())}")
            return None, 0.0, None, []
        
        # Lấy model và label map tương ứng với topic
        model = self.models[topic_id]
        id_to_label = self.id_to_labels[topic_id]
        
        # Trích xuất keypoints từ mỗi frame
        keypoints_list = []
        for frame_data in sequence_data:
            results = frame_data.results
            keypoints = extract_keypoints_from_results(results)
            keypoints_list.append(keypoints)
        
        if len(keypoints_list) == 0:
            return None, 0.0, None, []
        
        # Interpolate về 60 frames (đã thay đổi từ 40)
        interpolated = interpolate_sequence(keypoints_list, target_len=self.num_frames)
        if interpolated is None:
            return None, 0.0, None, []
        
        # Reshape sequence cho GCN:
        # interpolated shape: (60, 65, 3)
        # Cần chuyển thành: (65, 180) - mỗi node có 60 timesteps * 3 features
        # Transpose để có shape (65, 60, 3), rồi flatten thành (65, 180)
        sequence_transposed = np.transpose(interpolated, (1, 0, 2))  # (65, 60, 3)
        features = sequence_transposed.reshape(self.num_nodes, -1)  # (65, 180)
        
        # Tạo Data object cho PyTorch Geometric
        x = torch.tensor(features, dtype=torch.float32)
        data = Data(x=x, edge_index=self.edge_index)
        
        # Tạo batch (vì model mong đợi batch)
        batch = Batch.from_data_list([data]).to(self.device)
        
        # Dự đoán
        with torch.no_grad():
            output = model(batch)
            probabilities = F.softmax(output, dim=1)
            
            top_k = min(3, len(probabilities[0]))
            top_probs, top_indices = torch.topk(probabilities[0], top_k)
            
            predicted_id = top_indices[0].item()
            confidence = top_probs[0].item()
            predicted_label = id_to_label.get(predicted_id, "Unknown")
            
            top_predictions = []
            for i in range(top_k):
                idx = top_indices[i].item()
                prob = top_probs[i].item()
                label = id_to_label.get(idx, "Unknown")
                top_predictions.append({
                    "id": idx,
                    "word": label,
                    "probability": float(prob)
                })
        
        return predicted_label, confidence, predicted_id, top_predictions
