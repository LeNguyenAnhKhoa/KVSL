import os
import json
import glob
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

DATA_PATH = '/kaggle/input/tu-thong-dung/Data'
MODEL_SAVE_PATH = 'tu_thong_dung.pth'
EPOCHS = 7
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4
TEST_SIZE = 0.05

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(os.path.join(DATA_PATH, 'label_map.json'), 'r', encoding='utf-8') as f:
    label_map = json.load(f)
NUM_CLASSES = len(label_map)

# --- START: THAY ĐỔI ĐỂ TẢI KÍCH THƯỚC ĐỘNG ---

# 1. Tìm tất cả các tệp trước
all_files = glob.glob(os.path.join(DATA_PATH, '*/*.npz'))
if not all_files:
    raise FileNotFoundError(f"Không tìm thấy tệp .npz nào trong {DATA_PATH}. Hãy chạy script checkpoint_extraction.py trước.")

# 2. Tải tệp đầu tiên để xác định kích thước dữ liệu
print(f"Đang tải tệp đầu tiên để xác định kích thước: {all_files[0]}")
try:
    with np.load(all_files[0]) as npz_file:
        first_sequence = npz_file['sequence']
except Exception as e:
    raise IOError(f"Không thể tải tệp đầu tiên để kiểm tra kích thước: {e}")

if first_sequence.ndim != 3:
    raise ValueError(f"Kích thước sequence không hợp lệ. Mong đợi 3 chiều (timesteps, nodes, features), nhưng nhận được {first_sequence.ndim} chiều.")

# 3. Gán kích thước từ dữ liệu thực tế (thay vì hardcode)
# Hình dạng (shape) mong đợi từ script_extraction là (timesteps, nodes, features)
NUM_TIMESTEPS, NUM_NODES, NUM_FEATURES_PER_NODE = first_sequence.shape

print("--- Tự động phát hiện kích thước dữ liệu ---")
print(f"  NUM_NODES: {NUM_NODES}")
print(f"  NUM_TIMESTEPS: {NUM_TIMESTEPS}")
print(f"  NUM_FEATURES_PER_NODE: {NUM_FEATURES_PER_NODE}")
print("-----------------------------------------")

# --- END: THAY ĐỔI ĐỂ TẢI KÍCH THƯỚC ĐỘNG ---


def create_fully_connected_edge_index(num_nodes):
    rows, cols = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    return torch.tensor([rows, cols], dtype=torch.long)

# 4. Tạo edge_index SAU KHI đã biết NUM_NODES
edge_index = create_fully_connected_edge_index(NUM_NODES)

def process_file(file_path):
    try:
        with np.load(file_path) as npz_file:
            sequence = npz_file['sequence']
            label = npz_file['label']
        
        # 5. Kiểm tra xem tệp này có khớp với kích thước đã phát hiện không
        if sequence.shape != (NUM_TIMESTEPS, NUM_NODES, NUM_FEATURES_PER_NODE):
            # Bỏ qua tệp nếu nó không khớp (ví dụ: từ một lần chạy cũ)
            return None

        # Dữ liệu đầu vào của GCN (node features) có kích thước (số lượng nodes, số features mỗi node)
        # Chúng ta "làm phẳng" (flatten) chiều thời gian và chiều features (x,y,z) lại
        # (NUM_TIMESTEPS, NUM_NODES, NUM_FEATURES) -> (NUM_NODES, NUM_TIMESTEPS, NUM_FEATURES)
        # -> (NUM_NODES, NUM_TIMESTEPS * NUM_FEATURES)
        features = torch.tensor(sequence, dtype=torch.float).permute(1, 0, 2).reshape(NUM_NODES, -1)
        y = torch.tensor(int(label), dtype=torch.long)
        
        return Data(x=features, edge_index=edge_index, y=y)
    except Exception as e:
        # print(f"Lỗi khi xử lý tệp {file_path}: {e}") # Bỏ comment này nếu bạn muốn debug
        return None

class SignLanguageDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

# all_files đã được tải ở trên
print("Đang tải toàn bộ dữ liệu (có thể mất một lúc)...")
data_list = [process_file(f) for f in tqdm(all_files, desc="Loading data")]

original_count = len(all_files) # So sánh với tổng số tệp, không phải len(data_list)
data_list = [d for d in data_list if d is not None]
filtered_count = len(data_list)

if original_count > filtered_count:
    print(f"Cảnh báo: Đã lọc bỏ {original_count - filtered_count} mẫu bị lỗi hoặc không khớp kích thước.")
else:
    print("Tất cả các tệp đã được tải thành công!")

if not data_list:
    raise ValueError("Không có dữ liệu hợp lệ nào được tải. Vui lòng kiểm tra lại các tệp .npz.")

train_data_list, test_data_list = train_test_split(data_list, test_size=TEST_SIZE, stratify=[d.y.item() for d in data_list], random_state=42)

train_dataset = SignLanguageDataset(train_data_list)
test_dataset = SignLanguageDataset(test_data_list)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

class STGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(STGCN, self).__init__()
        # 6. Sử dụng số features đầu vào đã được tính toán động
        self.conv1 = GCNConv(num_node_features, 256)
        self.conv2 = GCNConv(256, 256)
        self.conv3 = GCNConv(256, 256)
        self.fc = Linear(256, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

# 7. Khởi tạo mô hình với các kích thước động
input_features = NUM_TIMESTEPS * NUM_FEATURES_PER_NODE
model = STGCN(num_node_features=input_features, num_classes=NUM_CLASSES).to(device)
print(f"Đã khởi tạo mô hình STGCN với {input_features} features đầu vào cho mỗi node.")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas = (0.9, 0.98), eps = 1e-9, weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train_epoch():
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0
    
    for data in tqdm(train_loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        total_samples += data.num_graphs
        
    return total_loss / total_samples, correct / total_samples

def test_epoch(loader):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data in tqdm(loader, desc="Testing"):
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            
            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total_samples += data.num_graphs
            
    return total_loss / total_samples, correct / total_samples

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_epoch()
    test_loss, test_acc = test_epoch(test_loader)
    
    print(f'Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")