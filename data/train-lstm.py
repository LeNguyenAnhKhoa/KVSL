import os
import json
import glob
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

DATA_DIR = '/kaggle/input/visign/Data2'
LABEL_MAP_PATH = '/kaggle/input/visign/Data/label_map.json'
MODEL_SAVE_PATH = 'lstm.pth'
NUM_EPOCHS = 20
BATCH_SIZE = 8192
LEARNING_RATE = 1e-4
TEST_SIZE = 0.1
RANDOM_STATE = 42
NUM_WORKERS = os.cpu_count()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_label(file_path):
    try:
        with np.load(file_path) as data:
            return file_path, int(data['label'].item())
    except Exception as e:
        return None

def get_filepaths_and_labels(data_dir):
    filepaths = glob.glob(os.path.join(data_dir, '*', '*.npz'))
    filepaths_and_labels = []
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        with tqdm(total=len(filepaths), desc="Scanning Labels") as pbar:
            futures = [executor.submit(load_label, fp) for fp in filepaths]
            for future in futures:
                result = future.result()
                if result:
                    filepaths_and_labels.append(result)
                pbar.update(1)
                
    if not filepaths_and_labels:
        return [], []
        
    paths, labels = zip(*filepaths_and_labels)
    return list(paths), list(labels)

class SignLanguageDataset(Dataset):
    def __init__(self, filepaths, labels):
        self.filepaths = filepaths
        self.labels = labels

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]
        with np.load(filepath) as data:
            sequence = data['sequence'].astype(np.float32)
        
        sequence = sequence.reshape(sequence.shape[0], -1)
        return torch.from_numpy(sequence), torch.tensor(label, dtype=torch.long)

class SignLanguageModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SignLanguageModel, self).__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.2)
        
        self.gru = nn.GRU(64, 160, batch_first=True)
        self.bn2 = nn.BatchNorm1d(160)
        self.dropout2 = nn.Dropout(0.4)
        
        self.fc1 = nn.Linear(160, 224)
        self.bn3 = nn.BatchNorm1d(224)
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(224, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        x = self.bn1(x)
        x = x.permute(0, 2, 1)
        x = self.dropout1(x)
        
        x, h_n = self.gru(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        x = self.bn2(x)
        x = x.permute(0, 2, 1)
        x = self.dropout2(x)
        
        x = h_n.squeeze(0)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.dropout3(x)
        
        x = self.fc2(x)
        return x

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
    return running_loss / total_samples, correct_predictions / total_samples

def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
    return running_loss / total_samples, correct_predictions / total_samples

print(f"Using device: {device}")
print(f"Number of available CPU cores: {NUM_WORKERS}")

try:
    with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    NUM_CLASSES = len(label_map)
except FileNotFoundError:
    print(f"Error: label_map.json not found at {LABEL_MAP_PATH}. Please ensure it exists.")
    exit()

all_filepaths, all_labels = get_filepaths_and_labels(DATA_DIR)

if not all_filepaths:
    print(f"No .npz files found in {DATA_DIR}. Please check the data path.")
    exit()

train_filepaths, test_filepaths, train_labels, test_labels = train_test_split(
    all_filepaths,
    all_labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=all_labels
)

train_dataset = SignLanguageDataset(train_filepaths, train_labels)
test_dataset = SignLanguageDataset(test_filepaths, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# --- BẮT ĐẦU THAY ĐỔI ---
# Tự động xác định INPUT_SIZE thay vì hardcode
try:
    # Lấy một file mẫu từ tập train để kiểm tra shape
    sample_filepath = train_filepaths[0] 
    with np.load(sample_filepath) as data:
        sample_sequence = data['sequence']
        
        # Shape gốc của sequence là (frames, num_landmarks, 3)
        # Lớp Dataset sẽ reshape nó thành (frames, num_landmarks * 3)
        # Vì vậy INPUT_SIZE chính là (num_landmarks * 3)
        
        if sample_sequence.ndim != 3 or sample_sequence.shape[2] != 3:
             raise ValueError(f"Định dạng dữ liệu không mong đợi. " \
                              f"Shape: {sample_sequence.shape}. " \
                              "Mong đợi (frames, landmarks, 3)")

        num_landmarks = sample_sequence.shape[1]
        num_coords = sample_sequence.shape[2] # Phải là 3 (x, y, z)
        INPUT_SIZE = num_landmarks * num_coords
        
        print(f"Đã tự động phát hiện INPUT_SIZE: {INPUT_SIZE} " \
              f"({num_landmarks} landmarks * {num_coords} coords)")

except Exception as e:
    print(f"Lỗi khi đọc file mẫu {sample_filepath} để xác định INPUT_SIZE: {e}")
    print("Vui lòng kiểm tra lại dữ liệu đầu ra từ checkpoint_extraction.py.")
    exit()
# --- KẾT THÚC THAY ĐỔI ---

model = SignLanguageModel(input_size=INPUT_SIZE, num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas = (0.9, 0.98), eps = 1e-9, weight_decay=0.01)

print("\nStarting model training...")
for epoch in range(NUM_EPOCHS):
    print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate_epoch(model, test_loader, criterion, device)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\nTraining finished. Model saved to {MODEL_SAVE_PATH}")