import torch
import torch.nn as nn

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