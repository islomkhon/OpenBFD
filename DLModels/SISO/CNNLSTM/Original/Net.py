import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_channels=1, total_classes=10):
        super(Net, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, 84, kernel_size=84)
        self.activation = nn.ELU()
        self.dropout1 = nn.Dropout(0.01)
        self.batchnorm1 = nn.BatchNorm1d(84)
        self.maxpool = nn.MaxPool1d(8)
        self.lstm = nn.LSTM(10, 1, batch_first=True)
        self.dropout2 = nn.Dropout(0.01)
        self.fc = nn.Linear(1, total_classes)
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1d(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.batchnorm1(x)
        x = self.maxpool(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout2(x)
        x = self.fc(x)
        # x = self.sigmoid(x)
        return x