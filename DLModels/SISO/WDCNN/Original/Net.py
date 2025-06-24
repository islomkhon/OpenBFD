'''
WDCNN model with pytorch
Reference:
Wei Zhang, Gaoliang Peng, Chuanhao Li, Yuanhang Chen and Zhujun Zhang
A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals
Sensors, MDPI
doi:10.3390/s17020425
'''
import torch.nn as nn

class FilterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        
        super(FilterBlock, self).__init__()

        self.filter_layers = nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.filter_layers(x)
        return x

class Classifier(nn.Module):
    def __init__(self,input_features, total_classes):

        super(Classifier, self).__init__()
        
        self.classifier_layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_features, 100), 
        nn.ReLU(),
        nn.Linear(100, total_classes)
        )

    def forward(self, x):
        x = self.classifier_layers(x)
        return x


class Net(nn.Module):
    def __init__(self, in_channels=1, total_classes=10):

        super(Net, self).__init__()

        self.model = nn.Sequential(
            FilterBlock(in_channels, 16 , 64, 16, 24),
            FilterBlock(16, 32, 3, 1, 1),
            FilterBlock(32, 64, 3, 1, 1),
            FilterBlock(64, 64, 3, 1, 1),
            FilterBlock(64, 64, 3, 1, 0),
            Classifier(192, total_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x