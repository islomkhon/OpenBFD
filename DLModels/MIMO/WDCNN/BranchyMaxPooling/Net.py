import torch
import torch.nn as nn
from DLModels.SISO.WDCNN.Original.Net import FilterBlock , Classifier

class Net(nn.Module):
    def __init__(self, in_channels=1, total_classes=10):

        super(Net, self).__init__()

        self.Edge_0 = nn.Sequential(
            FilterBlock(in_channels, 16 , 64, 16, 24),
            FilterBlock(16, 32, 3, 1, 1),
        )

        self.Edge_1 = nn.Sequential(
            FilterBlock(in_channels, 16 , 64, 16, 24),
            FilterBlock(16, 32, 3, 1, 1),
        )

        self.Cloud_WDCNN = nn.Sequential(
            FilterBlock(32, 64, 3, 1, 1),
            FilterBlock(64, 64, 3, 1, 1),
            FilterBlock(64, 64, 3, 1, 0)
        )

        self.Edge_Classifier = Classifier(1024, total_classes)
        self.Cloud_Classifier = Classifier(192,total_classes)

        self.output_layers = ['edge', 'cloud']

    def forward(self, x):
        
        egde_features_0 = self.Edge_0(x[: , 0])
        egde_features_1 = self.Edge_1(x[: , 1])

        egde_features = torch.maximum(egde_features_1, egde_features_0)
        
        cloud_features = self.Cloud_WDCNN(egde_features)
        
        egde_classificaton = self.Edge_Classifier(egde_features)
        cloud_classificaton = self.Cloud_Classifier(cloud_features)

        return egde_classificaton, cloud_classificaton


