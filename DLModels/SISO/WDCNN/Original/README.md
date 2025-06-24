# WDCNN: Wide and Deep Convolutional Neural Network for Fault Diagnosis

## Overview
This repository implements the Wide and Deep Convolutional Neural Network (WDCNN) model for fault diagnosis, as proposed in the research paper by Zhang et al. The model is designed to provide robust fault diagnosis capabilities with good anti-noise and domain adaptation abilities on raw vibration signals.

## Paper Reference
- **Title:** A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals
- **Authors:** Wei Zhang, Gaoliang Peng, Chuanhao Li, Yuanhang Chen, and Zhujun Zhang
- **Published in:** Sensors, MDPI [doi:10.3390/s17020425](https://doi.org/10.3390/s17020425).

## Input Requirements
- **Input Length:** Strictly 2048 data points
- **Input Channels:** 1 (single-channel signal)
- **Tensor Shape:** `[batch_size, 1, 2048]`

**Note:** The model is specifically designed to process input signals of exactly 2048 points. Inputs with different lengths will cause errors.

## Model Architecture

The model is built using **PyTorch** and consists of the following components:

### FilterBlock
The `FilterBlock` class represents a convolutional block in the WDCNN model. It includes:
- 1D Convolutional Layer (`nn.Conv1d`)
- Batch Normalization (`nn.BatchNorm1d`)
- ReLU Activation (`nn.ReLU`)
- Max Pooling (`nn.MaxPool1d`)

Each `FilterBlock` serves as a feature extractor from raw input signals.

### Classifier
The `Classifier` class takes the output from the last convolutional layer and applies:
- Flatten Layer (`nn.Flatten`)
- Linear Layers (`nn.Linear`) for classification into the final `total_classes`

### Net
The `Net` class represents the complete WDCNN architecture, which is composed of multiple `FilterBlock` layers followed by the `Classifier` block:
- 5 FilterBlocks with increasing feature map sizes.
- A final fully connected layer that performs classification.

## Usage
```python
import torch
from wdcnn_model import Net

# Initialize the model
model = Net(in_channels=1, total_classes=10)

# Create a sample input tensor
# IMPORTANT: Input tensor MUST have length 2048
x = torch.randn(batch_size, 1, 2048)

# Forward pass
output = model(x)
```

## Model Layers
- **First FilterBlock:** 
  - Input: 1 channel
  - Output: 16 channels
  - Kernel Size: 64
  - Stride: 16
  - Padding: 24

- Subsequent FilterBlocks progressively increase feature complexity
- Final Classifier reduces features to class probabilities

## Training Considerations
- Suitable for raw vibration signal fault diagnosis
- Demonstrates good performance with noisy signals
- Adaptable to different domain scenarios
- Requires input signals of exactly 2048 points

## Customization
- Adjust `in_channels` for different input signal types
- Modify `total_classes` based on your specific classification task
- **Important:** Maintain input length at 2048 points

## Dependencies
- PyTorch
- Python 3.x

## License
This network implementation is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributions
Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements
Original research by Zhang et al., published in Sensors (MDPI)