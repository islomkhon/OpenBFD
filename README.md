# OpenBFD: Open-Source Bearing Fault Diagnosis Tool

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8.10%2B-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1%2B-redcpu)](https://pytorch.org)

OpenBFD is an open-source deep learning tool for bearing fault diagnosis built with PyTorch. It provides a comprehensive solution for detecting and classifying bearing faults using vibration analysis and advanced deep learning techniques.

## Features

- Deep learning models specifically designed for bearing fault detection
- Real-time vibration data analysis
- Pre-trained models for common bearing fault types
- Support for custom dataset training
- Comprehensive visualization tools for fault analysis
- Easy-to-use API for integration into existing systems

## Installation

```bash
pip install openbfd
```

Or install from source:

```bash
git clone https://github.com/yourusername/OpenBFD.git
cd OpenBFD
pip install -e .
```

## Quick Start

```python
import openbfd

# Load a pre-trained model
model = openbfd.load_model('bearing_fault_detector')

# Analyze vibration data
data = openbfd.load_data('vibration_data.csv')
results = model.predict(data)

# Visualize results
openbfd.visualize(results)
```

## Documentation

For detailed documentation, visit our [documentation page](link-to-your-docs).

### Examples

- [Basic Fault Detection](examples/basic_fault_detection.py)
- [Custom Model Training](examples/custom_training.py)
- [Real-time Analysis](examples/realtime_analysis.py)
- [Data Preprocessing](examples/preprocessing.py)

## Supported Fault Types

- Inner Race Faults
- Outer Race Faults
- Ball Faults
- Cage Faults
- Combination Faults

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to:

- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## Model Architecture

OpenBFD uses a deep learning architecture specifically designed for bearing fault diagnosis:

- Input Layer: Raw vibration signals
- Feature Extraction: Convolutional layers
- Classification: Fully connected layers
- Output: Fault type and severity

## Citation

If you use OpenBFD in your research, please cite:

```bibtex
@software{openbfd2025,
  title = {OpenBFD: An Open-Source Tool for Bearing Fault Diagnosis},
  author = {Islomkhon Nizomkhonov (伊斯罗贡)},
  year = {2025},
  url = {https://github.com/islomkhon/OpenBFD}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Contributors and maintainers
- PyTorch community
- Open-source vibration analysis tools that inspired this project

## Contact

- GitHub Issues: For bug reports and feature requests
- Email: your.email@example.com
- Twitter: [@YourTwitterHandle](https://twitter.com/YourTwitterHandle)