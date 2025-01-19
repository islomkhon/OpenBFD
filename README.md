# OpenBFD: Open-Source Bearing Fault Diagnosis Tool

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8.10%2B-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1%2B-redcpu)](https://pytorch.org)

OpenBFD is an open-source deep learning tool for bearing fault diagnosis built with PyTorch. It provides comprehensive solutions for detecting and classifying bearing faults using raw vibration data and advanced deep learning techniques.

## Authors

![Islomkhon Nizomkhonov](https://github.com/islomkhon/OpenBFD/blob/main/Images/Authors/IslomkhonNizomkhonov.png)
### Islomkhon Nizomkhonov (小伊)

Islomkhon Nizomkhonov was born in 1994 in Taskent, Uzbekistan. He is a senior software engineer with over 6 years of experience in the field of machine learning and predictive maintenance. He has worked with several industries and big companies like [XCMG (徐工集团)](https://www.xcmg.com) and [Uztelecom (Uzbekistan National Telecom Provider)](https://uztelecom.uz), providing solutions using deep learning techniques. His research primarily focuses on using neural networks for fault diagnosis in industrial IoT systems.

- **Education**: Ms.D. in Computer Science, China University of Mining and Technology
- **Key Skills**: Software Engineering, Deep Learning, Machine Learning, Data Science, IoT, Predictive Maintenance.

![Mr. ZhangBo](https://github.com/islomkhon/OpenBFD/blob/main/Images/Authors/ZhangBo.png)
### Mr. ZhangBo (张博)

Mr. ZhangBo is Islomkhon Nizomkhonov's master's degree supervisor for his research project on distributed deep neural networks, which Islomkhon conducted at China University of Mining and Technology.

Mr. ZhangBo, was born in 1981 in Xuzhou, Jiangsu Province. He graduated from the Institute of Computing Technology, Chinese Academy of Sciences, with a Ph.D., and completed his postdoctoral research at China University of Mining and Technology. In 2018, he was a visiting scholar at the Institute of Data Science (IDS), National University of Singapore (NUS). He is currently an associate professor at the School of Computer Science, China University of Mining and Technology.

ZhangBo is engaged in theoretical and applied research in artificial intelligence, machine learning, pattern recognition, mechanical equipment condition monitoring, and cloud computing. He has led one project funded by the Jiangsu Provincial Basic Research Program (Natural Science Foundation) under the Youth Fund, one project under the 62nd batch of the China Postdoctoral Science Foundation, and one project under the Jiangsu Postdoctoral Research Funding Program. In recent years, he has applied for/been granted 3 invention patents, published 7 papers in SCI journals, and 3 papers in EI journals. He serves as a reviewer for international journals such as *IEEE Transactions on Industrial Electronics* and *Journal of Computer Science* as well as other domestic and international journals. 

In December 2016, the project titled "Research and Application of Belt Conveyor Monitoring and Fire Warning Technology" led by ZhangBo won the second prize of the "China Coal Industry Science and Technology Award" from the China National Coal Association and the China Coal Society. The project was successfully applied to the belt conveyors at the underground mines of the Zhongmei Pingshuo Group, effectively addressing the challenges of condition monitoring, fault diagnosis and forecasting, as well as fire warning for belt conveyors under operational conditions in the mining environment.

- **Education**: Associate Prof. in Artificial Intelligence and Cloud Computing, China University of Mining and Technology
- **Key Skills**: AI, Cloud Computing, Industrial Automation, Mechanical Equipment Condition Monitoring

## Features

- Deep learning models specifically designed for bearing fault detection
- Real-time vibration data analysis
- Pre-trained models for common bearing fault types
- Support for custom dataset training
- Comprehensive visualization tools for fault analysis

## Installation

```bash
git clone https://github.com/islomkhon/OpenBFD.git
cd OpenBFD
pip install -r requirements.txt
```

## Quick Start

OpenBFD/
│
├── Authors/
│   ├── IslomkhonNizomkhonov.png
│   ├── ZhangBo.png
│   └── README.md
│
├── Datasets/
│   ├── DatasetName/
│   │   ├── RawData/
│   │   ├── preprocessor.py
│   │   └── README.md
│   │
│   └── README.md
│  
├── Experiments/
│   ├── ModelInputOutputType/
│   │   └── ModelName/
│   │       └── DatasetName/
│   │           └── ExperimentName.py
│   │
│   └── README.md
│
├── Models/
│   ├── ModelInputOutputType/
│   │   └── ModelName/
│   │       ├── Model.py
│   │       ├── ResearchPaper.pdf
│   │       └── README.md
│   │           
│   └── README.md
│
├── TrainingLoops/
│   ├── ModelInputOutputType/
│   │   └── train.py
│   │           
│   └── README.md
│
├── README.md 
└── requirements.txt


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
@software{openbfd,
  title = {OpenBFD: An Open-Source Tool for Bearing Fault Diagnosis},
  author = {Islomkhon Nizomkhonov (小伊) , Mr. ZhangBo (张博)},
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
- Email: islomkhon@mail.com
- Instagram: [islomkhonnizomkhonov](https://instagram.com/islomkhonnizomkhonov)

## Support the project

If you find this project helpful, you can support OpenBFD by making donations via:

<table>
  <tr>
    <th>WeChat</th>
    <th>Alipay</th>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/islomkhon/OpenBFD/blob/main/Images/QR/wechat.jpg" alt="WeChat QR Code" width="280"/>
    </td>
    <td>
      <img src="https://github.com/islomkhon/OpenBFD/blob/main/Images/QR/alipay.jpg" alt="Alipay QR Code" width="280"/>
    </td>
  </tr>
</table>