# CWRU Bearing Dataset Processing Tool

[DataHelper.py](/Datasets/CWRU/DataHelper.py) script provides tools for processing and handling the Case Western Reserve University (CWRU) Bearing Dataset. It includes functionality for reading .mat files, data augmentation, creating PyTorch datasets, and managing data loaders.

## Features

- Read and process .mat files from the CWRU Bearing Dataset
- Data augmentation with sliding window approach
- Automatic labeling for bearing fault categories
- PyTorch Dataset and DataLoader implementation
- Train/test split functionality

## Requirements

```python
numpy
scipy
torch
scikit-learn
```

## Installation

1. Clone this repository
2. Install the required packages by executing the command below in the root of the OpenBFD:
```bash
pip install -e .
```

## Usage

### 1. Reading Raw Data

```python
raw_data = read_raw(
    data_path="RawData/0HP",  # Path to your .mat files
    data_points=['DE'],       # Data points to extract (e.g., 'DE', 'FE', 'BA') DE=DeviceEnd,FE=FanEnd. Be carefull reading BA from "normal" data file, because it does not contain BA
    data_transformer=None     # Optional data transformer function
)
```

### 2. Data Augmentation

```python
augmented_data = argumentation(
    data=raw_data,
    length=864,      # Length of each sample
    number=1000,     # Number of samples to generate per key
    enc=True,        # Enable sliding window augmentation
    enc_step=28      # Step size for sliding window
)
```

### 3. Adding Labels

```python
X, Y = add_label(augmented_data)
# Labels correspond to:
# - normal
# - B007~B021 (Ball fault)
# - IR007~IR021 (Inner Race fault)
# - OR007~OR021 (Outer Race fault)
```

### 4. Creating Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_spliter(
    X, Y,
    test_size=0.3,
    shuffle=True
)
```

### 5. Creating PyTorch DataLoader

```python
train_loader = get_torch_dataloder(
    X_train, y_train,
    batch_size=32,
    shuffle=True
)
```

### Complete Example

```python
# Read raw data
raw_data = read_raw("RawData/0HP", ['DE'])

# Augment data
augmented_data = argumentation(raw_data, length=864, number=1000)

# Add labels
X, Y = add_label(augmented_data)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_spliter(X, Y, test_size=0.3)

# Create dataloaders
train_loader = get_torch_dataloder(X_train, y_train, batch_size=32)
test_loader = get_torch_dataloder(X_test, y_test, batch_size=32)
```

## Functions Documentation

### read_raw(data_path, data_points, data_transformer=None)
Reads .mat files from the specified path and returns a dictionary of data.
- `data_path`: Path to the directory containing .mat files
- `data_points`: List of data points to extract
- `data_transformer`: Optional transformation function

### argumentation(data, length=864, number=1000, enc=True, enc_step=28)
Performs data augmentation using a sliding window approach.
- `data`: Input dictionary of data
- `length`: Length of each sample
- `number`: Number of samples to generate per key
- `enc`: Enable sliding window augmentation
- `enc_step`: Step size for sliding window

### add_label(data)
Assigns numerical labels to data samples for classification.
- `data`: Dictionary of augmented data
- Returns: Tuple of (X, Y) arrays

### train_test_spliter(X, Y, test_size=0.3, shuffle=True)
Splits data into training and test sets.
- `X`: Sample data
- `Y`: Labels
- `test_size`: Proportion of data for testing
- `shuffle`: Whether to shuffle before splitting

### get_torch_dataloder(X, Y, batch_size=32, shuffle=True)
Creates a PyTorch DataLoader for the dataset.
- `X`: Sample data
- `Y`: Labels
- `batch_size`: Number of samples per batch
- `shuffle`: Whether to shuffle the data

## Dataset Structure

The `CWRUDataset` class inherits from `torch.utils.data.Dataset` and provides:
- Automatic conversion to PyTorch tensors
- Proper data types (float32 for data, int64 for labels)
- Standard dataset interfaces for PyTorch

## Notes

- The script assumes a specific directory structure for the CWRU dataset
- Labels are assigned automatically based on file names
- Data augmentation uses a sliding window approach for generating more samples
- All tensors are automatically converted to the appropriate dtype (float32 for data, int64 for labels)

## License

[MIT](LICENSE)