import os
from scipy import io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split

class CWRUDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float) 
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        return sample, label

def read_raw(data_path, data_points, data_transformer = None):
   """Read the .mat files from the specified path and save them as a dictionary.
   data_path is the file path, which can contain multiple files.
   The return value is data, in the format of a dictionary:

   The key in data is the corresponding filename.

   The value in data is the array with corresponding vibration signal data provided in data_points.

   The order of values in data_points affects the order in data value.

   The transform shoud be a one param lambda function used for transformation of the Data
   """
   
   # Get absolute path by joining current file's directory with provided path
   data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_path)

   # Get list of all files in directory
   filenames = os.listdir(data_path) 
   
   # Initialize empty dictionary to store results
   data = {} 
   
   # Process each file
   for f in filenames:
       # Create full file path
       filePath = os.path.join(data_path, f) 
       
       # Load .mat file
       file = io.loadmat(filePath) 
       
       # Get all keys in the .mat file
       keys = file.keys()
       
       # Initialize dictionary for this file
       data[f] = {}
       
       # Process each key in the .mat file
       for key in keys:
           # Check each data point we're looking for
           for index, point in enumerate(data_points):
               # If current key contains the data point name
               if point in key:
                   # Apply transformer function if provided
                   if data_transformer:
                       # Transform and flatten the data
                       data[f][index] = data_transformer(file[key].ravel())
                   else: 
                       # Just flatten the data without transformation
                       data[f][index] = file[key].ravel() 

   return data


def argumentation(data, length=864, number=1000, enc=True, enc_step=28):
   """Randomly slice data into individual samples with optional data augmentation
   Args:
       data: Dictionary containing input data arrays
       length: Length of each sliced sample (default: 864)
       number: Number of samples to generate per key (default: 1000) 
       enc: Enable data sub augmentation (default: True)
       enc_step: Step size for sub augmented samples (default: 28)
   Returns:
       Dictionary of numpy arrays containing sliced samples"""
       
   train_samples = {}  # Output dictionary storing sliced samples
   keys = data.keys()  # Get keys from input data dictionary
   
   for key in keys:
       train_data = []  # Store samples for current key
       number_of_inputs = len(data[key])  # Number of input arrays per key
       total_length = len(data[key][0])  # Length of input arrays
       train_num = number  # Total samples to generate
       
       if enc:  # Data sub augmentation enabled
           enc_times = length // enc_step  # Number of augmented samples per window
           steps = 0  # Track total samples generated
           
           for j in range(train_num):
               label = False  # Flag to break nested loops
               # Random start within valid range for sub augmented sampling
               start_index = np.random.randint(low=0, high=total_length - 2 * length)
               
               for h in range(enc_times):
                   temp = []
                   # Slice and store all input arrays
                   for i in range(number_of_inputs):
                       temp.append(data[key][i][start_index:start_index + length])
                   train_data.append(temp)
                   steps += 1
                   start_index = start_index + enc_step  # Slide window for next sub augmented sample
                   
                   if steps == train_num:
                       label = True
                       break
               if label:
                   break
                   
       else:  # No data augmentation
           for j in range(train_num):
               # Random start within valid range
               start_index = np.random.randint(low=0, high=total_length - 2 * length)
               temp = []
               # Slice and store all input arrays
               for i in range(number_of_inputs):
                   temp.append(data[key][i][start_index:start_index + length])
               train_data.append(temp)
               
       train_samples[key] = np.array(train_data)  # Convert samples to numpy array
               
   return train_samples

import numpy as np

def add_label(data):
    """Assigns labels to data samples with values 0-9 for different categories:
    - normal
    - B007~B021
    - IR007~IR021
    - OR007~OR021

    Args:
        data (dict): Input sample data dictionary
        
    Returns:
        tuple: (X, Y) where:
            - X (np.ndarray): Sample data array
            - Y (np.ndarray): Labels array with values 0-9 representing 10 classes
            
    Raises:
        TypeError: If input is not a dictionary
        ValueError: If input dictionary is empty
    """
    if not isinstance(data, dict):
        raise TypeError("Input 'data' must be a dictionary")
    if not data:
        raise ValueError("Input dictionary cannot be empty")
    
    X = []
    Y = []
    
    for label, (key, samples) in enumerate(data.items()):
        if samples.size == 0:  # Skip empty sample lists
            continue
        X.extend(samples)
        Y.extend([label] * len(samples))
    
    if not X:
        raise ValueError("No valid samples found in input data")
    
    return np.array(X), np.array(Y)

def get_torch_dataloder(X, Y, batch_size=32, shuffle=True):
    """
    Create a PyTorch DataLoader from arrays of samples and labels
    
    Args:
        X (np.ndarray): Sample data
        Y (np.ndarray): Labels
        batch_size (int): Size of each batch
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        DataLoader: PyTorch DataLoader object
    """
    dataset = CWRUDataset(X, Y)
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset_loader

def train_test_spliter(X,Y,test_size=0.3,shuffle=True):
    """
    Split data into training and test sets
    
    Args:
        X (np.ndarray): Sample data
        Y (np.ndarray): Labels
        test_size (float): Proportion of data to use for testing (0.0 to 1.0)
        shuffle (bool): Whether to shuffle the data before splitting
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=test_size,shuffle=shuffle, stratify=Y)
    return X_train, X_test, y_train, y_test

def one_hot(data_Y):
    """One-hot encoding function
    For 10 classes, using 10 binary digits (0-1) to represent
    Parameters:
        data_Y: labels ranging from 0-9
    Returns:
        Y: generated one-hot encoding, shape is (num_samples, 10)
    """
    # Create a OneHotEncoder object that stores the encoding mapping information
    Encoder = preprocessing.OneHotEncoder()
    
    # Reshape labels into column vector
    data_Y = data_Y.reshape((-1, 1))
    
    # Fit encoder to determine unique values in data_Y
    # First binary position represents original label 0
    # Second binary position represents original label 1, and so on
    Encoder.fit(data_Y)
    
    # Transform data_Y using the fitted encoding pattern
    # Convert sparse matrix to dense array
    Y = Encoder.transform(data_Y).toarray()
    
    # Convert Y to numpy array with int32 dtype
    Y = np.asarray(Y, dtype=np.int32)
    
    return Y