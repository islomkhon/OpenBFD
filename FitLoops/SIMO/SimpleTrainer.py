import torch
from torch import nn
from matplotlib_inline import backend_inline
from IPython import display
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.nn.functional as F

class Animator:
    """Dynamically plot training curves"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:', 'g:'), nrows=1, ncols=1,
                 figsize=(4.5, 3.5)):
        # Use incremental plotting for curves
        if legend is None:
            legend=[]
        backend_inline.set_matplotlib_formats('svg')  # Set curves to display in SVG format
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use lambda function to capture parameters
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim,
                                            xscale, yscale, legend)
        # X stores x-axis data, Y stores y-axis data, fmts sets line styles and colors
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y, clear=True):
        """Add data points to the plot, x is for x-axis, y is for y-axis"""
        if not hasattr(y, "__len__"):
            y = [y]  # If not iterable, convert to list
        n = len(y)  # Only add one set of data points at a time, n is number of data series
        if not hasattr(x, "__len__"):
            x = [x] * n  # Generate a list of n x elements
        if not self.X:
            self.X = [[] for _ in range(n)]  # Create n groups of x-coordinates, one for each line
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()  # Clear the current axes
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        plt.draw()
        # plt.pause(0.001)  # Commented out pause
        display.display(self.fig)
        if clear == True:
            display.clear_output(wait=True)  # Clear the output for dynamic updates

    def show(self):
        """Display the current figure"""
        display.display(self.fig)
        
class Accumulator:
    """对n个变量进行累加"""
    def __init__(self, n):
        """n表示要累加的变量的个数"""
        self.data = [0.0] * n #为n个变量创建列表，数据类型为float32
    def add(self, *args):
        """
        :param args: 允许传入多个参数，将传入的多个参数作为元组，传入的参数个数必须与data中的变量个数n相同
        args中就保存了新的需要增加的变量的值
        :return:没有返回值
        """
        self.data = [a + float(b) for a, b in zip(self.data, args)] #a是原来变量的旧值，b是需要增加的值，a + b就得到增加后总的值

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        """这个函数的作用是实现该类的对象能够使用索引访问数据
            也就是对这个类的对象使用索引的方式时就会调用这个函数，索引值作为参数idx"""
        return self.data[idx]

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Configure plot parameters"""
    axes.set_xlabel(xlabel)          # Set x-axis label
    axes.set_ylabel(ylabel)          # Set y-axis label
    axes.set_xscale(xscale)         # Set x-axis scale (linear, log, etc.)
    axes.set_yscale(yscale)         # Set y-axis scale (linear, log, etc.)
    axes.set_xlim(xlim)             # Set x-axis limits
    axes.set_ylim(ylim)             # Set y-axis limits
    if legend:
        axes.legend(legend)          # Add legend if provided
    axes.grid()                      # Add grid to the plot

def accuracy(y_hat, y):
    """计算一个batch的精度"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1) #获取预测值最大的标签所在的位置
    y_true = y.clone() #必须克隆，否则使用y = y.argmax会修改传入的原值，因为python参数传递是引用
    y_true = y_true.argmax(axis=1) #在onehot标签中，最大值所在的位置就是标签1所在位置
    cmp = y_hat.type(y_true.dtype) == y_true #与真实标签对比
    return float(cmp.type(y.dtype).sum()) #返回预测精度

def test(net, data_iter, output_idx=None, device=None):
    """评估模型指定输出的准确率"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)  # 用于累加正确预测数和总样本数
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device) # 确保 y 是一个列表
            y_hat = net(X)  # 获取模型所有输出
            if output_idx is not None:
                y_hat = y_hat[output_idx]  # 获取指定输出
            else:
                raise ValueError("For multi-output models, output_idx must be specified.")
            metric.add(accuracy(y_hat, y), len(y))  # 累加正确数和样本总数
    return metric[0] / metric[1]  # 返回准确率


def train(net, train_iter, test_iter, num_epochs, loss, optimizer, device):
    print(f'training on {device}')
    net.to(device)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                       legend=['train loss', 'train acc', 'test acc'])
    num_batches = len(train_iter)
    
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                           (train_l, train_acc, None))
        test_acc = test(net, test_iter, device)
        animator.add(epoch + 1, (None, None, test_acc))
    
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    
def train(net, train_iter, test_iter, num_epochs, loss_fn, optimizer, device):
    """
    针对多输出模型的训练
    """
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv1d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print("training on ", device)
    net.to(device)
    
    animator = Animator(xlabel='epoch', 
                         xlim=[1, num_epochs],
                         legend=['train loss'] + [f'train acc {net.output_layers[i]}' for i in range(len(net.output_layers))] +
                                                 [f'test acc {net.output_layers[i]}' for i in range(len(net.output_layers))]  )
    
    for epoch in range(num_epochs):
        metric = Accumulator(2 + len(net.output_layers))  # 2: loss 和总样本数 + 每个输出的准确率
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)  # y是多输出的标签
            y_hat = net(X)  # 获取多输出预测
            total_loss = 0
            for i, prediction in enumerate(y_hat):# 对每个输出计算损失
                loss = loss_fn(prediction, y)
                total_loss += loss
            total_loss.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(total_loss * X.shape[0], X.shape[0], *[accuracy(y_hat[j], y) for j in range(len(y_hat))])
        train_l = metric[0] / metric[1]
        train_acc = [metric[j + 2] / metric[1] for j in range(len(net.output_layers))]
        test_acc = [test(net, test_iter, j) for j in range(len(net.output_layers))]
        animator.add(epoch + 1, [train_l] + train_acc + test_acc, epoch != num_epochs)
    print(f'loss {train_l:.3f}, train acc {[round(a, 3) for a in train_acc]},'
          f' test acc {[round(a, 3) for a in test_acc]}')
    plt.show()
    
def create_confusion_matrix(model, data_iterator, num_classes):
    """
    Creates and plots confusion matrix heatmaps for a multi-output model.
    Handles one-hot encoded labels.
    
    Args:
        model: PyTorch model with multiple outputs
        data_iterator: Iterator yielding (inputs, targets) pairs
        num_classes: Number of classes in the dataset
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize lists to store predictions and true labels for both classifiers
    edge_preds = []
    cloud_preds = []
    true_labels = []
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        for inputs, labels in data_iterator:
            # Move inputs to the same device as the model
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            
            # Get predictions from both classifiers
            edge_output, cloud_output = model(inputs)
            
            # Get predicted classes
            _, edge_pred = torch.max(edge_output, 1)
            _, cloud_pred = torch.max(cloud_output, 1)
            
            # Convert one-hot encoded labels to class indices
            if labels.dim() > 1 and labels.size(1) > 1:
                labels = torch.argmax(labels, dim=1)
            
            # Store predictions and labels
            edge_preds.extend(edge_pred.cpu().numpy())
            cloud_preds.extend(cloud_pred.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Convert lists to numpy arrays
    edge_preds = np.array(edge_preds)
    cloud_preds = np.array(cloud_preds)
    true_labels = np.array(true_labels)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # Calculate and plot Edge classifier confusion matrix
    edge_cm = confusion_matrix(
        true_labels, 
        edge_preds, 
        labels=range(num_classes)
    )
    
    sns.heatmap(
        edge_cm, 
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=range(num_classes),
        yticklabels=range(num_classes),
        ax=ax1
    )
    ax1.set_title('Edge Classifier Confusion Matrix')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # Calculate and plot Cloud classifier confusion matrix
    cloud_cm = confusion_matrix(
        true_labels, 
        cloud_preds, 
        labels=range(num_classes)
    )
    
    sns.heatmap(
        cloud_cm, 
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=range(num_classes),
        yticklabels=range(num_classes),
        ax=ax2
    )
    ax2.set_title('Cloud Classifier Confusion Matrix')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    plt.tight_layout()
    return fig, (edge_cm, cloud_cm)

def analyze_predictions(model, data_loader, device='cuda'):
    """
    Calculate the max and min probabilities of edge predictions and min probabilities 
    of cloud predictions for each predicted class.
    
    Args:
        model: The neural network model with edge and cloud components
        data_loader: DataLoader containing the test/validation data
        device: Device to run the computations on ('cuda' or 'cpu')
        
    Returns:
        dict: A dictionary containing max and min probabilities for each class
    """
    model.eval()
    model = model.to(device)
    
    # Initialize dictionaries to store results
    results = {
        'edge_max_probs': {},
        'edge_min_probs': {},
        'cloud_min_probs': {}
    }
    
    # Count of samples per class for each component
    edge_class_counts = {}
    cloud_class_counts = {}
    
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            
            # Forward pass
            edge_outputs, cloud_outputs = model(data)
            
            # Apply softmax to get probabilities
            edge_probs = F.softmax(edge_outputs, dim=1)
            cloud_probs = F.softmax(cloud_outputs, dim=1)
            
            # Get predicted classes
            edge_preds = torch.argmax(edge_probs, dim=1)
            cloud_preds = torch.argmax(cloud_probs, dim=1)
            
            # Get the probability of the predicted class for each sample
            for i in range(len(data)):
                # Edge predictions
                edge_pred_class = edge_preds[i].item()
                edge_pred_prob = edge_probs[i, edge_pred_class].item()
                
                # Cloud predictions
                cloud_pred_class = cloud_preds[i].item()
                cloud_pred_prob = cloud_probs[i, cloud_pred_class].item()
                
                # Update edge max probabilities
                if edge_pred_class not in results['edge_max_probs'] or edge_pred_prob > results['edge_max_probs'][edge_pred_class]:
                    results['edge_max_probs'][edge_pred_class] = edge_pred_prob
                
                # Update edge min probabilities
                if edge_pred_class not in results['edge_min_probs'] or edge_pred_prob < results['edge_min_probs'][edge_pred_class]:
                    results['edge_min_probs'][edge_pred_class] = edge_pred_prob
                
                # Update cloud min probabilities
                if cloud_pred_class not in results['cloud_min_probs'] or cloud_pred_prob < results['cloud_min_probs'][cloud_pred_class]:
                    results['cloud_min_probs'][cloud_pred_class] = cloud_pred_prob
                
                # Update class counts
                edge_class_counts[edge_pred_class] = edge_class_counts.get(edge_pred_class, 0) + 1
                cloud_class_counts[cloud_pred_class] = cloud_class_counts.get(cloud_pred_class, 0) + 1
    
    # Print summary statistics
    print("Summary Statistics:")
    print("-" * 50)
    
    print("\nEdge Predictions:")
    print("Class\tSamples\tMax Probability\tMin Probability")
    for cls in sorted(results['edge_max_probs'].keys()):
        count = edge_class_counts.get(cls, 0)
        max_prob = results['edge_max_probs'].get(cls, 0)
        min_prob = results['edge_min_probs'].get(cls, 0)
        print(f"{cls}\t{count}\t{max_prob:.4f}\t\t{min_prob:.4f}")
    
    print("\nCloud Predictions:")
    print("Class\tSamples\tMin Probability")
    for cls in sorted(results['cloud_min_probs'].keys()):
        count = cloud_class_counts.get(cls, 0)
        min_prob = results['cloud_min_probs'].get(cls, 0)
        print(f"{cls}\t{count}\t{min_prob:.4f}")
    
    return results

# Example usage:
# analyze_predictions(model, test_loader, device='cuda')
import matplotlib.pyplot as plt
def visualize_probability_distribution(model, data_loader, device='cuda'):
    """
    Visualize the probability distributions for edge and cloud predictions.
    
    Args:
        model: The neural network model with edge and cloud components
        data_loader: DataLoader containing the test/validation data
        device: Device to run the computations on ('cuda' or 'cpu')
    """
    
    
    model.eval()
    model = model.to(device)
    
    edge_probs_by_class = {}
    cloud_probs_by_class = {}
    
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            
            # Forward pass
            edge_outputs, cloud_outputs = model(data)
            
            # Apply softmax to get probabilities
            edge_probs = F.softmax(edge_outputs, dim=1)
            cloud_probs = F.softmax(cloud_outputs, dim=1)
            
            # Get predicted classes
            edge_preds = torch.argmax(edge_probs, dim=1)
            cloud_preds = torch.argmax(cloud_probs, dim=1)
            
            # Collect probabilities by predicted class
            for i in range(len(data)):
                edge_pred_class = edge_preds[i].item()
                edge_pred_prob = edge_probs[i, edge_pred_class].item()
                
                cloud_pred_class = cloud_preds[i].item()
                cloud_pred_prob = cloud_probs[i, cloud_pred_class].item()
                
                if edge_pred_class not in edge_probs_by_class:
                    edge_probs_by_class[edge_pred_class] = []
                edge_probs_by_class[edge_pred_class].append(edge_pred_prob)
                
                if cloud_pred_class not in cloud_probs_by_class:
                    cloud_probs_by_class[cloud_pred_class] = []
                cloud_probs_by_class[cloud_pred_class].append(cloud_pred_prob)
    

    num_classes = len(edge_probs_by_class)
    cols = 2  # Two columns
    rows = (num_classes + 1) // cols  # Determine number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))  # Adjust figure size
    axes = axes.flatten()  # Flatten the 2D array to 1D for easy iteration

    for i, cls in enumerate(sorted(edge_probs_by_class.keys())):
        ax = axes[i]
        ax.hist(edge_probs_by_class[cls], alpha=0.5, bins=80, label='edge')
        ax.hist(cloud_probs_by_class[cls], alpha=0.5, bins=80, label='cloud')
        ax.set_title(cls)
        ax.set_xlabel('Probability')
        ax.set_ylabel('Samples')
        ax.legend()
        # ax.grid(True)

    # Hide any unused subplots if the number of classes is not an exact multiple of cols
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()


def visualize_probability_distribution2(model, data_loader, device='cuda'):
    """
    Visualize the probability distributions for edge and cloud predictions.
    
    Args:
        model: The neural network model with edge and cloud components
        data_loader: DataLoader containing the test/validation data
        device: Device to run the computations on ('cuda' or 'cpu')
    """
    
    model.eval()
    model = model.to(device)
    
    edge_probs_by_class = {}
    cloud_probs_by_class = {}
    
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            
            # Forward pass
            edge_outputs, cloud_outputs = model(data)
            
            # Apply softmax to get probabilities
            edge_probs = F.softmax(edge_outputs, dim=1)
            cloud_probs = F.softmax(cloud_outputs, dim=1)
            
            # Get predicted classes
            edge_preds = torch.argmax(edge_probs, dim=1)
            cloud_preds = torch.argmax(cloud_probs, dim=1)
            
            # Collect probabilities by predicted class
            for i in range(len(data)):
                edge_pred_class = edge_preds[i].item()
                edge_pred_prob = edge_probs[i, edge_pred_class].item()
                
                cloud_pred_class = cloud_preds[i].item()
                cloud_pred_prob = cloud_probs[i, cloud_pred_class].item()
                
                if edge_pred_class not in edge_probs_by_class:
                    edge_probs_by_class[edge_pred_class] = []
                edge_probs_by_class[edge_pred_class].append(edge_pred_prob)
                
                if cloud_pred_class not in cloud_probs_by_class:
                    cloud_probs_by_class[cloud_pred_class] = []
                cloud_probs_by_class[cloud_pred_class].append(cloud_pred_prob)
    

    num_classes = len(edge_probs_by_class)
    cols = 2  # Two columns
    rows = (num_classes + 1) // cols  # Determine number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))  # Adjust figure size
    axes = axes.flatten()  # Flatten the 2D array to 1D for easy iteration

    for i, cls in enumerate(sorted(edge_probs_by_class.keys())):
        ax = axes[i]

        edge_probs = edge_probs_by_class[cls]
        cloud_probs = cloud_probs_by_class[cls]  # Get cloud probs, default empty list

        # Compute min/max for edge and min for cloud
        edge_min = min(edge_probs)
        edge_max = max(edge_probs)
        cloud_min = min(cloud_probs)

        # Plot histograms with labels including min/max values
        ax.hist(cloud_probs, alpha=0.5, bins=80, label=f'Cloud (Min: {cloud_min:.4f})')
        ax.hist(edge_probs, alpha=0.5, bins=80, label=f'Edge (Min: {edge_min:.4f}, Max: {edge_max:.4f})')
        
        
        ax.set_title(f'Class {cls}')
        ax.set_xlabel('Probability')
        ax.set_ylabel('Samples')
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots if the number of classes is not an exact multiple of cols
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()


def visualize_probability_distribution3(model, data_loader, device='cuda', k=2):
    """
    Visualize the probability distributions for edge and cloud predictions,
    with thresholds based on standard deviation.
    
    Args:
        model: The neural network model with edge and cloud components
        data_loader: DataLoader containing the test/validation data
        device: Device to run the computations on ('cuda' or 'cpu')
        k: Number of standard deviations to calculate the threshold range
    """
    
    model.eval()
    model = model.to(device)
    
    edge_probs_by_class = {}
    cloud_probs_by_class = {}
    
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            
            # Forward pass
            edge_outputs, cloud_outputs = model(data)
            
            # Apply softmax to get probabilities
            edge_probs = F.softmax(edge_outputs, dim=1)
            cloud_probs = F.softmax(cloud_outputs, dim=1)
            
            # Get predicted classes
            edge_preds = torch.argmax(edge_probs, dim=1)
            cloud_preds = torch.argmax(cloud_probs, dim=1)
            
            # Collect probabilities by predicted class
            for i in range(len(data)):
                edge_pred_class = edge_preds[i].item()
                edge_pred_prob = edge_probs[i, edge_pred_class].item()
                
                cloud_pred_class = cloud_preds[i].item()
                cloud_pred_prob = cloud_probs[i, cloud_pred_class].item()
                
                if edge_pred_class not in edge_probs_by_class:
                    edge_probs_by_class[edge_pred_class] = []
                edge_probs_by_class[edge_pred_class].append(edge_pred_prob)
                
                if cloud_pred_class not in cloud_probs_by_class:
                    cloud_probs_by_class[cloud_pred_class] = []
                cloud_probs_by_class[cloud_pred_class].append(cloud_pred_prob)
    

    num_classes = len(edge_probs_by_class)
    cols = 2  # Two columns
    rows = (num_classes + 1) // cols  # Determine number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))  # Adjust figure size
    axes = axes.flatten()  # Flatten the 2D array to 1D for easy iteration

    for i, cls in enumerate(sorted(edge_probs_by_class.keys())):
        ax = axes[i]

        edge_probs = edge_probs_by_class[cls]
        cloud_probs = cloud_probs_by_class[cls]  # Get cloud probs, default empty list

        # Compute mean and standard deviation for edge and cloud
        edge_mean = np.mean(edge_probs)
        edge_std = np.std(edge_probs)
        cloud_mean = np.mean(cloud_probs)
        cloud_std = np.std(cloud_probs)

        # Calculate min and max thresholds based on standard deviations
        edge_min_threshold = edge_mean - k * edge_std
        edge_max_threshold = edge_mean + k * edge_std
        cloud_min_threshold = cloud_mean - k * cloud_std
        cloud_max_threshold = cloud_mean + k * cloud_std
        
        # Plot histograms with labels including min/max threshold values
        ax.hist(cloud_probs, alpha=0.5, bins=80, label=f'Cloud (Min: {min(cloud_probs):.4f}, Max: {max(cloud_probs):.4f})')
        ax.hist(edge_probs, alpha=0.5, bins=80, label=f'Edge (Min: {min(edge_probs):.4f}, Max: {max(edge_probs):.4f})')

        # Draw lines for the thresholds on the plot
        ax.axvline(edge_min_threshold, color='r', linestyle='--', label=f'Edge Min Threshold ({edge_min_threshold:.4f})')
        ax.axvline(edge_max_threshold, color='g', linestyle='--', label=f'Edge Max Threshold ({edge_max_threshold:.4f})')
        ax.axvline(cloud_min_threshold, color='b', linestyle='--', label=f'Cloud Min Threshold ({cloud_min_threshold:.4f})')
        ax.axvline(cloud_max_threshold, color='purple', linestyle='--', label=f'Cloud Max Threshold ({cloud_max_threshold:.4f})')

        ax.set_title(f'Class {cls}')
        ax.set_xlabel('Probability')
        ax.set_ylabel('Samples')
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots if the number of classes is not an exact multiple of cols
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()

def visualize_probability_distribution4(model, data_loader, device='cuda', k=2):
    """
    Visualize the probability distributions for edge and cloud predictions,
    with thresholds based on standard deviation.
    
    Args:
        model: The neural network model with edge and cloud components
        data_loader: DataLoader containing the test/validation data
        device: Device to run the computations on ('cuda' or 'cpu')
        k: Number of standard deviations to calculate the threshold range
    """
    
    model.eval()
    model = model.to(device)
    
    edge_probs_by_class = {}
    cloud_probs_by_class = {}
    
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            
            # Forward pass
            edge_outputs, cloud_outputs = model(data)
            
            # Apply softmax to get probabilities
            edge_probs = F.softmax(edge_outputs, dim=1)
            cloud_probs = F.softmax(cloud_outputs, dim=1)
            
            # Get predicted classes
            edge_preds = torch.argmax(edge_probs, dim=1)
            cloud_preds = torch.argmax(cloud_probs, dim=1)
            
            # Collect probabilities by predicted class
            for i in range(len(data)):
                edge_pred_class = edge_preds[i].item()
                edge_pred_prob = edge_probs[i, edge_pred_class].item()
                
                cloud_pred_class = cloud_preds[i].item()
                cloud_pred_prob = cloud_probs[i, cloud_pred_class].item()
                
                if edge_pred_class not in edge_probs_by_class:
                    edge_probs_by_class[edge_pred_class] = []
                edge_probs_by_class[edge_pred_class].append(edge_pred_prob)
                
                if cloud_pred_class not in cloud_probs_by_class:
                    cloud_probs_by_class[cloud_pred_class] = []
                cloud_probs_by_class[cloud_pred_class].append(cloud_pred_prob)
    

    num_classes = len(edge_probs_by_class)
    cols = 2  # Two columns
    rows = (num_classes + 1) // cols  # Determine number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))  # Adjust figure size
    axes = axes.flatten()  # Flatten the 2D array to 1D for easy iteration

    for i, cls in enumerate(sorted(edge_probs_by_class.keys())):
        ax = axes[i]

        edge_probs = edge_probs_by_class[cls]
        cloud_probs = cloud_probs_by_class[cls]  # Get cloud probs, default empty list

        # Compute mean and standard deviation for edge and cloud
        edge_mean = np.mean(edge_probs)
        edge_std = np.std(edge_probs)
        cloud_mean = np.mean(cloud_probs)
        cloud_std = np.std(cloud_probs)

        # Calculate min and max thresholds based on standard deviations
        edge_min_threshold = edge_mean - k * edge_std
        edge_max_threshold = edge_mean + k * edge_std
        cloud_min_threshold = cloud_mean - k * cloud_std
        # cloud_max_threshold = cloud_mean + k * cloud_std
        
        # Plot histograms with labels including min/max threshold values
        ax.hist(cloud_probs, alpha=0.5, bins=80, label=f'Cloud (Min: {min(cloud_probs):.4f}, Max: {max(cloud_probs):.4f})')
        ax.hist(edge_probs, alpha=0.5, bins=80, label=f'Edge (Min: {min(edge_probs):.4f}, Max: {max(edge_probs):.4f})')

        # Draw lines for the thresholds on the plot
        ax.axvline(edge_min_threshold, color='r', linestyle='--', label=f'Edge Min Threshold ({edge_min_threshold:.4f})')
        ax.axvline(edge_max_threshold, color='g', linestyle='--', label=f'Edge Max Threshold ({edge_max_threshold:.4f})')
        ax.axvline(cloud_min_threshold, color='b', linestyle='--', label=f'Cloud Min Threshold ({cloud_min_threshold:.4f})')
        # ax.axvline(cloud_max_threshold, color='purple', linestyle='--', label=f'Cloud Max Threshold ({cloud_max_threshold:.4f})')

        ax.set_title(f'Class {cls}')
        ax.set_xlabel('Probability')
        ax.set_ylabel('Samples')
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots if the number of classes is not an exact multiple of cols
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()

def visualize_probability_distribution_mean_sqr(model, data_loader, device='cuda', k=2):
     
    
    model.eval()
    model = model.to(device)
    
    edge_probs_by_class = {}
    cloud_probs_by_class = {}
    
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            
            # Forward pass
            edge_outputs, cloud_outputs = model(data)
            
            # Apply softmax to get probabilities
            edge_probs = F.softmax(edge_outputs, dim=1)
            cloud_probs = F.softmax(cloud_outputs, dim=1)
            
            # Get predicted classes
            edge_preds = torch.argmax(edge_probs, dim=1)
            cloud_preds = torch.argmax(cloud_probs, dim=1)
            
            # Collect probabilities by predicted class
            for i in range(len(data)):
                edge_pred_class = edge_preds[i].item()
                edge_pred_prob = edge_probs[i, edge_pred_class].item()
                
                cloud_pred_class = cloud_preds[i].item()
                cloud_pred_prob = cloud_probs[i, cloud_pred_class].item()
                
                if edge_pred_class not in edge_probs_by_class:
                    edge_probs_by_class[edge_pred_class] = []
                edge_probs_by_class[edge_pred_class].append(edge_pred_prob)
                
                if cloud_pred_class not in cloud_probs_by_class:
                    cloud_probs_by_class[cloud_pred_class] = []
                cloud_probs_by_class[cloud_pred_class].append(cloud_pred_prob)
    

    num_classes = len(edge_probs_by_class)
    cols = 2  # Two columns
    rows = (num_classes + 1) // cols  # Determine number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))  # Adjust figure size
    axes = axes.flatten()  # Flatten the 2D array to 1D for easy iteration

    for i, cls in enumerate(sorted(edge_probs_by_class.keys())):
        ax = axes[i]

        edge_probs = edge_probs_by_class[cls]
        cloud_probs = cloud_probs_by_class[cls]  # Get cloud probs, default empty list

        
        edge_mean = np.mean(edge_probs)
        cloud_mean = np.mean(cloud_probs)
        # Compute the Mean Squared Error (MSE) between edge and cloud probabilities
        mse_edge = np.mean((np.array(edge_mean) - np.array(edge_probs))**2)
        mse_cloud = np.mean((np.array(cloud_mean) - np.array(cloud_probs))**2)
        

        # Use MSE to define thresholds for edge and cloud
        edge_min_threshold = np.mean(edge_probs) - k * mse_edge
        edge_max_threshold = np.mean(edge_probs) + k * mse_edge
        cloud_min_threshold = np.mean(cloud_probs) - k * mse_cloud
        
        # Plot histograms with labels including min/max threshold values
        ax.hist(cloud_probs, alpha=0.5, bins=80, label=f'Cloud (Min: {min(cloud_probs):.4f}, Max: {max(cloud_probs):.4f})')
        ax.hist(edge_probs, alpha=0.5, bins=80, label=f'Edge (Min: {min(edge_probs):.4f}, Max: {max(edge_probs):.4f})')

        # Draw lines for the thresholds on the plot
        ax.axvline(edge_min_threshold, color='r', linestyle='--', label=f'Edge Min Threshold ({edge_min_threshold:.4f})')
        ax.axvline(edge_max_threshold, color='g', linestyle='--', label=f'Edge Max Threshold ({edge_max_threshold:.4f})')
        ax.axvline(cloud_min_threshold, color='b', linestyle='--', label=f'Cloud Min Threshold ({cloud_min_threshold:.4f})')

        ax.set_title(f'Class {cls}')
        ax.set_xlabel('Probability')
        ax.set_ylabel('Samples')
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots if the number of classes is not an exact multiple of cols
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()


def cascade_confusion_matrix(model, data_iterator, num_classes, edge_threshold=0.5):
    """
    Creates and plots confusion matrix heatmaps for a multi-output model.
    Handles one-hot encoded labels and cascading logic where Edge predictions
    below a threshold are treated as Cloud predictions.
    
    Args:
        model: PyTorch model with multiple outputs
        data_iterator: Iterator yielding (inputs, targets) pairs
        num_classes: Number of classes in the dataset
        edge_threshold: The threshold value to determine if an Edge prediction is valid
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize lists to store predictions and true labels for both classifiers
    edge_preds = []
    cloud_preds = []
    true_labels = []
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        for inputs, labels in data_iterator:
            # Move inputs to the same device as the model
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            
            # Get predictions from both classifiers
            edge_output, cloud_output = model(inputs)
            
            # Convert model outputs to probabilities (using softmax)
            edge_probs = torch.nn.functional.softmax(edge_output, dim=1)
            cloud_probs = torch.nn.functional.softmax(cloud_output, dim=1)
            
            # Get predicted classes (taking the max of probabilities)
            _, edge_pred = torch.max(edge_probs, 1)
            _, cloud_pred = torch.max(cloud_probs, 1)
            
            # Store true labels
            if labels.dim() > 1 and labels.size(1) > 1:
                labels = torch.argmax(labels, dim=1)
            
            # Iterate over the batch to apply thresholding logic
            for i in range(len(labels)):
                if edge_probs[i, edge_pred[i]] < edge_threshold:  # If Edge prediction is below threshold
                    cloud_preds.append(cloud_pred[i].cpu().numpy())
                    edge_preds.append(-1)  # Mark invalid prediction for Edge classifier
                else:
                    edge_preds.append(edge_pred[i].cpu().numpy())
                    cloud_preds.append(-1)  # No need to record Cloud prediction
                
                true_labels.append(labels[i].cpu().numpy())
    
    # Convert lists to numpy arrays
    edge_preds = np.array(edge_preds)
    cloud_preds = np.array(cloud_preds)
    true_labels = np.array(true_labels)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # Calculate and plot Edge classifier confusion matrix (excluding invalid predictions)
    edge_cm = confusion_matrix(
        true_labels[edge_preds != -1],  # Only consider valid Edge predictions
        edge_preds[edge_preds != -1],   # Only valid Edge predictions
        labels=range(num_classes)
    )
    
    sns.heatmap(
        edge_cm, 
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=range(num_classes),
        yticklabels=range(num_classes),
        ax=ax1
    )
    ax1.set_title('Edge Classifier Confusion Matrix')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # Calculate and plot Cloud classifier confusion matrix (excluding invalid predictions)
    cloud_cm = confusion_matrix(
        true_labels[cloud_preds != -1],  # Only consider valid Cloud predictions
        cloud_preds[cloud_preds != -1],  # Only valid Cloud predictions
        labels=range(num_classes)
    )
    
    sns.heatmap(
        cloud_cm, 
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=range(num_classes),
        yticklabels=range(num_classes),
        ax=ax2
    )
    ax2.set_title('Cloud Classifier Confusion Matrix')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    plt.tight_layout()
    return fig, (edge_cm, cloud_cm)

def cascade_confusion_matrix_percent(model, data_iterator, num_classes, edge_threshold=0.5):
    # Set model to evaluation mode
    model.eval()

    # Initialize lists to store predictions and true labels for both classifiers
    edge_preds = []
    cloud_preds = []
    true_labels = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for inputs, labels in data_iterator:
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            edge_output, cloud_output = model(inputs)
            edge_probs = torch.nn.functional.softmax(edge_output, dim=1)
            cloud_probs = torch.nn.functional.softmax(cloud_output, dim=1)
            _, edge_pred = torch.max(edge_probs, 1)
            _, cloud_pred = torch.max(cloud_probs, 1)

            if labels.dim() > 1 and labels.size(1) > 1:
                labels = torch.argmax(labels, dim=1)

            for i in range(len(labels)):
                if edge_probs[i, edge_pred[i]] < edge_threshold:
                    cloud_preds.append(cloud_pred[i].cpu().numpy())
                    edge_preds.append(-1)
                else:
                    edge_preds.append(edge_pred[i].cpu().numpy())
                    cloud_preds.append(-1)

                true_labels.append(labels[i].cpu().numpy())

    edge_preds = np.array(edge_preds)
    cloud_preds = np.array(cloud_preds)
    true_labels = np.array(true_labels)

    # Count how many predictions were processed by Edge and Cloud
    total_samples = len(true_labels)
    edge_count = np.sum(edge_preds != -1)
    cloud_count = np.sum(cloud_preds != -1)
    edge_percent = 100.0 * edge_count / total_samples
    cloud_percent = 100.0 * cloud_count / total_samples

    print(f"Edge processed {edge_count}/{total_samples} samples ({edge_percent:.2f}%)")
    print(f"Cloud processed {cloud_count}/{total_samples} samples ({cloud_percent:.2f}%)")

    # Plotting confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    edge_cm = confusion_matrix(
        true_labels[edge_preds != -1],
        edge_preds[edge_preds != -1],
        labels=range(num_classes)
    )

    sns.heatmap(
        edge_cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=range(num_classes),
        yticklabels=range(num_classes),
        ax=ax1
    )
    ax1.set_title(f'Edge Confusion Matrix\n({edge_percent:.2f}% of data)')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')

    cloud_cm = confusion_matrix(
        true_labels[cloud_preds != -1],
        cloud_preds[cloud_preds != -1],
        labels=range(num_classes)
    )

    sns.heatmap(
        cloud_cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=range(num_classes),
        yticklabels=range(num_classes),
        ax=ax2
    )
    ax2.set_title(f'Cloud Confusion Matrix\n({cloud_percent:.2f}% of data)')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')

    plt.tight_layout()
    return fig, (edge_cm, cloud_cm), (edge_percent, cloud_percent)