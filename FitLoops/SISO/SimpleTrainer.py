import torch
from torch import nn
from matplotlib_inline import backend_inline
from IPython import display
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

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

def test(net, data_iter, device=None):
    """计算测试精度，使用gpu"""
    if isinstance(net, nn.Module):
        net.eval() #设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), len(y))
    return metric[0] / metric[1]

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
    
def create_confusion_matrix(model, data_iterator, num_classes):
    """
    Creates and plots a confusion matrix heatmap for the given model and dataset.
    Handles one-hot encoded labels.
    
    Args:
        model: PyTorch model in evaluation mode
        data_iterator: Iterator yielding (inputs, targets) pairs
        num_classes: Number of classes in the dataset
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize lists to store predictions and true labels
    all_preds = []
    all_labels = []
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        for inputs, labels in data_iterator:
            # Move inputs to the same device as the model
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            
            # Get predictions
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Convert one-hot encoded labels to class indices
            if labels.dim() > 1 and labels.size(1) > 1:
                labels = torch.argmax(labels, dim=1)
            
            # Convert to numpy arrays
            batch_preds = preds.cpu().numpy()
            batch_labels = labels.cpu().numpy()
            
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
    
    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate confusion matrix
    cm = confusion_matrix(
        all_labels, 
        all_preds, 
        labels=range(num_classes)
    )
    
    # Create heatmap with class labels
    plt.figure(figsize=(8, 4))
    sns.heatmap(
        cm, 
        annot=True,      # Show numbers in cells
        fmt='d',         # Format as integers
        cmap='Blues',    # Color scheme
        xticklabels=range(num_classes),
        yticklabels=range(num_classes)
    )
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    return plt.gcf()


