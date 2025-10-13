# Step-by-Step Guide: Classifying MNIST with PyTorch

This guide will teach you how to classify handwritten digits from the MNIST dataset using PyTorch, starting from raw IDX files.

---

## 1. Install PyTorch

Open your terminal and run:

```bash
pip install torch torchvision
```

---

## 2. Load and Prepare MNIST Data

### Understanding the MNIST Data Format

MNIST data is stored in IDX format, which is a simple binary format for storing vectors and multidimensional matrices. Each image is 28x28 pixels (grayscale, values 0-255), and each label is a single digit (0-9).

### Loading the Data

You use NumPy to read the binary files and extract the images and labels as arrays. See your `main.py` for the custom functions that do this.

### Preprocessing Steps

1. **Convert to PyTorch Tensors:** PyTorch models require data as tensors, not NumPy arrays.

    **What is a Tensor?**
    A tensor is a multi-dimensional array, similar to a NumPy array, but with additional capabilities for automatic differentiation and GPU acceleration. Tensors are the fundamental data structure in PyTorch and are used to represent inputs, outputs, and parameters of neural networks.

    - **Scalars** are 0-dimensional tensors (e.g., a single number).
    - **Vectors** are 1-dimensional tensors (e.g., a list of numbers).
    - **Matrices** are 2-dimensional tensors (e.g., a table of numbers).
    - **Images** are often 3D or 4D tensors (e.g., batch of grayscale images: `(batch_size, channels, height, width)`).

    PyTorch tensors support fast mathematical operations and can be moved between CPU and GPU for efficient computation. This makes them ideal for deep learning tasks.
2. **Reshape Images:** PyTorch expects images in the format `(batch_size, channels, height, width)`. MNIST images are single-channel (grayscale), so we add a channel dimension.
3. **Normalize Pixel Values:** Neural networks train better when input values are scaled. Dividing by 255.0 converts pixel values from `[0, 255]` to `[0, 1]`.
4. **Create Dataset and DataLoader:**
   - `Dataset` wraps your data and provides `__getitem__` and `__len__` methods for easy access.
   - `DataLoader` batches your data and shuffles it for training.

### Why These Steps?

- **Tensor conversion** allows GPU acceleration and compatibility with PyTorch layers.
- **Reshaping** ensures the model receives data in the expected format.
- **Normalization** helps gradients flow and speeds up training.
- **Batching and shuffling** improve training stability and generalization.

### Example Code

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        # Convert to float tensor, add channel dimension, and normalize
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Load your data (see your main.py for details)
train_images, train_labels, test_images, test_labels = load_mnist('data')
train_dataset = MNISTDataset(train_images, train_labels)
test_dataset = MNISTDataset(test_images, test_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)
```

---

## 3. Define a Neural Network

### What is a Neural Network?

A neural network is a series of layers that transform input data (images) into predictions (digit classes). Each layer applies mathematical operations to learn patterns from the data.

### PyTorch Neural Network Basics

- **Module**: The base class for all neural network models in PyTorch is `nn.Module`.
- **Layers**: Layers like `nn.Linear` (fully connected) are building blocks of the network.
- **Activation Function**: Functions like `ReLU` introduce non-linearity, helping the network learn complex patterns.

### Example: SimpleNet for MNIST

This network has:

- An input layer that flattens the 28x28 image into a 784-element vector.
- A hidden layer (`fc1`) with 128 neurons.
- A ReLU activation for non-linearity.
- An output layer (`fc2`) with 10 neurons (one for each digit class).

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # First fully connected layer: input 784 (28x28), output 128
        self.fc1 = nn.Linear(28*28, 128)
        # Second fully connected layer: input 128, output 10 (digit classes)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Flatten the image tensor to (batch_size, 784)
        x = x.view(-1, 28*28)
        # Apply first layer and ReLU activation
        x = F.relu(self.fc1(x))
        # Apply second layer (outputs raw scores for each class)
        x = self.fc2(x)
        return x
```

#### Layer-by-Layer Explanation

- `self.fc1 = nn.Linear(28*28, 128)`: This layer takes the flattened image and outputs 128 features. Each feature is a weighted sum of all input pixels plus a bias.
- `F.relu(self.fc1(x))`: The ReLU (Rectified Linear Unit) activation replaces negative values with zero. This non-linearity helps the network learn complex patterns and relationships in the data. Without an activation function, the network would only be able to learn linear mappings, which are not sufficient for most real-world tasks. ReLU is popular because it is simple, fast to compute, and helps mitigate the vanishing gradient problem, allowing deeper networks to train more effectively.

**What is Non-Linearity and Why is it Important?**
**Why is ReLU Often Preferred Over Other Non-Linear Activations?**

### Mathematical Details of Activation Functions

**ReLU (Rectified Linear Unit):**

$$
 ext{ReLU}(x) = \max(0, x)
$$

This means for any input $x$, if $x$ is positive, ReLU returns $x$; if $x$ is negative, it returns $0$.

**Sigmoid:**

$$
 ext{Sigmoid}(x) = \frac{1}{1 + e^{-x}}
$$

Sigmoid squashes input values to the range $(0, 1)$. It can cause gradients to vanish for very large or small $x$.

**Tanh (Hyperbolic Tangent):**

$$
 anh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
$$

Tanh squashes input values to the range $(-1, 1)$, but also suffers from vanishing gradients.

**Why ReLU is Better Mathematically:**

- The derivative of ReLU is $1$ for $x > 0$ and $0$ for $x \leq 0$, so gradients flow well for positive activations.
- Sigmoid and tanh derivatives become very small for large $|x|$, making learning slow in deep networks (the vanishing gradient problem).

**Summary Table:**

| Activation | Formula                             | Output Range  | Derivative                               | Vanishing Gradient? |
| ---------- | ----------------------------------- | ------------- | ---------------------------------------- | ------------------- |
| ReLU       | $\max(0, x)$                        | $[0, \infty)$ | $1$ if $x>0$, $0$ otherwise              | No (for $x>0$)      |
| Sigmoid    | $\frac{1}{1+e^{-x}}$                | $(0, 1)$      | $\text{Sigmoid}(x)(1-\text{Sigmoid}(x))$ | Yes                 |
| Tanh       | $\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$ | $(-1, 1)$     | $1-\tanh^2(x)$                           | Yes                 |

ReLU (Rectified Linear Unit) is widely used because:

- **Simplicity**: ReLU is easy to compute—just set negative values to zero.
- **Efficiency**: It speeds up training by allowing gradients to flow well, reducing the risk of vanishing gradients (where gradients become too small for effective learning).
- **Sparse Activation**: By zeroing out negatives, ReLU makes the network more efficient and can help with generalization.
- **Empirical Success**: In practice, ReLU often leads to faster convergence and better performance compared to older activations like sigmoid or tanh.

Other non-linear functions (like sigmoid and tanh) can cause gradients to vanish, slowing or even stopping learning in deep networks. ReLU avoids this for most inputs, making it a strong default choice for modern neural networks.

Non-linearity means the output of a function is not directly proportional to its input. In neural networks, non-linear activation functions (like ReLU) allow the model to learn and represent complex relationships in data, such as curves, edges, and patterns that cannot be captured by straight lines alone.

If a network only used linear functions, no matter how many layers it had, the entire network would behave like a single linear transformation. This would severely limit its ability to solve complex problems. By introducing non-linearity, neural networks can approximate any function and solve tasks like image recognition, speech processing, and more.

- `self.fc2 = nn.Linear(128, 10)`: This layer maps the 128 features to 10 output classes (digits 0-9).
- The output of `forward` is a tensor of shape `(batch_size, 10)`, containing scores for each digit class. During training, these scores are passed to a loss function (like `CrossEntropyLoss`) which computes how well the model's predictions match the true labels.

#### Why This Structure?

- **Fully connected layers** are simple and effective for MNIST, which is a small image dataset.
- **ReLU activation** helps the network learn faster and more accurately than linear-only models.
- **Output layer** matches the number of classes (10 digits).

You can experiment by adding more layers, changing the number of neurons, or using other activation functions for more complex models.

---

## 4. Train the Model

---

## Advanced Topics

### Regularization

Regularization helps prevent overfitting, where the model memorizes training data but fails to generalize to new data.

- **Weight Decay (L2 Regularization):** Adds a penalty to large weights in the loss function.
  - In PyTorch, set `weight_decay` in the optimizer:

        ```python
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        ```

- **Dropout:** Randomly sets some activations to zero during training, forcing the network to learn redundant representations.
  - Add `nn.Dropout(p)` layers to your model, e.g.:

        ```python
        self.dropout = nn.Dropout(0.5)
        x = self.dropout(x)
        ```

### Learning Rate Schedules

The learning rate controls how much weights are updated during training. Scheduling the learning rate can help the model converge faster and avoid local minima.

- **StepLR:** Reduces the learning rate by a factor every few epochs.

    ```python
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(epochs):
            ...
            scheduler.step()
    ```

- **ReduceLROnPlateau:** Reduces the learning rate when a metric (like validation loss) stops improving.

### Visualization

Visualizing training progress and model predictions helps diagnose issues and understand model behavior.

- **Loss and Accuracy Curves:** Plot loss and accuracy over epochs to monitor training and detect overfitting.

    ```python
    import matplotlib.pyplot as plt
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.show()
    ```

- **Confusion Matrix:** Shows how well the model distinguishes between classes.

    ```python
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    cm = confusion_matrix(true_labels, predicted_labels)
    sns.heatmap(cm, annot=True)
    plt.show()
    ```

- **Visualize Predictions:** Display sample images with predicted and true labels to see where the model succeeds or fails.

    ```python
    plt.imshow(images[0].squeeze(), cmap='gray')
    plt.title(f'Predicted: {predicted[0]}, True: {labels[0]}')
    plt.show()
    ```

---

#### Mathematical Details of Training

Training a neural network is an optimization problem. The goal is to find the weights $W$ and biases $b$ that minimize the loss function $L$ over the training data.

**Forward Pass:**
For each input $x$, the network computes:

$$
h_1 = \text{ReLU}(W_1 x + b_1) \\
 ext{output} = W_2 h_1 + b_2
$$

**Loss Calculation:**
For classification, we use cross-entropy loss:

$$
L = -\sum_{i=1}^{N} y_i \log(p_i)
$$
where $y_i$ is the true label (one-hot encoded) and $p_i$ is the predicted probability for class $i$.

**Backward Pass (Gradient Descent):**
PyTorch automatically computes gradients of $L$ with respect to all weights using backpropagation. The optimizer (Adam) updates weights:

$$
W \leftarrow W - \eta \frac{\partial L}{\partial W}
$$
where $\eta$ is the learning rate.

**Batch Training:**
Instead of updating weights after every sample, we use batches (mini-batch gradient descent) for efficiency and stability.

**Adam Optimizer:**
Adam combines momentum and adaptive learning rates for each parameter, making training faster and more robust to noisy gradients.

**Epochs:**
One epoch means the model has seen all training samples once. Multiple epochs allow the model to refine its weights.

#### Practical Tips

- Monitor loss during training; if it doesn't decrease, try adjusting the learning rate or model architecture.
- Use validation data to check for overfitting (model memorizing training data but failing on new data).

### What Happens During Training?

Training a neural network means adjusting its weights so it can make accurate predictions. This is done by showing it batches of data, calculating how wrong its predictions are (loss), and updating the weights to reduce this error.

#### Key Components

- **Device**: Training can be done on CPU or GPU. GPU is much faster for large models and datasets.
- **Model**: The neural network you defined.
- **Optimizer**: Algorithm that updates the model's weights. Adam is popular for its speed and reliability.
- **Loss Function**: Measures how far the model's predictions are from the true labels. CrossEntropyLoss is standard for classification.

#### Training Loop Explained

1. Set the model to training mode (`model.train()`).
2. For each batch of images and labels:
    - Move data to the chosen device (CPU/GPU).
    - Zero out previous gradients (`optimizer.zero_grad()`).
    - Run the model to get predictions (`outputs = model(images)`).
    - Calculate the loss (`loss = criterion(outputs, labels)`).
    - Compute gradients (`loss.backward()`).
    - Update weights (`optimizer.step()`).
3. Print progress after each epoch.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done.")
```

#### Why These Steps?

- **Zeroing gradients** prevents accumulation from previous batches.
- **Backward pass** computes how each weight contributed to the error.
- **Optimizer step** nudges weights to reduce future errors.

Repeat for several epochs to allow the model to learn from the data.

---

## 5. Evaluate the Model

#### Mathematical Details of Evaluation

**Forward Pass:**
For each test input $x$, compute predicted scores as in training.

**Prediction:**
Select the class with the highest score:

$$
\hat{y} = \arg\max_j \text{output}_j
$$

**Accuracy Calculation:**
Accuracy is the fraction of correct predictions:

$$
 ext{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of samples}}
$$

**No Gradients:**
During evaluation, gradients are not needed, so we disable them for speed and memory savings.

#### Practical Tips

- If accuracy is low, check for data preprocessing issues, insufficient training, or model underfitting.
- For deeper analysis, use confusion matrices, precision, recall, and F1-score to understand model strengths and weaknesses.

### What Happens During Evaluation?

Evaluation checks how well the trained model performs on unseen data (the test set). This tells you if the model has learned to generalize, not just memorize the training data.

#### Key Steps

- **Set model to evaluation mode** (`model.eval()`): disables dropout and batch normalization, if present.
- **No gradient calculation** (`torch.no_grad()`): saves memory and speeds up computation.
- **Prediction**: For each batch, get the model's output and choose the class with the highest score.
- **Accuracy Calculation**: Compare predictions to true labels and count correct ones.

```python
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
print(f"Test Accuracy: {correct / total:.4f}")
```

#### Why These Steps?

- **Evaluation mode** ensures consistent behavior for layers like dropout.
- **No gradients** makes evaluation faster and safer.
- **Accuracy** is a simple, intuitive metric for classification tasks.

You can also compute other metrics (precision, recall, confusion matrix) for deeper analysis.

---

## Summary

1. Install PyTorch
2. Load and preprocess MNIST data
3. Define a neural network
4. Train the model
5. Evaluate accuracy

You can now classify MNIST digits using PyTorch!
