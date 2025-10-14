# Building a Convolutional Neural Network (CNN) for MNIST Classification

---

## Mathematical Foundations of CNNs

### Kernel (Filter)

#### Visual Example: 3x3 Kernel on a 5x5 Image

Input image:

```
1 2 3 4 5
6 7 8 9 10
11 12 13 14 15
16 17 18 19 20
21 22 23 24 25
```

3x3 kernel (filter):

```
a b c
d e f
g h i
```

The kernel slides over the image, multiplying and summing values in each 3x3 patch to produce a feature map.

A kernel (or filter) in a convolutional layer is a small matrix of weights that slides over the input image to extract features. For example, a $3 \times 3$ kernel looks at a $3 \times 3$ patch of the image at a time. Each kernel learns to detect a specific pattern, such as edges or textures.

Mathematically, for an input image $I$ and kernel $K$:
$$
[I * K](i, j) = \sum_{u=0}^{m-1} \sum_{v=0}^{n-1} I[i+u, j+v] \cdot K[u, v]
$$
where $m$ and $n$ are the height and width of the kernel.

Multiple kernels are used in each convolutional layer, producing multiple feature maps.

### Padding

#### Visual Example: Zero Padding

Original image (3x3):

```
1 2 3
4 5 6
7 8 9
```

With padding of 1 (zeros added):

```
0 0 0 0 0
0 1 2 3 0
0 4 5 6 0
0 7 8 9 0
0 0 0 0 0
```

This allows the kernel to process edge pixels and control the output size.

Padding refers to adding extra pixels (usually zeros) around the border of the input image before applying the convolution. Padding controls the spatial size of the output feature map.

- **No padding (valid):** The kernel only slides within the original image, so the output is smaller than the input.
- **Zero padding (same):** Zeros are added so the output size matches the input size.

For a $3 \times 3$ kernel and input of size $28 \times 28$:

- With no padding, output size is $26 \times 26$.
- With padding of 1, output size remains $28 \times 28$.

Padding helps preserve edge information and control the output size, which is important for stacking multiple convolutional layers.

### 1. Convolution Operation

The core of a CNN is the convolution operation. For a 2D image $I$ and a filter (kernel) $K$ of size $m \times n$:

$$
[I * K](i, j) = \sum_{u=0}^{m-1} \sum_{v=0}^{n-1} I[i+u, j+v] \cdot K[u, v]
$$

This slides the kernel over the image, multiplying and summing values to produce a feature map. Each filter learns to detect specific patterns (edges, shapes, etc.).

### 2. Activation Function (ReLU)

After convolution, an activation function introduces non-linearity:

$$
 \text{ReLU}(x) = \max(0, x)
$$

This helps the network learn complex features.

### 3. Pooling (Max Pooling)

Pooling reduces the spatial size of feature maps, making computation efficient and providing translation invariance. For max pooling with window size $2 \times 2$:

$$
 \text{MaxPool}(x) = \max\{x_{1}, x_{2}, x_{3}, x_{4}\}
$$

### 4. Flattening and Fully Connected Layers

After several convolution and pooling layers, the output is flattened into a vector and passed to fully connected layers:

$$
z = W x + b
$$

where $W$ is the weight matrix, $x$ is the input vector, and $b$ is the bias.

### 5. Output Layer and Softmax

For classification, the final layer uses softmax to convert raw scores to probabilities:

$$
 ext{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}
$$

where $C$ is the number of classes (10 for MNIST).

### 6. Loss Function (Cross-Entropy)

The loss function measures how well the predicted probabilities match the true labels:

$$
L = -\sum_{i=1}^{C} y_i \log(p_i)
$$

where $y_i$ is the true label (one-hot encoded) and $p_i$ is the predicted probability for class $i$.

### 7. Backpropagation and Optimization

Gradients of the loss with respect to all weights are computed using backpropagation. The optimizer (e.g., Adam) updates weights:

$$
W \leftarrow W - \eta \frac{\partial L}{\partial W}
$$

where $\eta$ is the learning rate.

---

This guide will teach you how to build, train, and evaluate a CNN for handwritten digit recognition using the MNIST dataset in PyTorch.

---

## 1. What is a CNN?

A Convolutional Neural Network (CNN) is a type of deep learning model especially effective for image data. CNNs use convolutional layers to automatically learn spatial hierarchies and features from images, making them ideal for tasks like digit recognition.

**Key components:**

- **Convolutional layers:** Extract local features using learnable filters.
- **Activation functions (e.g., ReLU):** Introduce non-linearity.
- **Pooling layers:** Downsample feature maps to reduce computation and control overfitting.
- **Fully connected layers:** Combine features for final classification.

---

## 2. Define a Simple CNN in PyTorch

### Layer-by-Layer Reasoning

- **Conv2d:** Learns spatial features by convolving filters over the image. Each filter detects a different pattern.
- **ReLU:** Applies non-linearity, allowing the network to learn complex mappings.
- **MaxPool2d:** Reduces feature map size, keeps strongest activations, and provides translation invariance.
- **Flatten:** Converts 3D feature maps to 1D vector for classification.
- **Linear (Fully Connected):** Combines features for decision making.

### Parameter Calculation Example

For `self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)`, the number of parameters is:

$$
    \text{Params} = (\text{in\_channels} \times \text{kernel\_height} \times \text{kernel\_width} + 1) \times \text{out\_channels}
$$

For MNIST: $(1 \times 3 \times 3 + 1) \times 16 = 160$ parameters.

---

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 input channel (grayscale), 16 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Convolution + ReLU + Pooling
        x = self.pool(F.relu(self.conv1(x)))  # Output: [batch, 16, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # Output: [batch, 32, 7, 7]
        x = x.view(-1, 32 * 7 * 7)           # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

**Explanation:**

- `conv1`: Learns 16 filters from the input image.
- `pool`: Reduces spatial size by half (max pooling).
- `conv2`: Learns 32 filters from the previous layer.
- `fc1`: Fully connected layer for feature combination.
- `fc2`: Output layer for 10 digit classes.

---

## 3. Training the CNN

### Training Details

During training, the model learns by minimizing the cross-entropy loss using gradient descent. Each epoch consists of:

- Forward pass: Compute outputs and loss.
- Backward pass: Compute gradients.
- Optimizer step: Update weights.

### Why Adam Optimizer?

Adam adapts learning rates for each parameter and uses momentum, making training faster and more stable than vanilla SGD.

### Monitoring Training

Track loss and accuracy over epochs to detect overfitting or underfitting.

---

The training loop is similar to a fully connected network:

```python
model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
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

---

## 4. Evaluating the CNN

### Evaluation Metrics

- **Accuracy:** Fraction of correct predictions.
- **Precision, Recall, F1-score:** Useful for imbalanced datasets.
- **Confusion Matrix:** Visualizes prediction errors for each class.

### Visualization

Display sample images with predicted and true labels to interpret model performance.

---

After training, evaluate the model on the test set:

```python
model.eval()
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
accuracy = correct / len(test_dataset)
print(f"Test Accuracy: {accuracy:.4f}")
```

---

## 5. Why CNNs Work Well for Images

### Related Concepts

- **Translation invariance:** CNNs recognize patterns regardless of position.
- **Parameter sharing:** Filters are reused, reducing the number of parameters.
- **Local connectivity:** Each neuron connects to a small region, capturing local features.
- **Hierarchical feature learning:** Stacking layers allows the network to learn low-level (edges) and high-level (shapes, digits) features.

---

- **Local connectivity:** Convolutions focus on small regions, capturing edges, shapes, and textures.
- **Parameter sharing:** Filters are reused across the image, reducing the number of parameters.
- **Translation invariance:** Pooling and convolutions help the network recognize patterns regardless of their position.

---

## 6. Tips for Better CNNs

- Add more convolutional layers for deeper feature extraction.
- Use dropout or batch normalization to improve generalization.
- Experiment with different kernel sizes, number of filters, and learning rates.
- Visualize feature maps to understand what the network is learning.

---

## 7. Summary

---

## 8. Advanced Architectures in Deep Learning

---

## 9. K-Fold Cross-Validation and StratifiedKFold

### What is K-Fold Cross-Validation?

K-Fold Cross-Validation is a technique for evaluating model performance and robustness. The dataset is split into $k$ equal parts (folds). The model is trained on $k-1$ folds and validated on the remaining fold. This process repeats $k$ times, each time with a different fold as the validation set. The final score is the average of all $k$ validation results.

**Benefits:**

- Provides a more reliable estimate of model performance.
- Reduces risk of overfitting to a single train/test split.

### What is StratifiedKFold?

StratifiedKFold is a variant of k-fold cross-validation that preserves the percentage of samples for each class in every fold. This is especially important for classification tasks with imbalanced classes, ensuring each fold is representative of the overall class distribution.

**Example Usage (with scikit-learn):**

```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)
for train_index, val_index in skf.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    # Train and validate your model here
```

**Summary:**

- K-Fold Cross-Validation helps assess model generalization.
- StratifiedKFold ensures balanced class representation in each fold, leading to fairer and more robust evaluation for classification problems like MNIST.

Advanced architectures go beyond simple CNNs to solve more complex tasks, improve accuracy, and address challenges like vanishing gradients and computational efficiency.

### Examples of Advanced Architectures

- **LeNet, AlexNet, VGG:** Early deep CNNs with many convolutional and pooling layers. VGG uses very deep stacks of 3x3 convolutions.
- **GoogLeNet (Inception):** Uses parallel filters of different sizes (inception modules) to capture multi-scale features.
- **ResNet (Residual Networks):** Introduces skip (residual) connections:
    $$
    y = F(x) + x
    $$
    This allows gradients to flow through very deep networks, solving the vanishing gradient problem.
- **DenseNet:** Connects each layer to every other layer, improving feature reuse and gradient flow.
- **MobileNet, EfficientNet:** Designed for speed and efficiency, often used in mobile or embedded devices.
- **UNet:** Encoder-decoder structure with skip connections, used for image segmentation.
- **Vision Transformers (ViT):** Use attention mechanisms to focus on important regions of the image.

### Key Techniques in Advanced Architectures

- **Batch Normalization:** Normalizes activations to speed up training and improve stability.
- **Dropout:** Randomly disables neurons during training to prevent overfitting.
- **Skip Connections:** Help gradients flow and allow deeper networks.
- **Attention Mechanisms:** Let the network focus on relevant parts of the input.

### Why Use Advanced Architectures?

- Handle larger and more complex datasets.
- Improve accuracy and generalization.
- Enable new tasks (object detection, segmentation, image generation).
- Reduce computational cost for deployment on devices.

### Try It

For MNIST, you can experiment with ResNet, VGG, or even Vision Transformers using PyTorch’s `torchvision.models` or community implementations.

A CNN is a powerful tool for image classification. By stacking convolutional, pooling, and fully connected layers, you can build models that learn to recognize digits in MNIST with high accuracy.

Try modifying the architecture and training parameters to see how they affect performance!
