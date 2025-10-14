import os
import struct

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

from data import test_loader, train_loader
from simple_net import SimpleNet

# # display a sample image

# plt.imshow(train_images[1], cmap='gray')
# plt.title(f'Label: {train_labels[1]}')
# plt.show()
# plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
number_of_hidden_neurons = 128
model = SimpleNet(number_of_hidden_neurons).to(device)

# if `simple_net.pth` not exist, train the model
if not os.path.exists('simple_net.pth'):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    number_of_epochs = 10

    for epoch in range(number_of_epochs):
        # set model to training mode
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            # move to device
            images, labels = images.to(device), labels.to(device)
            # forward pass
            optimizer.zero_grad()
            outputs = model(images)
            # compute loss
            loss = criterion(outputs, labels)
            # backward pass
            loss.backward()
            # update weights
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        print(f'Epoch [{epoch + 1}/{number_of_epochs}], Loss: {avg_loss:.4f}')

    # save the model
    torch.save(model.state_dict(), 'simple_net.pth')

    #  print weights and biases of the first layer
    print("Weights of the first layer:")
    print(model.fc1.weight)
    print("Biases of the first layer:")
    print(model.fc1.bias)

# load model from file and display some predictions
model.load_state_dict(torch.load('simple_net.pth'))
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # move to cpu for visualization
    images = images.cpu()
    labels = labels.cpu()
    predicted = predicted.cpu()

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        # randomly select 9 images
        idx = np.random.randint(0, images.size(0), 9)
        ax.imshow(images[idx[i]].squeeze(), cmap='gray')
        ax.set_title(f'True: {labels[idx[i]].item()}, Pred: {predicted[idx[i]].item()}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
