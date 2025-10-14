import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

from cnn import CNN
from data import test_loader, train_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
model = CNN().to(device)

if not os.path.exists('cnn.pth'):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    number_of_epochs = 5

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

        print(f'Epoch [{epoch+1}/{number_of_epochs}], Loss: {avg_loss:.4f}')

    # save the model
    torch.save(model.state_dict(), 'cnn.pth')

model.load_state_dict(torch.load('cnn.pth'))
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
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')

    # write evaluation metrics to a file
    with open('cnn_evaluation.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy:.2f}%\n')
        f.write(f'Precision: {precision:.4f}\n')
        f.write(f'Recall: {recall:.4f}\n')
        f.write(f'F1-score: {f1:.4f}\n')

    # display conv filter 3*3s of the first layer
    filters = model.conv1.weight.data.cpu().numpy()
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(filters[i, 0, :, :], cmap='gray')
        ax.set_title(f'Filter {i+1}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

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
