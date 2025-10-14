import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from data import test_loader

torch_model = torch.load('./cnn_model.pth', map_location=torch.device('cpu'), weights_only=False)
torch_model.eval()
dummy_input = torch.randn(1, 1, 28, 28)

# export onnx model
if not os.path.exists('cnn.onnx'):
    # `dynamic_axes` allow variable batch size for input and output, allow we run inference with different batch size
    torch.onnx.export(torch_model, (dummy_input,), 'cnn.onnx', verbose=True, input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

# show graph
onnx_model = onnx.load('cnn.onnx')
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))

# run inference with onnxruntime
data_iter = iter(test_loader)
images, labels = next(data_iter)
images = images.numpy()
# shape of images
print(f"images.shape:{images.shape}")
labels = labels.numpy()

ort_session = ort.InferenceSession('cnn.onnx', providers=['CPUExecutionProvider'])
# input_shape
print(f"Input shape: {ort_session.get_inputs()[0].shape}")

# run inference
ort_input = {'input': images}
ort_outs = ort_session.run(['output'], ort_input)
ort_outs_np = np.array(ort_outs)
print(f"shape of output: {ort_outs_np.shape}")
# get the predicted class
predicted_class = np.argmax(ort_outs_np, axis=2)

# evaluate the model with precision, recall, F1-score
correct = (predicted_class.flatten() == labels).sum()
total = labels.shape[0]
accuracy = 100 * correct / total
print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

precision = precision_score(labels, predicted_class.flatten(), average='weighted')
recall = recall_score(labels, predicted_class.flatten(), average='weighted')
f1 = f1_score(labels, predicted_class.flatten(), average='weighted')
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

# compare with pytorch cnn results in cnn_evaluation.txt then save to onnx_evaluation.txt belong with pytorch results to compare
with open('cnn_evaluation.txt', 'r') as f:
    lines = f.readlines()
with open('onnx_evaluation.txt', 'w') as f:
    for line in lines:
        # for each line write result of pytorch and onnx
        if 'Accuracy' in line:
            f.write(f'pytorch {line.strip()}' + f', ONNX Accuracy: {accuracy:.2f}%\n')
        elif 'Precision' in line:
            f.write(f'pytorch {line.strip()}' + f', ONNX Precision: {precision:.4f}\n')
        elif 'Recall' in line:
            f.write(f'pytorch {line.strip()}' + f', ONNX Recall: {recall:.4f}\n')
        elif 'F1-score' in line:
            f.write(f'pytorch {line.strip()}' + f', ONNX F1-score: {f1:.4f}\n')
