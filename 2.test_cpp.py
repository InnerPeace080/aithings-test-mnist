import subprocess

from sklearn.metrics import f1_score, precision_score, recall_score

from data import test_loader


def get_execution_command():
    import platform
    system = platform.system()
    if system == 'Linux':
        return './cpp/onnx_infer_linux'
    elif system == 'Darwin':
        return './cpp/onnx_infer_mac'
    else:
        raise RuntimeError(f'Unsupported platform: {system}')

# for each image run inference by `cpp/onnx_infer_linux` or `cpp/onnx_infer_mac`


correct = 0
total = 0
all_labels = []
all_preds = []
for images, labels in test_loader:
    for i in range(images.size(0)):
        # logging every 100 images
        if total % 100 == 0:
            print(f'Processing image {total}')

        image = images[i].numpy()
        label = labels[i].item()
        label = labels[i].item()
        # convert image to string of space-separated values split by comma
        image_str = ','.join(map(str, image.flatten()))

        # print(f'Data {i}: {image_str}')

        command = get_execution_command()
        result = subprocess.run([command, image_str], capture_output=True, text=True)
        output = result.stdout.strip()
        # print(f'Output: {output}')

        # parse output to get predicted label
        # output will have line like "Predicted class: X"
        predicted_label = int(output.split('Predicted class: ')[1])
        # print(f'Label: {label}, Predicted: {predicted_label}')
        total += 1
        correct += (predicted_label == label)
        all_labels.append(label)
        all_preds.append(predicted_label)

accuracy = 100 * correct / total
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f'Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
print(f'Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')

# write evaluation metrics to a file
with open('cpp/cpp_evaluation.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy:.2f}%\n')
    f.write(f'Precision: {precision:.4f}\n')
    f.write(f'Recall: {recall:.4f}\n')
    f.write(f'F1-score: {f1:.4f}\n')
