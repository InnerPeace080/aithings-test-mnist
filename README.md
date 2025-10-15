# Experiment with pytorch MNIST and ONNX

## Setup

- create and activate conda environment

    ```shell
    conda env create -f environment.yml
    conda activate mnist
    ```

## pytorch with MNIST

- Train and infer with simple neural network. Weights are saved to `simple_net.pth`. Result store in `simple_net_evaluation.txt`.

    ```shell
    python 1.1.test_simple_net.py
    ```

- Train and infer with convolutional neural network. Weights are saved to `cnn.pth`. Whole model saved to `cnn_model.pth`. Result store in `conv_net_evaluation.txt`.

    ```shell
    python 1.2.test_cnn.py
    ```

## ONNX

- Export the trained model `cnn_model.pth` to ONNX format `cnn.onnx` and run inference with ONNX Runtime

    ```shell
    python 1.3test_onnx.py
    ```

## C++

- build

    ```shell
    pushd cpp
    mkdir build
    popd
    ```

- run evaluation

    ```shell
    python 2.test_cpp.py
    ```
