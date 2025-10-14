# Converting Python PyTorch to C++ for Inference

This guide explains how to translate your Python PyTorch CNN model for MNIST classification to C++ for production inference. We'll cover two main approaches: using LibTorch (PyTorch C++) and ONNX Runtime C++.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Method 1: LibTorch (PyTorch C++)](#method-1-libtorch-pytorch-c)
4. [Method 2: ONNX Runtime C++](#method-2-onnx-runtime-c)
5. [Performance Comparison](#performance-comparison)
6. [Deployment Considerations](#deployment-considerations)
7. [Troubleshooting](#troubleshooting)

---

## Overview

Converting Python models to C++ offers several advantages:

- **Performance**: Faster inference due to compiled code
- **Memory Efficiency**: Lower memory footprint
- **Deployment**: Easier integration into production systems
- **Portability**: Better cross-platform compatibility
- **Dependencies**: Reduced runtime dependencies

### Understanding ONNX (Open Neural Network Exchange)

**ONNX** is an open standard that defines a common set of operators and file format to represent deep learning models. It acts as an intermediate representation (IR) that enables models trained in one framework to be transferred and executed in another.

**Key ONNX Benefits:**

- **Framework Agnostic**: Export from PyTorch, TensorFlow, Keras, scikit-learn, etc.
- **Hardware Optimization**: ONNX Runtime provides optimizations for CPU, GPU, FPGA, and specialized accelerators
- **Production Ready**: Designed specifically for deployment scenarios
- **Smaller Runtime**: Lighter inference engine compared to full frameworks
- **Cross-Platform**: Runs on Windows, Linux, macOS, mobile, and embedded devices

**ONNX vs LibTorch Comparison:**

| Aspect             | ONNX Runtime                       | LibTorch                      |
| ------------------ | ---------------------------------- | ----------------------------- |
| Model Format       | `.onnx` (standardized)             | `.pt` (PyTorch-specific)      |
| Runtime Size       | ~10-50MB                           | ~100-200MB                    |
| Framework Support  | Multi-framework                    | PyTorch only                  |
| Optimization Level | Highly optimized for inference     | General purpose               |
| Hardware Support   | Extensive (CPU, GPU, mobile, edge) | Limited to PyTorch backends   |
| Learning Curve     | Moderate                           | Easy if familiar with PyTorch |

### Your Current Model

Based on your project structure, you have:

- A CNN model defined in `cnn/cnn.py`
- Trained model weights in `cnn/cnn.pth`
- ONNX exported model in `onnx/cnn.onnx`

---

## Prerequisites

### System Requirements

- **Linux/WSL**: Recommended for maximum compatibility
- **macOS (Intel/Apple Silicon M1/M2/M3)**: Supported for both ONNX Runtime and LibTorch. See notes below for Apple Silicon.
- **C++ Compiler**: GCC 7+ or Clang 12+ (Apple Clang recommended on macOS)
- **CMake**: Version 3.12+
- **GPU Support** (optional): CUDA 10.2+ for GPU inference (Linux/WSL/Windows only)

**Apple Silicon (M1/M2/M3) Notes:**

- ONNX Runtime provides official macOS ARM64 builds. Download from the [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases).
- LibTorch provides macOS builds, but ARM64 support may require building from source or using community builds. See [PyTorch LibTorch Mac ARM docs](https://pytorch.org/cppdocs/installing.html).
- For best performance, use Homebrew to install dependencies (e.g., `brew install cmake wget unzip`).
- Some third-party libraries may require Rosetta 2 for x86_64 emulation, but most ONNX/LibTorch features work natively on M-series chips.
- GPU acceleration (CUDA) is not available on macOS; use CPU execution providers.

### Install Development Tools

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential cmake wget unzip

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install cmake wget unzip
```

## macOS (Intel & Apple Silicon) Setup Guide

This section provides a step-by-step guide for running LibTorch and ONNX Runtime C++ inference on macOS, including Apple Silicon (M1/M2/M3).

### 1. Install Development Tools

Use [Homebrew](https://brew.sh/) for easy installation:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew update
brew install cmake wget unzip
```

### 2. Download LibTorch for macOS

#### Intel (x86_64)

Download the official macOS LibTorch build:

```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.1.0.zip
unzip libtorch-macos-2.1.0.zip
```

#### Apple Silicon (ARM64)

- Official ARM64 builds may not be available for all PyTorch versions. You may need to build from source or use a community build. See [PyTorch C++ Mac ARM docs](https://pytorch.org/cppdocs/installing.html).
- For most projects, the x86_64 build works under Rosetta 2, but native ARM64 is recommended for best performance.

### 3. Download ONNX Runtime

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-osx-x86_64-1.16.3.tgz
tar -xzf onnxruntime-osx-x86_64-1.16.3.tgz
```

### 4. Platform-Specific Tips

- Use `arch -arm64` or `arch -x86_64` to run commands under the desired architecture.
- If you encounter compatibility issues, try running your build or app under Rosetta 2: `arch -x86_64 <your_command>`
- For CMake, specify the architecture if needed:
  - Native ARM64: `cmake -DCMAKE_OSX_ARCHITECTURES=arm64 ..`
  - x86_64: `cmake -DCMAKE_OSX_ARCHITECTURES=x86_64 ..`
- Most ONNX Runtime and LibTorch features work natively on M-series chips, but some third-party libraries may require Rosetta 2.
- GPU acceleration (CUDA) is not available on macOS; use CPU execution providers.

---

## Method 1: LibTorch (PyTorch C++)

LibTorch is the C++ distribution of PyTorch that allows you to load and run PyTorch models directly.

### Step 1: Download LibTorch

```bash
# CPU version
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip

# GPU version (if CUDA available)
# wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
```

### Step 2: Export TorchScript Model

First, modify your Python code to export a TorchScript model:

```python
# export_torchscript.py
import torch
from cnn.cnn import CNN

# Load the trained model
model = CNN()
model.load_state_dict(torch.load('cnn/cnn.pth', map_location='cpu'))
model.eval()

# Create example input
example_input = torch.randn(1, 1, 28, 28)

# Trace the model
traced_script_module = torch.jit.trace(model, example_input)

# Save the traced model
traced_script_module.save("cnn/cnn_traced.pt")
print("TorchScript model saved to cnn/cnn_traced.pt")
```

### Step 3: Create C++ Inference Code

Create `inference_libtorch.cpp`:

```cpp
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <memory>
#include <vector>
#include <fstream>
#include <cstdint>

class MNISTInference {
private:
    torch::jit::script::Module model;
    torch::Device device;

public:
    MNISTInference(const std::string& model_path, bool use_gpu = false) 
        : device(use_gpu && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
        
        try {
            // Load the model
            model = torch::jit::load(model_path);
            model.to(device);
            model.eval();
            
            std::cout << "Model loaded successfully on " 
                      << (device.is_cuda() ? "GPU" : "CPU") << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error loading model: " << e.msg() << std::endl;
            throw;
        }
    }

    // Function to load MNIST image from IDX format
    std::vector<float> loadMNISTImage(const std::string& filename, int index) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        // Read IDX header
        uint32_t magic, num_images, rows, cols;
        file.read(reinterpret_cast<char*>(&magic), 4);
        file.read(reinterpret_cast<char*>(&num_images), 4);
        file.read(reinterpret_cast<char*>(&rows), 4);
        file.read(reinterpret_cast<char*>(&cols), 4);

        // Convert from big-endian to little-endian
        magic = __builtin_bswap32(magic);
        num_images = __builtin_bswap32(num_images);
        rows = __builtin_bswap32(rows);
        cols = __builtin_bswap32(cols);

        if (index >= num_images) {
            throw std::runtime_error("Index out of range");
        }

        // Seek to the specific image
        file.seekg(16 + index * rows * cols);

        // Read image data
        std::vector<uint8_t> image_data(rows * cols);
        file.read(reinterpret_cast<char*>(image_data.data()), rows * cols);

        // Convert to float and normalize (0-255 -> 0-1)
        std::vector<float> normalized_data(rows * cols);
        for (int i = 0; i < rows * cols; ++i) {
            normalized_data[i] = static_cast<float>(image_data[i]) / 255.0f;
        }

        return normalized_data;
    }

    int predict(const std::vector<float>& image_data) {
        // Create tensor from image data
        auto tensor = torch::from_blob(
            const_cast<float*>(image_data.data()), 
            {1, 1, 28, 28}, 
            torch::kFloat
        ).to(device);

        // Disable gradient computation for inference
        torch::NoGradGuard no_grad;

        // Forward pass
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor);

        at::Tensor output = model.forward(inputs).toTensor();
        
        // Get prediction (argmax)
        auto prediction = torch::argmax(output, 1);
        
        return prediction.item<int>();
    }

    std::vector<float> predictProbabilities(const std::vector<float>& image_data) {
        auto tensor = torch::from_blob(
            const_cast<float*>(image_data.data()), 
            {1, 1, 28, 28}, 
            torch::kFloat
        ).to(device);

        torch::NoGradGuard no_grad;

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor);

        at::Tensor output = model.forward(inputs).toTensor();
        
        // Apply softmax to get probabilities
        auto probabilities = torch::softmax(output, 1);
        
        // Convert to std::vector
        probabilities = probabilities.to(torch::kCPU);
        std::vector<float> result(probabilities.data_ptr<float>(), 
                                  probabilities.data_ptr<float>() + probabilities.numel());
        
        return result;
    }
};

int main() {
    try {
        // Initialize inference engine
        MNISTInference inference("cnn/cnn_traced.pt", false); // Set true for GPU

        // Load a test image (first image from test set)
        auto image_data = inference.loadMNISTImage("data/t10k-images.idx3-ubyte", 0);

        // Make prediction
        int prediction = inference.predict(image_data);
        std::cout << "Predicted class: " << prediction << std::endl;

        // Get probabilities
        auto probabilities = inference.predictProbabilities(image_data);
        std::cout << "Probabilities:" << std::endl;
        for (int i = 0; i < probabilities.size(); ++i) {
            std::cout << "Class " << i << ": " << probabilities[i] << std::endl;
        }

        // Benchmark inference speed
        auto start = std::chrono::high_resolution_clock::now();
        const int num_runs = 1000;
        
        for (int i = 0; i < num_runs; ++i) {
            inference.predict(image_data);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Average inference time: " 
                  << duration.count() / num_runs << " microseconds" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
```

### Step 4: Create CMakeLists.txt for LibTorch

```cmake
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(mnist_inference_libtorch)

set(CMAKE_CXX_STANDARD 17)

# Find LibTorch
set(CMAKE_PREFIX_PATH "/path/to/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add executable
add_executable(mnist_inference_libtorch inference_libtorch.cpp)
target_link_libraries(mnist_inference_libtorch "${TORCH_LIBRARIES}")

# Copy dlls to output directory on Windows
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET mnist_inference_libtorch
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:mnist_inference_libtorch>)
endif (MSVC)
```

### Step 5: Build and Run LibTorch Version

```bash
# Create build directory
mkdir build_libtorch && cd build_libtorch

# Configure with CMake
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..

# Build
make -j$(nproc)

# Run
./mnist_inference_libtorch
```

---

## Method 2: ONNX Runtime C++

ONNX Runtime provides excellent performance and broader hardware support. It's Microsoft's cross-platform inference engine that's heavily optimized for production workloads.

### Understanding ONNX Runtime Architecture

ONNX Runtime consists of several key components:

1. **Core Engine**: Optimized execution engine with graph-level optimizations
2. **Execution Providers**: Hardware-specific implementations (CPU, CUDA, DirectML, etc.)
3. **Graph Optimizations**: Automatic fusion, constant folding, redundant node elimination
4. **Memory Management**: Efficient memory allocation and reuse strategies

**ONNX Runtime Optimizations:**

- **Graph Optimizations**: Automatically fuses operations (Conv + BatchNorm + ReLU)
- **Kernel Optimizations**: Uses optimized libraries (MLAS, cuDNN, MKL-DNN)
- **Memory Optimizations**: Memory planning and arena-based allocation
- **Quantization Support**: INT8 and other reduced precision formats
- **Threading**: Intelligent intra and inter-op parallelism

### Step 1: Understanding Your ONNX Model

First, let's examine your existing ONNX model to understand its structure:

```python
# analyze_onnx_model.py
import onnx
import onnxruntime as ort
import numpy as np

def analyze_onnx_model(model_path):
    """Comprehensive analysis of your ONNX model"""
    
    # Load and validate the model
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print("✅ ONNX model is valid")
    
    # Model metadata
    print(f"\n📊 Model Information:")
    print(f"IR Version: {model.ir_version}")
    print(f"Producer: {model.producer_name} {model.producer_version}")
    print(f"Domain: {model.domain}")
    print(f"Model Version: {model.model_version}")
    
    # Graph information
    graph = model.graph
    print(f"\n🔗 Graph Structure:")
    print(f"Nodes: {len(graph.node)}")
    print(f"Inputs: {len(graph.input)}")
    print(f"Outputs: {len(graph.output)}")
    print(f"Initializers: {len(graph.initializer)}")
    
    # Input/Output details
    print(f"\n📥 Input Information:")
    for i, input_tensor in enumerate(graph.input):
        if input_tensor.name not in [init.name for init in graph.initializer]:
            shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
            print(f"  Input {i}: {input_tensor.name}")
            print(f"    Shape: {shape}")
            print(f"    Type: {input_tensor.type.tensor_type.elem_type}")
    
    print(f"\n📤 Output Information:")
    for i, output_tensor in enumerate(graph.output):
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"  Output {i}: {output_tensor.name}")
        print(f"    Shape: {shape}")
        print(f"    Type: {output_tensor.type.tensor_type.elem_type}")
    
    # Operators used
    ops = {}
    for node in graph.node:
        if node.op_type in ops:
            ops[node.op_type] += 1
        else:
            ops[node.op_type] = 1
    
    print(f"\n⚙️ Operations Used:")
    for op, count in sorted(ops.items()):
        print(f"  {op}: {count}")
    
    # Test with ONNX Runtime
    print(f"\n🚀 ONNX Runtime Test:")
    session = ort.InferenceSession(model_path)
    
    # Available providers
    print(f"Available Providers: {ort.get_available_providers()}")
    print(f"Current Providers: {session.get_providers()}")
    
    # Performance test
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    # Handle dynamic shapes
    if any(dim is None or (isinstance(dim, str)) for dim in input_shape):
        input_shape = [1, 1, 28, 28]  # MNIST default
    
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Warm-up runs
    for _ in range(10):
        _ = session.run(None, {input_name: dummy_input})
    
    # Timing test
    import time
    start_time = time.time()
    num_runs = 1000
    for _ in range(num_runs):
        outputs = session.run(None, {input_name: dummy_input})
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000  # ms
    print(f"Average inference time: {avg_time:.3f} ms")
    print(f"Throughput: {1000/avg_time:.1f} inferences/second")
    
    return {
        'input_name': input_name,
        'input_shape': input_shape,
        'output_shape': outputs[0].shape,
        'avg_inference_time_ms': avg_time
    }

if __name__ == "__main__":
    model_info = analyze_onnx_model('onnx/cnn.onnx')
    print(f"\n📋 Summary: {model_info}")
```

### Step 2: Download ONNX Runtime

ONNX Runtime offers different builds optimized for different scenarios:

```bash
# Standard CPU build (recommended for most users)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz

# GPU build (if CUDA available)
# wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-gpu-1.16.3.tgz

# Verify the installation
ls onnxruntime-linux-x64-1.16.3/
# Should show: include/ lib/ LICENSE ThirdPartyNotices.txt VERSION_NUMBER
```

**Understanding ONNX Runtime Builds:**

- **Standard**: CPU-only, optimized with MLAS (Microsoft Linear Algebra Subprograms)
- **GPU**: CUDA support with cuDNN optimizations
- **Minimal**: Smaller footprint, limited operator support
- **Mobile**: Optimized for mobile and edge devices

```bash
# Download ONNX Runtime C++
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz
```

### Step 3: Simple ONNX Inference in C++ (Quick Start)

Before diving into advanced usage, here’s a minimal example to run ONNX inference in C++:

#### 1. Download and Extract ONNX Runtime

See the macOS/Linux instructions above for downloading and extracting the ONNX Runtime library.

#### 2. Minimal C++ Example

Create a file named `simple_onnx_infer.cpp`:

```cpp
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "cnn.onnx", session_options);

    // Example input: shape and data must match your model
    std::vector<float> input_tensor_values(784, 0.0f); // MNIST: 1x28x28
    std::vector<int64_t> input_shape{1, 1, 28, 28};

    Ort::AllocatorWithDefaultOptions allocator;
    const char* input_name = session.GetInputName(0, allocator);
    const char* output_name = session.GetOutputName(0, allocator);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        allocator, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size()
    );

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        &input_name, &input_tensor, 1,
        &output_name, 1
    );

    float* output = output_tensors.front().GetTensorMutableData<float>();
    std::cout << "Output[0]: " << output[0] << std::endl;

    return 0;
}
```

#### 3. Compile

Replace paths as needed:

```fish
g++ simple_onnx_infer.cpp -I/path/to/onnxruntime/include -L/path/to/onnxruntime/lib -lonnxruntime -o simple_onnx_infer
```

#### 4. Run

```fish
./simple_onnx_infer
```

This demonstrates a minimal ONNX inference workflow in C++. Expand this for real data and error handling in the enhanced step below.

---

### Step 4: Advanced ONNX Runtime Configuration

Create `advanced_onnx_config.hpp` for sophisticated configurations:

```cpp
#pragma once
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <map>

/**
 * Advanced ONNX Runtime Configuration Manager
 * 
 * Provides easy access to various ONNX Runtime optimizations and configurations
 */
class ONNXRuntimeConfig {
public:
    enum class OptimizationLevel {
        DISABLE_ALL,
        ENABLE_BASIC,
        ENABLE_EXTENDED,
        ENABLE_ALL
    };
    
    enum class ExecutionMode {
        SEQUENTIAL,
        PARALLEL
    };
    
    struct PerformanceConfig {
        int intra_op_num_threads = 0;  // 0 = use default
        int inter_op_num_threads = 1;
        OptimizationLevel opt_level = OptimizationLevel::ENABLE_ALL;
        ExecutionMode exec_mode = ExecutionMode::SEQUENTIAL;
        bool enable_profiling = false;
        std::string profile_file_prefix = "onnx_profile";
    };
    
    struct MemoryConfig {
        bool enable_memory_pattern = true;
        bool enable_cpu_mem_arena = true;
        size_t memory_limit_mb = 0;  // 0 = no limit
    };
    
    struct ProviderConfig {
        std::vector<std::string> preferred_providers = {"CPUExecutionProvider"};
        
        // CUDA specific
        int cuda_device_id = 0;
        size_t cuda_memory_limit = 2ULL * 1024 * 1024 * 1024;  // 2GB
        bool cuda_enable_cudnn_conv_algo_search = true;
        
        // CPU specific
        bool cpu_enable_fast_math = true;
    };
    
    static Ort::SessionOptions createSessionOptions(
        const PerformanceConfig& perf_config = {},
        const MemoryConfig& mem_config = {},
        const ProviderConfig& provider_config = {}) {
        
        Ort::SessionOptions options;
        
        // Performance settings
        if (perf_config.intra_op_num_threads > 0) {
            options.SetIntraOpNumThreads(perf_config.intra_op_num_threads);
        } else {
            options.SetIntraOpNumThreads(std::thread::hardware_concurrency());
        }
        options.SetInterOpNumThreads(perf_config.inter_op_num_threads);
        
        // Optimization level
        GraphOptimizationLevel opt_level;
        switch (perf_config.opt_level) {
            case OptimizationLevel::DISABLE_ALL:
                opt_level = GraphOptimizationLevel::ORT_DISABLE_ALL;
                break;
            case OptimizationLevel::ENABLE_BASIC:
                opt_level = GraphOptimizationLevel::ORT_ENABLE_BASIC;
                break;
            case OptimizationLevel::ENABLE_EXTENDED:
                opt_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
                break;
            case OptimizationLevel::ENABLE_ALL:
                opt_level = GraphOptimizationLevel::ORT_ENABLE_ALL;
                break;
        }
        options.SetGraphOptimizationLevel(opt_level);
        
        // Execution mode
        if (perf_config.exec_mode == ExecutionMode::PARALLEL) {
            options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
        } else {
            options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        }
        
        // Profiling
        if (perf_config.enable_profiling) {
            options.EnableProfiling(perf_config.profile_file_prefix.c_str());
        }
        
        // Memory settings
        if (mem_config.enable_memory_pattern) {
            options.EnableMemPattern();
        } else {
            options.DisableMemPattern();
        }
        
        if (mem_config.enable_cpu_mem_arena) {
            options.EnableCpuMemArena();
        } else {
            options.DisableCpuMemArena();
        }
        
        // Provider-specific configurations
        for (const auto& provider : provider_config.preferred_providers) {
            if (provider == "CUDAExecutionProvider") {
                OrtCUDAProviderOptions cuda_options{};
                cuda_options.device_id = provider_config.cuda_device_id;
                cuda_options.gpu_mem_limit = provider_config.cuda_memory_limit;
                cuda_options.arena_extend_strategy = 0;
                cuda_options.cudnn_conv_algo_search = provider_config.cuda_enable_cudnn_conv_algo_search ? 
                    OrtCudnnConvAlgoSearch::EXHAUSTIVE : OrtCudnnConvAlgoSearch::HEURISTIC;
                
                try {
                    options.AppendExecutionProvider_CUDA(cuda_options);
                } catch (const std::exception& e) {
                    std::cout << "⚠️  CUDA provider not available: " << e.what() << std::endl;
                }
            }
        }
        
        return options;
    }
    
    /**
     * Get optimal configuration based on hardware and use case
     */
    static PerformanceConfig getOptimalConfig(const std::string& use_case = "general") {
        PerformanceConfig config;
        
        if (use_case == "high_throughput") {
            config.intra_op_num_threads = std::thread::hardware_concurrency();
            config.inter_op_num_threads = 1;
            config.opt_level = OptimizationLevel::ENABLE_ALL;
            config.exec_mode = ExecutionMode::PARALLEL;
        } else if (use_case == "low_latency") {
            config.intra_op_num_threads = 1;
            config.inter_op_num_threads = 1;
            config.opt_level = OptimizationLevel::ENABLE_ALL;
            config.exec_mode = ExecutionMode::SEQUENTIAL;
        } else if (use_case == "memory_constrained") {
            config.intra_op_num_threads = 2;
            config.inter_op_num_threads = 1;
            config.opt_level = OptimizationLevel::ENABLE_BASIC;
            config.exec_mode = ExecutionMode::SEQUENTIAL;
        } else {
            // General purpose - balanced
            config.intra_op_num_threads = std::min(4, static_cast<int>(std::thread::hardware_concurrency()));
            config.inter_op_num_threads = 1;
            config.opt_level = OptimizationLevel::ENABLE_ALL;
            config.exec_mode = ExecutionMode::SEQUENTIAL;
        }
        
        return config;
    }
    
    /**
     * Print configuration summary
     */
    static void printConfig(const PerformanceConfig& config) {
        std::cout << "⚙️  ONNX Runtime Configuration:" << std::endl;
        std::cout << "  Intra-op threads: " << config.intra_op_num_threads << std::endl;
        std::cout << "  Inter-op threads: " << config.inter_op_num_threads << std::endl;
        std::cout << "  Optimization level: ";
        switch (config.opt_level) {
            case OptimizationLevel::DISABLE_ALL: std::cout << "DISABLE_ALL"; break;
            case OptimizationLevel::ENABLE_BASIC: std::cout << "ENABLE_BASIC"; break;
            case OptimizationLevel::ENABLE_EXTENDED: std::cout << "ENABLE_EXTENDED"; break;
            case OptimizationLevel::ENABLE_ALL: std::cout << "ENABLE_ALL"; break;
        }
        std::cout << std::endl;
        std::cout << "  Execution mode: " 
                  << (config.exec_mode == ExecutionMode::PARALLEL ? "PARALLEL" : "SEQUENTIAL") 
                  << std::endl;
        std::cout << "  Profiling: " << (config.enable_profiling ? "enabled" : "disabled") << std::endl;
    }
};
```

---

### Step 5: Enhanced C++ ONNX Inference Implementation

Create `inference_onnx_enhanced.cpp` with comprehensive features:

```cpp
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include <memory>
#include <filesystem>
#include <string>
#include <map>

/**
 * Enhanced MNIST ONNX Inference Class
 * 
 * Features:
 * - Comprehensive error handling
 * - Performance optimization
 * - Multiple execution providers
 * - Detailed logging and profiling
 * - Input validation and preprocessing
 * - Batch processing support
 */
class EnhancedMNISTONNXInference {
private:
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memory_info;
    
    // Model metadata
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<ONNXTensorElementDataType> input_types;
    std::vector<ONNXTensorElementDataType> output_types;
    
    // Performance tracking
    struct PerformanceStats {
        size_t total_inferences = 0;
        double total_time_ms = 0.0;
        double min_time_ms = std::numeric_limits<double>::max();
        double max_time_ms = 0.0;
        
        void update(double time_ms) {
            total_inferences++;
            total_time_ms += time_ms;
            min_time_ms = std::min(min_time_ms, time_ms);
            max_time_ms = std::max(max_time_ms, time_ms);
        }
        
        double getAverageTime() const {
            return total_inferences > 0 ? total_time_ms / total_inferences : 0.0;
        }
    } perf_stats;

public:
    /**
     * Constructor with detailed configuration options
     */
    EnhancedMNISTONNXInference(const std::string& model_path, 
                              const std::vector<std::string>& preferred_providers = {"CPUExecutionProvider"}) 
        : env(ORT_LOGGING_LEVEL_WARNING, "EnhancedMNIST"),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        
        initializeSession(model_path, preferred_providers);
        extractModelMetadata();
        printModelInformation();
    }

private:
    void initializeSession(const std::string& model_path, 
                          const std::vector<std::string>& preferred_providers) {
        
        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error("Model file does not exist: " + model_path);
        }
        
        // Configure session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency());
        session_options.SetInterOpNumThreads(1);
        
        // Enable optimizations
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Set execution providers in order of preference
        for (const auto& provider : preferred_providers) {
            try {
                if (provider == "CUDAExecutionProvider") {
                    // CUDA provider options
                    OrtCUDAProviderOptions cuda_options{};
                    cuda_options.device_id = 0;
                    cuda_options.arena_extend_strategy = 0;
                    cuda_options.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024; // 2GB
                    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE;
                    session_options.AppendExecutionProvider_CUDA(cuda_options);
                    std::cout << "✅ CUDA Execution Provider enabled" << std::endl;
                } else if (provider == "CPUExecutionProvider") {
                    // CPU is always available
                    std::cout << "✅ CPU Execution Provider enabled" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cout << "⚠️  Failed to enable " << provider << ": " << e.what() << std::endl;
            }
        }
        
        try {
            session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
            std::cout << "✅ ONNX model loaded successfully from: " << model_path << std::endl;
        } catch (const Ort::Exception& e) {
            throw std::runtime_error("Failed to load ONNX model: " + std::string(e.what()));
        }
    }
    
    void extractModelMetadata() {
        // Extract input information
        size_t input_count = session->GetInputCount();
        for (size_t i = 0; i < input_count; ++i) {
            auto input_name = session->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions{});
            input_names.push_back(std::string(input_name.get()));
            
            auto input_shape_info = session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
            input_shapes.push_back(input_shape_info.GetShape());
            input_types.push_back(input_shape_info.GetElementType());
        }
        
        // Extract output information
        size_t output_count = session->GetOutputCount();
        for (size_t i = 0; i < output_count; ++i) {
            auto output_name = session->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions{});
            output_names.push_back(std::string(output_name.get()));
            
            auto output_shape_info = session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
            output_shapes.push_back(output_shape_info.GetShape());
            output_types.push_back(output_shape_info.GetElementType());
        }
    }
    
    void printModelInformation() {
        std::cout << "\n📊 Model Information:" << std::endl;
        std::cout << "Execution Providers: ";
        for (const auto& provider : session->GetProviders()) {
            std::cout << provider << " ";
        }
        std::cout << std::endl;
        
        std::cout << "\n📥 Inputs:" << std::endl;
        for (size_t i = 0; i < input_names.size(); ++i) {
            std::cout << "  " << i << ": " << input_names[i] << " ";
            std::cout << "Shape: [";
            for (size_t j = 0; j < input_shapes[i].size(); ++j) {
                if (j > 0) std::cout << ", ";
                if (input_shapes[i][j] == -1) {
                    std::cout << "dynamic";
                } else {
                    std::cout << input_shapes[i][j];
                }
            }
            std::cout << "] Type: " << input_types[i] << std::endl;
        }
        
        std::cout << "\n📤 Outputs:" << std::endl;
        for (size_t i = 0; i < output_names.size(); ++i) {
            std::cout << "  " << i << ": " << output_names[i] << " ";
            std::cout << "Shape: [";
            for (size_t j = 0; j < output_shapes[i].size(); ++j) {
                if (j > 0) std::cout << ", ";
                if (output_shapes[i][j] == -1) {
                    std::cout << "dynamic";
                } else {
                    std::cout << output_shapes[i][j];
                }
            }
            std::cout << "] Type: " << output_types[i] << std::endl;
        }
    }

public:
    /**
     * Load and preprocess MNIST image from IDX format
     */
    std::vector<float> loadMNISTImage(const std::string& filename, int index) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        // Read IDX header with proper error checking
        uint32_t magic, num_images, rows, cols;
        file.read(reinterpret_cast<char*>(&magic), 4);
        file.read(reinterpret_cast<char*>(&num_images), 4);
        file.read(reinterpret_cast<char*>(&rows), 4);
        file.read(reinterpret_cast<char*>(&cols), 4);

        // Convert from big-endian to little-endian
        magic = __builtin_bswap32(magic);
        num_images = __builtin_bswap32(num_images);
        rows = __builtin_bswap32(rows);
        cols = __builtin_bswap32(cols);

        // Validate header
        if (magic != 2051) {
            throw std::runtime_error("Invalid IDX file magic number");
        }
        if (rows != 28 || cols != 28) {
            throw std::runtime_error("Expected 28x28 images, got " + 
                                   std::to_string(rows) + "x" + std::to_string(cols));
        }
        if (index >= static_cast<int>(num_images)) {
            throw std::runtime_error("Index " + std::to_string(index) + 
                                   " out of range (0-" + std::to_string(num_images-1) + ")");
        }

        // Seek to the specific image
        file.seekg(16 + index * rows * cols);

        // Read image data
        std::vector<uint8_t> image_data(rows * cols);
        file.read(reinterpret_cast<char*>(image_data.data()), rows * cols);

        // Convert to float and normalize with proper range
        std::vector<float> normalized_data(rows * cols);
        for (size_t i = 0; i < image_data.size(); ++i) {
            normalized_data[i] = static_cast<float>(image_data[i]) / 255.0f;
        }

        return normalized_data;
    }
    
    /**
     * Validate input data before inference
     */
    bool validateInput(const std::vector<float>& image_data) const {
        // Check size
        if (image_data.size() != 784) { // 28x28
            std::cerr << "❌ Input size mismatch: expected 784, got " << image_data.size() << std::endl;
            return false;
        }
        
        // Check value range
        for (size_t i = 0; i < image_data.size(); ++i) {
            if (image_data[i] < 0.0f || image_data[i] > 1.0f) {
                std::cerr << "❌ Input value out of range [0,1] at index " << i 
                          << ": " << image_data[i] << std::endl;
                return false;
            }
        }
        
        return true;
    }

    /**
     * Single image prediction with detailed error handling
     */
    int predict(const std::vector<float>& image_data) {
        if (!validateInput(image_data)) {
            throw std::invalid_argument("Invalid input data");
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            // Prepare input tensor
            std::vector<int64_t> input_shape = {1, 1, 28, 28}; // Batch, Channel, Height, Width
            auto input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, 
                const_cast<float*>(image_data.data()), 
                image_data.size(),
                input_shape.data(), 
                input_shape.size()
            );

            // Prepare input/output names for C API
            std::vector<const char*> input_names_cstr;
            std::vector<const char*> output_names_cstr;
            
            for (const auto& name : input_names) {
                input_names_cstr.push_back(name.c_str());
            }
            for (const auto& name : output_names) {
                output_names_cstr.push_back(name.c_str());
            }

            // Run inference
            auto output_tensors = session->Run(
                Ort::RunOptions{nullptr},
                input_names_cstr.data(),
                &input_tensor,
                1,
                output_names_cstr.data(),
                output_names.size()
            );

            // Process output
            if (output_tensors.empty()) {
                throw std::runtime_error("No output tensors returned");
            }
            
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            
            if (output_shape.size() < 2 || output_shape[1] != 10) {
                throw std::runtime_error("Unexpected output shape");
            }
            
            // Find argmax
            int prediction = 0;
            float max_val = output_data[0];
            for (int i = 1; i < 10; ++i) {
                if (output_data[i] > max_val) {
                    max_val = output_data[i];
                    prediction = i;
                }
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double time_ms = duration.count() / 1000.0;
            
            // Update performance statistics
            perf_stats.update(time_ms);
            
            return prediction;
            
        } catch (const Ort::Exception& e) {
            throw std::runtime_error("ONNX Runtime error: " + std::string(e.what()));
        }
    }

    /**
     * Batch prediction for multiple images
     */
    std::vector<int> predictBatch(const std::vector<std::vector<float>>& batch_images) {
        if (batch_images.empty()) {
            throw std::invalid_argument("Empty batch");
        }
        
        size_t batch_size = batch_images.size();
        
        // Validate all inputs
        for (size_t i = 0; i < batch_size; ++i) {
            if (!validateInput(batch_images[i])) {
                throw std::invalid_argument("Invalid input at batch index " + std::to_string(i));
            }
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            // Prepare batch input data
            std::vector<float> batch_data;
            batch_data.reserve(batch_size * 784);
            
            for (const auto& image : batch_images) {
                batch_data.insert(batch_data.end(), image.begin(), image.end());
            }
            
            // Create input tensor with batch dimension
            std::vector<int64_t> input_shape = {static_cast<int64_t>(batch_size), 1, 28, 28};
            auto input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, 
                batch_data.data(), 
                batch_data.size(),
                input_shape.data(), 
                input_shape.size()
            );

            // Prepare input/output names
            std::vector<const char*> input_names_cstr;
            std::vector<const char*> output_names_cstr;
            
            for (const auto& name : input_names) {
                input_names_cstr.push_back(name.c_str());
            }
            for (const auto& name : output_names) {
                output_names_cstr.push_back(name.c_str());
            }

            // Run batch inference
            auto output_tensors = session->Run(
                Ort::RunOptions{nullptr},
                input_names_cstr.data(),
                &input_tensor,
                1,
                output_names_cstr.data(),
                output_names.size()
            );

            // Process batch output
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            
            std::vector<int> predictions(batch_size);
            for (size_t i = 0; i < batch_size; ++i) {
                int prediction = 0;
                float max_val = output_data[i * 10];
                for (int j = 1; j < 10; ++j) {
                    if (output_data[i * 10 + j] > max_val) {
                        max_val = output_data[i * 10 + j];
                        prediction = j;
                    }
                }
                predictions[i] = prediction;
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double time_ms = duration.count() / 1000.0;
            
            // Update performance statistics (per image)
            perf_stats.update(time_ms / batch_size);
            
            std::cout << "📊 Batch inference completed: " << batch_size 
                      << " images in " << time_ms << " ms"
                      << " (" << time_ms/batch_size << " ms/image)" << std::endl;
            
            return predictions;
            
        } catch (const Ort::Exception& e) {
            throw std::runtime_error("ONNX Runtime batch error: " + std::string(e.what()));
        }
    }
    
    /**
     * Get detailed probabilities for all classes
     */
    std::vector<float> predictProbabilities(const std::vector<float>& image_data) {
        if (!validateInput(image_data)) {
            throw std::invalid_argument("Invalid input data");
        }
        
        try {
            std::vector<int64_t> input_shape = {1, 1, 28, 28};
            auto input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, 
                const_cast<float*>(image_data.data()), 
                image_data.size(),
                input_shape.data(), 
                input_shape.size()
            );

            std::vector<const char*> input_names_cstr;
            std::vector<const char*> output_names_cstr;
            
            for (const auto& name : input_names) {
                input_names_cstr.push_back(name.c_str());
            }
            for (const auto& name : output_names) {
                output_names_cstr.push_back(name.c_str());
            }

            auto output_tensors = session->Run(
                Ort::RunOptions{nullptr},
                input_names_cstr.data(),
                &input_tensor,
                1,
                output_names_cstr.data(),
                output_names.size()
            );

            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            
            // Apply softmax for probabilities
            std::vector<float> probabilities(10);
            float sum = 0.0f;
            
            // Find max for numerical stability
            float max_val = *std::max_element(output_data, output_data + 10);
            
            for (int i = 0; i < 10; ++i) {
                probabilities[i] = std::exp(output_data[i] - max_val);
                sum += probabilities[i];
            }
            
            // Normalize to get probabilities
            for (auto& prob : probabilities) {
                prob /= sum;
            }

            return probabilities;
            
        } catch (const Ort::Exception& e) {
            throw std::runtime_error("ONNX Runtime probability error: " + std::string(e.what()));
        }
    }
    
    /**
     * Get performance statistics
     */
    void printPerformanceStats() const {
        std::cout << "\n📈 Performance Statistics:" << std::endl;
        std::cout << "Total inferences: " << perf_stats.total_inferences << std::endl;
        std::cout << "Average time: " << perf_stats.getAverageTime() << " ms" << std::endl;
        std::cout << "Min time: " << perf_stats.min_time_ms << " ms" << std::endl;
        std::cout << "Max time: " << perf_stats.max_time_ms << " ms" << std::endl;
        std::cout << "Throughput: " << 1000.0 / perf_stats.getAverageTime() << " inferences/second" << std::endl;
    }
    
    /**
     * Benchmark with detailed analysis
     */
    void benchmark(int num_runs = 1000) {
        std::cout << "\n🔬 Starting benchmark with " << num_runs << " runs..." << std::endl;
        
        // Load a test image
        auto image_data = loadMNISTImage("data/t10k-images.idx3-ubyte", 0);
        
        // Warm-up runs
        std::cout << "Warming up..." << std::endl;
        for (int i = 0; i < 10; ++i) {
            predict(image_data);
        }
        
        // Reset statistics
        perf_stats = PerformanceStats{};
        
        // Actual benchmark
        std::cout << "Running benchmark..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_runs; ++i) {
            predict(image_data);
            if ((i + 1) % (num_runs / 10) == 0) {
                std::cout << "Progress: " << ((i + 1) * 100 / num_runs) << "%" << std::endl;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "\n🏁 Benchmark completed!" << std::endl;
        std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;
        printPerformanceStats();
    }
};

/**
 * Comprehensive example usage with multiple scenarios
 */
int main() {
    try {
        std::cout << "🚀 Enhanced MNIST ONNX Inference Demo" << std::endl;
        std::cout << "=====================================" << std::endl;
        
        // Initialize with preferred execution providers
        std::vector<std::string> providers = {"CPUExecutionProvider"};
        // For GPU: providers = {"CUDAExecutionProvider", "CPUExecutionProvider"};
        
        EnhancedMNISTONNXInference inference("onnx/cnn.onnx", providers);
        
        // 1. Single image prediction
        std::cout << "\n1️⃣ Single Image Prediction:" << std::endl;
        auto image_data = inference.loadMNISTImage("data/t10k-images.idx3-ubyte", 0);
        
        int prediction = inference.predict(image_data);
        std::cout << "Predicted class: " << prediction << std::endl;
        
        // Get detailed probabilities
        auto probabilities = inference.predictProbabilities(image_data);
        std::cout << "Class probabilities:" << std::endl;
        for (int i = 0; i < 10; ++i) {
            std::cout << "  Class " << i << ": " << std::fixed << std::setprecision(4) 
                      << probabilities[i] << " (" << (probabilities[i] * 100) << "%)" << std::endl;
        }
        
        // 2. Batch prediction
        std::cout << "\n2️⃣ Batch Prediction:" << std::endl;
        std::vector<std::vector<float>> batch_images;
        for (int i = 0; i < 5; ++i) {
            batch_images.push_back(inference.loadMNISTImage("data/t10k-images.idx3-ubyte", i));
        }
        
        auto batch_predictions = inference.predictBatch(batch_images);
        std::cout << "Batch predictions: ";
        for (size_t i = 0; i < batch_predictions.size(); ++i) {
            std::cout << batch_predictions[i];
            if (i < batch_predictions.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        
        // 3. Performance benchmark
        std::cout << "\n3️⃣ Performance Benchmark:" << std::endl;
        inference.benchmark(1000);
        
        // 4. Error handling demonstration
        std::cout << "\n4️⃣ Error Handling Demo:" << std::endl;
        try {
            // Try to load non-existent image
            inference.loadMNISTImage("data/t10k-images.idx3-ubyte", 999999);
        } catch (const std::exception& e) {
            std::cout << "✅ Caught expected error: " << e.what() << std::endl;
        }
        
        try {
            // Try with invalid input size
            std::vector<float> invalid_input(100, 0.5f);  // Wrong size
            inference.predict(invalid_input);
        } catch (const std::exception& e) {
            std::cout << "✅ Caught expected error: " << e.what() << std::endl;
        }
        
        std::cout << "\n🎉 Demo completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Fatal error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
```

### Step 6: ONNX Model Optimization and Quantization

ONNX Runtime provides several optimization techniques to improve performance:

#### Model Optimization Script

Create `optimize_onnx_model.py` to optimize your ONNX model:

```python
import onnx
import onnxruntime as ort
from onnxruntime.tools import optimizer
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
import os

def optimize_onnx_model(input_model_path, output_dir="onnx_optimized"):
    """
    Comprehensive ONNX model optimization pipeline
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔧 Starting ONNX Model Optimization Pipeline")
    print("=" * 50)
    
    # 1. Basic Graph Optimization
    print("\n1️⃣ Applying Graph Optimizations...")
    
    # Load the original model
    model = onnx.load(input_model_path)
    
    # Apply graph optimizations
    optimized_model_path = os.path.join(output_dir, "cnn_optimized.onnx")
    
    # Create session options with optimization
    sess_options = ort.SessionOptions()
    sess_options.optimized_model_filepath = optimized_model_path
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Create session to trigger optimization
    session = ort.InferenceSession(input_model_path, sess_options)
    
    print(f"✅ Optimized model saved to: {optimized_model_path}")
    
    # 2. Dynamic Quantization
    print("\n2️⃣ Applying Dynamic Quantization...")
    
    quantized_model_path = os.path.join(output_dir, "cnn_quantized.onnx")
    
    try:
        quantize_dynamic(
            model_input=optimized_model_path,
            model_output=quantized_model_path,
            weight_type=QuantType.QUInt8,  # or QInt8
            nodes_to_quantize=None,  # Quantize all supported nodes
            nodes_to_exclude=None,
            optimize_model=True,
            use_external_data_format=False,
            reduce_range=True  # For better compatibility
        )
        print(f"✅ Quantized model saved to: {quantized_model_path}")
    except Exception as e:
        print(f"⚠️  Quantization failed: {e}")
        quantized_model_path = optimized_model_path
    
    # Performance comparison and analysis code...
    # [Additional code for comparison would go here]
    
    return {
        'optimized_model': optimized_model_path,
        'quantized_model': quantized_model_path if os.path.exists(quantized_model_path) else None
    }

if __name__ == "__main__":
    optimization_results = optimize_onnx_model('onnx/cnn.onnx')
```

```cpp
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <algorithm>

class MNISTONNXInference {
private:
    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memory_info;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<int64_t> input_shape;

public:
    MNISTONNXInference(const std::string& model_path) 
        : env(ORT_LOGGING_LEVEL_WARNING, "MNIST"),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
          session(env, model_path.c_str(), Ort::SessionOptions{nullptr}) {
        
        // Get input/output names and shapes
        auto input_count = session.GetInputCount();
        auto output_count = session.GetOutputCount();
        
        for (size_t i = 0; i < input_count; ++i) {
            auto input_name = session.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            input_names.push_back(input_name.release());
            
            auto input_shape_info = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
            input_shape = input_shape_info.GetShape();
        }
        
        for (size_t i = 0; i < output_count; ++i) {
            auto output_name = session.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            output_names.push_back(output_name.release());
        }
        
        std::cout << "ONNX model loaded successfully" << std::endl;
        std::cout << "Input shape: ";
        for (auto dim : input_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    }

    ~MNISTONNXInference() {
        for (auto name : input_names) {
            delete[] name;
        }
        for (auto name : output_names) {
            delete[] name;
        }
    }

    std::vector<float> loadMNISTImage(const std::string& filename, int index) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        // Read IDX header
        uint32_t magic, num_images, rows, cols;
        file.read(reinterpret_cast<char*>(&magic), 4);
        file.read(reinterpret_cast<char*>(&num_images), 4);
        file.read(reinterpret_cast<char*>(&rows), 4);
        file.read(reinterpret_cast<char*>(&cols), 4);

        // Convert from big-endian to little-endian
        magic = __builtin_bswap32(magic);
        num_images = __builtin_bswap32(num_images);
        rows = __builtin_bswap32(rows);
        cols = __builtin_bswap32(cols);

        if (index >= num_images) {
            throw std::runtime_error("Index out of range");
        }

        // Seek to the specific image
        file.seekg(16 + index * rows * cols);

        // Read image data
        std::vector<uint8_t> image_data(rows * cols);
        file.read(reinterpret_cast<char*>(image_data.data()), rows * cols);

        // Convert to float and normalize
        std::vector<float> normalized_data(rows * cols);
        for (int i = 0; i < rows * cols; ++i) {
            normalized_data[i] = static_cast<float>(image_data[i]) / 255.0f;
        }

        return normalized_data;
    }

    int predict(const std::vector<float>& image_data) {
        // Create input tensor
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, 
            const_cast<float*>(image_data.data()), 
            image_data.size(),
            input_shape.data(), 
            input_shape.size()
        );

        // Run inference
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            &input_tensor,
            1,
            output_names.data(),
            1
        );

        // Get prediction
        float* output_data = output_tensors.front().GetTensorMutableData<float>();
        auto output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
        
        // Find argmax
        int prediction = 0;
        float max_val = output_data[0];
        for (int i = 1; i < output_shape[1]; ++i) {
            if (output_data[i] > max_val) {
                max_val = output_data[i];
                prediction = i;
            }
        }

        return prediction;
    }

    std::vector<float> predictProbabilities(const std::vector<float>& image_data) {
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, 
            const_cast<float*>(image_data.data()), 
            image_data.size(),
            input_shape.data(), 
            input_shape.size()
        );

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            &input_tensor,
            1,
            output_names.data(),
            1
        );

        float* output_data = output_tensors.front().GetTensorMutableData<float>();
        auto output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
        
        // Apply softmax
        std::vector<float> probabilities(output_shape[1]);
        float sum = 0.0f;
        
        // Find max for numerical stability
        float max_val = *std::max_element(output_data, output_data + output_shape[1]);
        
        for (int i = 0; i < output_shape[1]; ++i) {
            probabilities[i] = std::exp(output_data[i] - max_val);
            sum += probabilities[i];
        }
        
        // Normalize
        for (auto& prob : probabilities) {
            prob /= sum;
        }

        return probabilities;
    }
};

int main() {
    try {
        // Initialize ONNX inference
        MNISTONNXInference inference("onnx/cnn.onnx");

        // Load a test image
        auto image_data = inference.loadMNISTImage("data/t10k-images.idx3-ubyte", 0);

        // Make prediction
        int prediction = inference.predict(image_data);
        std::cout << "Predicted class: " << prediction << std::endl;

        // Get probabilities
        auto probabilities = inference.predictProbabilities(image_data);
        std::cout << "Probabilities:" << std::endl;
        for (int i = 0; i < probabilities.size(); ++i) {
            std::cout << "Class " << i << ": " << probabilities[i] << std::endl;
        }

        // Benchmark inference speed
        auto start = std::chrono::high_resolution_clock::now();
        const int num_runs = 1000;
        
        for (int i = 0; i < num_runs; ++i) {
            inference.predict(image_data);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Average inference time: " 
                  << duration.count() / num_runs << " microseconds" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
```

### Step 3: Create CMakeLists.txt for ONNX Runtime

```cmake
cmake_minimum_required(VERSION 3.12)
project(mnist_inference_onnx)

set(CMAKE_CXX_STANDARD 17)

# Set ONNX Runtime path
set(ONNXRUNTIME_ROOT_PATH "/path/to/onnxruntime-linux-x64-1.16.3")

# Include directories
include_directories(${ONNXRUNTIME_ROOT_PATH}/include)

# Add executable
add_executable(mnist_inference_onnx inference_onnx.cpp)

# Link libraries
target_link_libraries(mnist_inference_onnx 
    ${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.so)

# Set rpath for runtime library finding
set_target_properties(mnist_inference_onnx PROPERTIES
    INSTALL_RPATH ${ONNXRUNTIME_ROOT_PATH}/lib)
```

### Step 4: Build and Run ONNX Version

```bash
# Create build directory
mkdir build_onnx && cd build_onnx

# Configure with CMake
cmake -DONNXRUNTIME_ROOT_PATH=/path/to/onnxruntime-linux-x64-1.16.3 ..

# Build
make -j$(nproc)

# Set library path and run
export LD_LIBRARY_PATH=/path/to/onnxruntime-linux-x64-1.16.3/lib:$LD_LIBRARY_PATH
./mnist_inference_onnx
```

---

## Performance Comparison

### Typical Performance Metrics

| Method           | Inference Time (µs) | Memory Usage (MB) | Model Size (MB) | Setup Complexity |
| ---------------- | ------------------- | ----------------- | --------------- | ---------------- |
| Python PyTorch   | 800-1200            | 150-200           | 0.5             | Low              |
| LibTorch C++     | 200-400             | 80-120            | 0.5             | Medium           |
| ONNX Runtime C++ | 150-300             | 60-100            | 0.5             | Medium           |

### Benchmarking Script

Create `benchmark.cpp` to compare performance:

```cpp
#include <chrono>
#include <iostream>
#include <vector>

class Benchmark {
public:
    static void measureInference(std::function<void()> inference_func, 
                               const std::string& method_name, 
                               int num_runs = 1000) {
        
        // Warm up
        for (int i = 0; i < 10; ++i) {
            inference_func();
        }
        
        // Actual benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_runs; ++i) {
            inference_func();
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_time = static_cast<double>(duration.count()) / num_runs;
        
        std::cout << method_name << " - Average inference time: " 
                  << avg_time << " µs" << std::endl;
        std::cout << method_name << " - Throughput: " 
                  << 1000000.0 / avg_time << " inferences/second" << std::endl;
    }
};
```

---

## Deployment Considerations

### 1. Memory Management

- Use memory pools for frequent allocations
- Pre-allocate tensors when possible
- Monitor memory usage in production

### 2. Thread Safety

- Both LibTorch and ONNX Runtime support multi-threading
- Create separate sessions for each thread
- Use thread-local storage for model instances

### 3. Error Handling

```cpp
try {
    // Inference code
} catch (const std::exception& e) {
    // Log error and handle gracefully
    std::cerr << "Inference error: " << e.what() << std::endl;
    return default_prediction;
}
```

### 4. Input Validation

```cpp
bool validateInput(const std::vector<float>& input) {
    if (input.size() != 28 * 28) {
        return false;
    }
    
    for (float val : input) {
        if (val < 0.0f || val > 1.0f) {
            return false;
        }
    }
    
    return true;
}
```

### 5. Production Optimizations

```cpp
class ProductionMNISTInference {
private:
    // Pre-allocated tensors
    std::vector<float> input_buffer;
    std::vector<float> output_buffer;
    
public:
    ProductionMNISTInference() : input_buffer(28 * 28), output_buffer(10) {
        // Pre-allocate buffers
    }
    
    int predict(const std::vector<float>& image_data) {
        // Reuse pre-allocated buffers
        std::copy(image_data.begin(), image_data.end(), input_buffer.begin());
        
        // Run inference with pre-allocated memory
        // ...
        
        return prediction;
    }
};
```

### 6. Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM ubuntu:20.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy application
COPY . /app
WORKDIR /app

# Build application
RUN mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc)

# Run
CMD ["./build/mnist_inference_onnx"]
```

---

## Conclusion

Both LibTorch and ONNX Runtime provide excellent performance improvements over Python:

**Choose LibTorch when:**

- You want to stay within the PyTorch ecosystem
- You need dynamic graph capabilities
- You're already familiar with PyTorch APIs

**Choose ONNX Runtime when:**

- You want maximum performance
- You need broad hardware support (CPU, GPU, mobile, edge devices)
- You want to optimize for deployment across different platforms
- You need the smallest memory footprint

The C++ implementations typically provide 3-5x performance improvements over Python while using significantly less memory, making them ideal for production deployment scenarios.

---

## Troubleshooting

This section covers common issues and solutions when converting Python PyTorch models to C++.

### 1. Build and Compilation Issues

#### CMake Cannot Find LibTorch

**Problem:**

```text
CMake Error: Could not find a package configuration file provided by "Torch"
```

**Solution:**

```bash
# Ensure you've downloaded LibTorch
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip

# Set the correct path in CMake
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..

# Or set environment variable
export CMAKE_PREFIX_PATH=/absolute/path/to/libtorch
```

#### ONNX Runtime Not Found

**Problem:**

```text
fatal error: onnxruntime_cxx_api.h: No such file or directory
```

**Solution:**

```bash
# Download ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz

# Update CMakeLists.txt with correct path
set(ONNXRUNTIME_ROOT_PATH "/absolute/path/to/onnxruntime-linux-x64-1.16.3")
```

#### ABI Compatibility Issues

**Problem:**

```text
undefined symbol: _ZN2at6detail20DynamicCUDAInterface...
```

**Solution:**

```bash
# Use the correct ABI version
# For GCC 5+ (new ABI):
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip

# For older GCC (pre-cxx11 ABI):
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.1.0%2Bcpu.zip

# Check your GCC version
gcc --version

# Force specific ABI in CMake
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)  # For new ABI
# or
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)  # For old ABI
```

### 2. Model Loading Issues

#### TorchScript Model Loading Fails

**Problem:**

```text
Error loading model: Cannot load model from file
```

**Solution:**

```python
# Ensure proper model export in Python
import torch
from cnn.cnn import CNN

model = CNN()
model.load_state_dict(torch.load('cnn/cnn.pth', map_location='cpu'))
model.eval()

# Use a representative input
example_input = torch.randn(1, 1, 28, 28)

# Trace the model
try:
    traced_model = torch.jit.trace(model, example_input)
    # Verify the traced model works
    test_output = traced_model(example_input)
    print("Traced model working correctly")
    
    # Save with explicit format
    traced_model.save("cnn/cnn_traced.pt")
    print("Model saved successfully")
except Exception as e:
    print(f"Tracing failed: {e}")

# Alternative: Use torch.jit.script for more complex models
scripted_model = torch.jit.script(model)
scripted_model.save("cnn/cnn_scripted.pt")
```

**Debug Model Loading in C++:**

```cpp
try {
    // Add verbose error checking
    std::cout << "Attempting to load model from: " << model_path << std::endl;
    
    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Model file does not exist: " + model_path);
    }
    
    model = torch::jit::load(model_path);
    std::cout << "Model loaded successfully" << std::endl;
    
    // Test with dummy input
    auto test_input = torch::randn({1, 1, 28, 28});
    auto test_output = model.forward({test_input}).toTensor();
    std::cout << "Model forward pass successful, output shape: " 
              << test_output.sizes() << std::endl;
              
} catch (const c10::Error& e) {
    std::cerr << "LibTorch error: " << e.msg() << std::endl;
    std::cerr << "Error type: " << e.what() << std::endl;
    throw;
} catch (const std::exception& e) {
    std::cerr << "Standard error: " << e.what() << std::endl;
    throw;
}
```

#### ONNX Model Issues

**Problem:**

```
Failed to load model: Invalid model format
```

**Solution:**

```python
# Verify ONNX model in Python first
import onnx
import onnxruntime as ort

# Check model validity
try:
    onnx_model = onnx.load('onnx/cnn.onnx')
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid")
    
    # Test with ONNX Runtime
    session = ort.InferenceSession('onnx/cnn.onnx')
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"Input name: {input_name}")
    print(f"Output name: {output_name}")
    print(f"Input shape: {session.get_inputs()[0].shape}")
    print(f"Output shape: {session.get_outputs()[0].shape}")
    
except Exception as e:
    print(f"ONNX model error: {e}")
    
    # Re-export with specific settings
    import torch
    from cnn.cnn import CNN
    
    model = CNN()
    model.load_state_dict(torch.load('cnn/cnn.pth', map_location='cpu'))
    model.eval()
    
    dummy_input = torch.randn(1, 1, 28, 28)
    
    torch.onnx.export(
        model,
        dummy_input,
        'onnx/cnn_fixed.onnx',
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
```

### 3. Runtime Issues

#### Memory Allocation Errors

**Problem:**

```
RuntimeError: CUDA out of memory
```

**Solution:**

```cpp
// For GPU memory issues
class MemoryEfficientInference {
private:
    bool use_gpu;
    torch::Device device;
    
public:
    MemoryEfficientInference(bool gpu = false) 
        : use_gpu(gpu && torch::cuda::is_available()) 
        , device(use_gpu ? torch::kCUDA : torch::kCPU) {
        
        if (use_gpu) {
            // Clear GPU cache
            torch::cuda::empty_cache();
            
            // Check available memory
            auto memory_info = torch::cuda::getCurrentDeviceProperties();
            std::cout << "GPU memory available: " 
                      << memory_info->totalGlobalMem / (1024*1024) << " MB" << std::endl;
        }
    }
    
    int predict(const std::vector<float>& image_data) {
        torch::NoGradGuard no_grad;  // Important: disable gradients
        
        auto tensor = torch::from_blob(
            const_cast<float*>(image_data.data()), 
            {1, 1, 28, 28}, 
            torch::kFloat
        ).to(device);
        
        // Process in smaller batches if needed
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor);
        
        auto output = model.forward(inputs).toTensor();
        auto prediction = torch::argmax(output, 1);
        
        // Explicitly move to CPU and get value
        auto cpu_prediction = prediction.to(torch::kCPU);
        return cpu_prediction.item<int>();
    }
};
```

#### Data Type Mismatches

**Problem:**

```
RuntimeError: Expected tensor to have dtype Float but got dtype Double
```

**Solution:**

```cpp
// Ensure correct data types
std::vector<float> preprocessImage(const std::vector<uint8_t>& raw_data) {
    std::vector<float> processed(raw_data.size());
    
    for (size_t i = 0; i < raw_data.size(); ++i) {
        // Ensure float conversion and normalization
        processed[i] = static_cast<float>(raw_data[i]) / 255.0f;
    }
    
    return processed;
}

// Create tensor with explicit type
auto tensor = torch::from_blob(
    const_cast<float*>(image_data.data()), 
    {1, 1, 28, 28}, 
    torch::dtype(torch::kFloat32)  // Explicit dtype
).to(device);
```

### 4. Performance Issues

#### Slow Inference Speed

**Problem:** C++ inference is not significantly faster than Python.

**Solution:**

```cpp
class OptimizedInference {
private:
    // Pre-allocate tensors
    torch::Tensor input_tensor;
    torch::Tensor output_tensor;
    bool tensors_allocated = false;
    
public:
    OptimizedInference() {
        // Pre-allocate tensors once
        input_tensor = torch::empty({1, 1, 28, 28}, torch::kFloat);
        tensors_allocated = true;
    }
    
    int fastPredict(const std::vector<float>& image_data) {
        // Reuse pre-allocated tensor
        std::memcpy(
            input_tensor.data_ptr<float>(), 
            image_data.data(), 
            image_data.size() * sizeof(float)
        );
        
        torch::NoGradGuard no_grad;
        
        // Use in-place operations where possible
        auto output = model.forward({input_tensor}).toTensor();
        
        // Efficient argmax
        return output.argmax(1).item<int>();
    }
};

// Enable optimization flags during compilation
// g++ -O3 -march=native -mtune=native ...
```

#### Thread Safety Issues

**Problem:** Crashes or incorrect results in multi-threaded environment.

**Solution:**

```cpp
#include <thread>
#include <mutex>

class ThreadSafeInference {
private:
    static std::mutex model_mutex;
    torch::jit::script::Module model;
    
public:
    int threadSafePredict(const std::vector<float>& image_data) {
        std::lock_guard<std::mutex> lock(model_mutex);
        
        torch::NoGradGuard no_grad;
        
        auto tensor = torch::from_blob(
            const_cast<float*>(image_data.data()), 
            {1, 1, 28, 28}, 
            torch::kFloat
        );
        
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor);
        
        auto output = model.forward(inputs).toTensor();
        return torch::argmax(output, 1).item<int>();
    }
};

std::mutex ThreadSafeInference::model_mutex;

// Better approach: Create separate model instances per thread
class MultiThreadInference {
private:
    thread_local static std::unique_ptr<torch::jit::script::Module> thread_model;
    std::string model_path;
    
public:
    MultiThreadInference(const std::string& path) : model_path(path) {}
    
    int predict(const std::vector<float>& image_data) {
        if (!thread_model) {
            thread_model = std::make_unique<torch::jit::script::Module>(
                torch::jit::load(model_path)
            );
        }
        
        // Now each thread has its own model instance
        torch::NoGradGuard no_grad;
        
        auto tensor = torch::from_blob(
            const_cast<float*>(image_data.data()), 
            {1, 1, 28, 28}, 
            torch::kFloat
        );
        
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor);
        
        auto output = thread_model->forward(inputs).toTensor();
        return torch::argmax(output, 1).item<int>();
    }
};

thread_local std::unique_ptr<torch::jit::script::Module> 
    MultiThreadInference::thread_model = nullptr;
```

### 5. Debugging Tools and Techniques

#### Enable Verbose Logging

```cpp
// For LibTorch debugging
#include <torch/torch.h>

void enableDebugLogging() {
    // Enable detailed logging
    torch::manual_seed(42);  // For reproducible results
    
    // Set log level (if using newer versions)
    // torch::jit::getExecutorMode() = torch::jit::ExecutorMode::SIMPLE;
    
    std::cout << "PyTorch version: " << TORCH_VERSION << std::endl;
    std::cout << "CUDA available: " << torch::cuda::is_available() << std::endl;
    std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
}

// For ONNX Runtime debugging
void enableONNXDebugLogging() {
    // Create session with verbose logging
    Ort::SessionOptions session_options;
    session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);
    session_options.EnableProfiling("onnx_profile");
    
    Ort::Session session(env, model_path.c_str(), session_options);
}
```

#### Model Inspection Tools

```cpp
void inspectModel(const torch::jit::script::Module& model) {
    std::cout << "Model structure:" << std::endl;
    
    // Print model parameters
    for (const auto& param : model.named_parameters()) {
        std::cout << "Parameter: " << param.name 
                  << ", Shape: " << param.value.sizes() 
                  << ", Type: " << param.value.dtype() << std::endl;
    }
    
    // Print model buffers
    for (const auto& buffer : model.named_buffers()) {
        std::cout << "Buffer: " << buffer.name 
                  << ", Shape: " << buffer.value.sizes() << std::endl;
    }
}

void validateInputOutput(const torch::Tensor& input, const torch::Tensor& output) {
    std::cout << "Input validation:" << std::endl;
    std::cout << "  Shape: " << input.sizes() << std::endl;
    std::cout << "  Type: " << input.dtype() << std::endl;
    std::cout << "  Device: " << input.device() << std::endl;
    std::cout << "  Min value: " << torch::min(input).item<float>() << std::endl;
    std::cout << "  Max value: " << torch::max(input).item<float>() << std::endl;
    std::cout << "  Mean: " << torch::mean(input).item<float>() << std::endl;
    
    std::cout << "Output validation:" << std::endl;
    std::cout << "  Shape: " << output.sizes() << std::endl;
    std::cout << "  Type: " << output.dtype() << std::endl;
    std::cout << "  Values: ";
    for (int i = 0; i < output.size(1); ++i) {
        std::cout << output[0][i].item<float>() << " ";
    }
    std::cout << std::endl;
}
```

### 6. Common Error Messages and Solutions

| Error Message                                  | Cause                   | Solution                               |
| ---------------------------------------------- | ----------------------- | -------------------------------------- |
| `undefined symbol: _ZN3c105ErrorC1E`           | ABI mismatch            | Use correct LibTorch ABI version       |
| `CUDA driver version is insufficient`          | CUDA version mismatch   | Update CUDA drivers or use CPU version |
| `Expected tensor to be on device`              | Device mismatch         | Ensure all tensors are on same device  |
| `RuntimeError: shape mismatch`                 | Input shape incorrect   | Verify input preprocessing             |
| `Model was not exported with save_code`        | Incomplete model export | Re-export with proper settings         |
| `libonnxruntime.so: cannot open shared object` | Missing library path    | Set LD_LIBRARY_PATH correctly          |

### 7. Testing and Validation

Create a comprehensive test to validate your C++ implementation:

```cpp
#include <cassert>
#include <cmath>

class ModelValidator {
public:
    static void validateAgainstPython() {
        // Load same image used in Python
        auto image_data = loadMNISTImage("data/t10k-images.idx3-ubyte", 0);
        
        // Expected output from Python (copy from your Python script)
        std::vector<float> expected_probabilities = {
            0.0001, 0.0002, 0.0015, 0.8523, 0.0003,
            0.0012, 0.1234, 0.0189, 0.0019, 0.0002
        };
        int expected_prediction = 3;
        
        // Test C++ implementation
        MNISTInference inference("cnn/cnn_traced.pt");
        auto cpp_probabilities = inference.predictProbabilities(image_data);
        int cpp_prediction = inference.predict(image_data);
        
        // Validate prediction
        assert(cpp_prediction == expected_prediction);
        
        // Validate probabilities (within tolerance)
        const float tolerance = 1e-4;
        for (size_t i = 0; i < expected_probabilities.size(); ++i) {
            float diff = std::abs(cpp_probabilities[i] - expected_probabilities[i]);
            if (diff > tolerance) {
                std::cerr << "Probability mismatch at index " << i 
                          << ": expected " << expected_probabilities[i]
                          << ", got " << cpp_probabilities[i] << std::endl;
                assert(false);
            }
        }
        
        std::cout << "✅ Model validation passed!" << std::endl;
    }
};
```

This troubleshooting section should help you resolve most common issues when converting your PyTorch models to C++. Remember to always test your C++ implementation against your Python version to ensure correctness.
