
#if defined(__linux__)
#include <linux/limits.h>
#endif

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

#include <onnxruntime_cxx_api.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

std::string get_executable_dir() {
  // for linux and darwin
#if defined(__linux__)
  char    result[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
  if (count != -1) {
    std::string exec_path(result, count);
    size_t      last_slash = exec_path.find_last_of('/');
    if (last_slash != std::string::npos) {
      return exec_path.substr(0, last_slash);
    }
  }

#elif defined(__APPLE__)
  char     result[PATH_MAX];
  uint32_t size = sizeof(result);
  if (_NSGetExecutablePath(result, &size) == 0) {
    std::string exec_path(result);
    size_t      last_slash = exec_path.find_last_of('/');
    if (last_slash != std::string::npos) {
      return exec_path.substr(0, last_slash);
    }
  }
#endif
  return "";
}

int main(int argc, char* argv[]) {
  Ort::Env            env(ORT_LOGGING_LEVEL_WARNING, "test");
  Ort::SessionOptions session_options;

  const std::string exe_path = get_executable_dir();
  Ort::Session      session(env, (exe_path + "/../onnx/cnn.onnx").c_str(), session_options);

  // input shape
  std::vector<int64_t> input_shape = {1, 1, 28, 28};
  std::vector<float>   input_tensor_values;

  // Read input tensor values from string (first argument)
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <input_data_string>" << std::endl;
    input_tensor_values = std::vector<float>(1 * 1 * 28 * 28, 0.0f);  // default to all zeros
  } else {
    std::string        input_str(argv[1]);
    std::istringstream iss(input_str);
    float              val;
    while (iss >> val) {
      input_tensor_values.push_back(val);
      // Accept comma separated values too
      if (iss.peek() == ',') iss.ignore();
    }
    if (input_tensor_values.size() != 1 * 1 * 28 * 28) {
      std::cerr << "Input string must contain exactly " << (1 * 1 * 28 * 28) << " float values." << std::endl;
      return 1;
    }
  }

  Ort::AllocatorWithDefaultOptions allocator;

  const auto input_names = session.GetInputNames();
  // const char* input_name         = session.GetInputNameAllocated(0, allocator).get();
  const char* input_name        = input_names[0].c_str();
  const auto  input_type_info   = session.GetInputTypeInfo(0);
  const auto  input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
  const auto  output_names      = session.GetOutputNames();
  // const char* output_name        = session.GetOutputNameAllocated(0, allocator).get();
  const char* output_name        = output_names[0].c_str();
  const auto  output_type_info   = session.GetOutputTypeInfo(0);
  const auto  output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();

  std::cout << "Input Names: ";
  for (size_t i = 0; i < input_names.size(); ++i) {
    std::cout << "[" << i << "] " << input_names[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "Input Name: " << input_name << std::endl;
  std::cout << "Input Type: " << input_type_info.GetTensorTypeAndShapeInfo().GetElementType() << std::endl;
  std::cout << "Input Shape: " << input_tensor_info.GetShape().size() << "D" << std::endl;
  for (const auto& dim : input_tensor_info.GetShape()) {
    std::cout << dim << " ";
  }
  std::cout << std::endl;

  std::cout << "Output Names: ";
  for (size_t i = 0; i < output_names.size(); ++i) {
    std::cout << "[" << i << "] " << output_names[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "Output Name: " << output_name << std::endl;
  std::cout << "Output Type: " << output_type_info.GetTensorTypeAndShapeInfo().GetElementType() << std::endl;
  std::cout << "Output Shape: " << output_tensor_info.GetShape().size() << "D" << std::endl;
  for (const auto& dim : output_tensor_info.GetShape()) {
    std::cout << dim << " ";
  }
  std::cout << std::endl;

  // Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
  //     allocator, (void*)(input_tensor_values.data()), input_tensor_values.size(), input_shape.data(),
  //     input_shape.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator, input_shape.data(), input_shape.size());
  // fill the input tensor with values
  float* input_tensor_data = input_tensor.GetTensorMutableData<float>();
  for (size_t i = 0; i < input_tensor_values.size(); i++) {
    input_tensor_data[i] = input_tensor_values[i];
  }

  auto output_tensors = session.Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1);

  float* output_data = output_tensors.front().GetTensorMutableData<float>();
  // log all output values
  std::cout << "Output values: ";
  for (size_t i = 0; i < 10; i++) {
    std::cout << output_data[i] << " ";
  }
  std::cout << std::endl;

  // find the index of the max output value
  size_t max_index = 0;
  float  max_value = output_data[0];
  for (size_t i = 1; i < 10; i++) {
    if (output_data[i] > max_value) {
      max_value = output_data[i];
      max_index = i;
    }
  }
  std::cout << "Predicted class: " << max_index << std::endl;

  return 0;
}