#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <vector>

int main() {
  Ort::Env            env(ORT_LOGGING_LEVEL_WARNING, "test");
  Ort::SessionOptions session_options;

  Ort::Session session(env, "../onnx/cnn.onnx", session_options);

  // input vector
  std::vector<float> input_tensor_values(1 * 1 * 28 * 28, 0.0f);
  // input shape
  std::vector<int64_t> input_shape = {1, 1, 28, 28};

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

  return 0;
}