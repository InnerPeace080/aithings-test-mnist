SRC = onnx_infer.cpp
TARGET = onnx_infer
ONNX_RUNTIME_DIR = ./onnxruntime-osx-arm64-1.23.1
COMPILER = g++

INCLUDE = -I$(ONNX_RUNTIME_DIR)/include
LIB = -L$(ONNX_RUNTIME_DIR)/lib -lonnxruntime

# check platform and set flags accordingly
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
		RPATH = -Wl,-rpath,$(ONNX_RUNTIME_DIR)/lib
else ifeq ($(UNAME_S),Darwin)		
		RPATH = -Wl,-rpath,$(ONNX_RUNTIME_DIR)/lib
endif

all: $(TARGET)

$(TARGET): $(SRC)
	$(COMPILER) $^ -o $@ $(INCLUDE) $(LIB) -std=c++11 $(RPATH)

clean:
	rm -f $(TARGET)

rebuild: clean all