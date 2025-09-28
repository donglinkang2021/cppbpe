PYTHON := python3
CXX := c++
CXXFLAGS := -O3 -Wall -shared -std=c++17 -fPIC

# 自动获取 python 和 pybind11 的 include 路径
PYINCLUDE := $(shell $(PYTHON) -m pybind11 --includes)
PYEXT := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

TARGET := bpe_core$(PYEXT)

all: $(TARGET)

$(TARGET): bpe_core.cpp
	$(CXX) $(CXXFLAGS) $(PYINCLUDE) $< -o $@

clean:
	rm -f $(TARGET) *.o
	rm -rf build
