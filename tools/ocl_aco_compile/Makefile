CXX ?= c++
PKG_CONFIG ?= pkg-config

CXXFLAGS ?= -O2 -g -std=c++17 -DCL_TARGET_OPENCL_VERSION=120

OPENCL_CFLAGS := $(shell $(PKG_CONFIG) --cflags OpenCL 2>/dev/null)
OPENCL_LIBS := $(shell $(PKG_CONFIG) --libs OpenCL 2>/dev/null)

TARGET := ocl_aco_compile
SRCS := main.cpp

ifeq ($(strip $(OPENCL_LIBS)),)
OPENCL_LIBS := -lOpenCL
endif

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(OPENCL_CFLAGS) -o $@ $(SRCS) $(OPENCL_LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
