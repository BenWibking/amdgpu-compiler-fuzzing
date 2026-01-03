CXX ?= c++
PKG_CONFIG ?= pkg-config

CXXFLAGS ?= -O2 -g -std=c++17

VK_CFLAGS := $(shell $(PKG_CONFIG) --cflags vulkan 2>/dev/null)
VK_LIBS := $(shell $(PKG_CONFIG) --libs vulkan 2>/dev/null)

TARGET := vk_aco_compile
SRCS := main.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(VK_CFLAGS) -o $@ $(SRCS) $(VK_LIBS)

clean:
	rm -f $(TARGET)
