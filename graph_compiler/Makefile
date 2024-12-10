ROSE_HOME = $(shell rose-config prefix)
include $(ROSE_HOME)/lib/rose-config.cfg

ROSE_CXX         = $(shell $(ROSE_HOME)/bin/rose-config ROSE_CXX)
ROSE_CPPFLAGS    = $(shell $(ROSE_HOME)/bin/rose-config ROSE_CPPFLAGS)
ROSE_CXXFLAGS    = $(shell $(ROSE_HOME)/bin/rose-config ROSE_CXXFLAGS)
ROSE_LDFLAGS     = $(shell $(ROSE_HOME)/bin/rose-config ROSE_LDFLAGS)
ROSE_LIBDIRS     = $(shell $(ROSE_HOME)/bin/rose-config ROSE_LIBDIRS)
ROSE_RPATHS      = $(shell $(ROSE_HOME)/bin/rose-config ROSE_RPATHS)
ROSE_LINK_RPATHS = $(shell $(ROSE_HOME)/bin/rose-config ROSE_LINK_RPATHS)

# Directories
SUB_DIRS := graph
SRC_DIR := src
SRC_DIRS := $(SRC_DIR) $(addprefix $(SRC_DIR)/,$(SUB_DIRS))
BUILD_DIR := build
BIN_DIR := bin
DEPDIR := .deps

# Project Name
PROJECT_NAME := $(notdir $(CURDIR))

# Source files
SRCS := $(wildcard $(addsuffix /*.cpp,$(SRC_DIRS)))
# Object files
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
# Executable name
EXECUTABLE := $(BIN_DIR)/$(PROJECT_NAME)

DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d

# Main target
all: $(EXECUTABLE)

# Ensure that subdirectories exist
$(shell mkdir -p $(BUILD_DIR))
$(shell mkdir -p $(addprefix $(BUILD_DIR)/,$(SUB_DIRS)))
$(shell mkdir -p $(DEPDIR))
$(shell mkdir -p $(addprefix $(DEPDIR)/,$(SUB_DIRS)))

$(shell mkdir -p $(BIN_DIR))

# Rule for linking object files and creating executable
$(EXECUTABLE): $(OBJS)
	$(ROSE_CXX) $(ROSE_CXXFLAGS) -o $@ $^ $(ROSE_LDFLAGS) $(ROSE_LINK_RPATHS) -Wl,-rpath=$(ROSE_HOME)/lib

# Rule for compiling individual source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR) $(DEPDIR)
	$(ROSE_CXX) $(ROSE_CPPFLAGS) $(ROSE_CXXFLAGS) $(DEPFLAGS) -c $< -o $@

# Clean rule to remove generated files
clean:
	rm -rf $(BUILD_DIR)/* $(BIN_DIR)/* $(DEPDIR)/*

clang-tidy:
	clang-tidy $(SRCS) -- $(ROSE_CPPFLAGS) 

.PHONY: all clean clang-tidy

DEPFILES := $(patsubst $(SRC_DIR)/%.cpp,$(DEPDIR)/%.d,$(SRCS))
$(DEPFILES):


include $(wildcard $(DEPFILES))