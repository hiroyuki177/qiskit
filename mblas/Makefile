# === Compiler and tools ===
CXX = clang++
METAL = xcrun -sdk macosx metal
METALLIB = xcrun -sdk macosx metallib
DUMPBIN = xxd

# === Target ===
TARGET = libComplexMatrix.dylib

# === Source and build ===
SRCS = $(shell find complex_matrix -name '*.mm') 
OBJS = $(SRCS:.mm=.o)

# === Metal shader ===
METAL_SRCS = $(wildcard kernels/*.metal)
AIR_FILES = $(METAL_SRCS:.metal=.air)
METALLIB_FILE = ComplexMatrix.metallib

# === Dumpbin settings ===
DUMP_FILE = complex_matrix/ComplexMatrixMetallib.h
DUMP_VAR_NAME = metallib_data
DUMP_ARGS = -n $(DUMP_VAR_NAME) -i

# === Compiler and linker flags ===
CXXFLAGS = -std=gnu++20 -ObjC++ -fvisibility=hidden -fobjc-arc -Wall -O2 -fPIC \
           -isysroot $(shell xcrun --sdk macosx --show-sdk-path) \
           -mmacosx-version-min=15.0 -I.

LDFLAGS = -dynamiclib \
          -install_name @loader_path/$(TARGET) \
          -framework Foundation \
          -framework Metal \
          -framework MetalPerformanceShaders \
          -framework MetalPerformanceShadersGraph

# === Build rules ===
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.mm $(DUMP_FILE)
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.air: %.metal
	$(METAL) -c $< -o $@ -I.

$(METALLIB_FILE): $(AIR_FILES)
	$(METALLIB) -o $@ $^
	rm -f $(AIR_FILES)

$(DUMP_FILE): $(METALLIB_FILE)
	$(DUMPBIN) $(DUMP_ARGS) $< > $@

clean:
	rm -f $(OBJS) $(TARGET) $(METALLIB_FILE) $(DUMP_FILE)
