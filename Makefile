CXX=nvc++
CXXFLAGS=-fast -std=c++20 --diag_suppress=3560

# Architecture-specific flags
GPUFLAGS_H100=-gpu=cc90
GPUFLAGS_A100=-gpu=cc80
GPUFLAGS_GH=-gpu=cc90

# Build directories
BUILD_H100=build_h100
BUILD_A100=build_a100
BUILD_GH=build_gh

# Common sources
SRCS=BlackScholes_reference.cpp BlackScholes_main.cpp pnl.cpp pnl_cuda.cu pnl_omp.cpp pnl_comparison.cu
HEADERS=BSM.hpp greek.hpp pnl_cuda.cuh pnl.hpp BlackScholes_stdpar.hpp

# Standard flags
STDPARFLAGS_OMP=-mp=gpu
STDPARFLAGS_GPU=-acc -cuda -mp=gpu -stdpar=gpu
STDPARFLAGS_CPU=-acc -mp -stdpar=multicore

.PHONY: all h100 a100 gh clean

all: h100 a100 gh

h100:
	$(MAKE) BUILD=$(BUILD_H100) GPUFLAGS="$(GPUFLAGS_H100)" binaries

a100:
	$(MAKE) BUILD=$(BUILD_A100) GPUFLAGS="$(GPUFLAGS_A100)" binaries

gh:
	$(MAKE) BUILD=$(BUILD_GH) GPUFLAGS="$(GPUFLAGS_GH)" binaries

# Pattern rules for object files in build dir
$(BUILD_H100)/%.o $(BUILD_A100)/%.o $(BUILD_GH)/%.o: %.cpp $(HEADERS)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -I. $(STDPARFLAGS_GPU) -c $< -o $@ $(GPUFLAGS)

$(BUILD_H100)/%.o $(BUILD_A100)/%.o $(BUILD_GH)/%.o: %.cu $(HEADERS)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -I. $(STDPARFLAGS_GPU) -c $< -o $@ $(GPUFLAGS)

# Object files for each build
OBJS=$(SRCS:.cpp=.o)
OBJS:=$(OBJS:.cu=.o)
OBJS_BUILD=$(addprefix $(BUILD)/, $(OBJS))

# Binaries for each architecture
binaries: \
	$(BUILD)/pnl_gpu \
	$(BUILD)/pnl_cpu \
	$(BUILD)/pnl_cuda \
	$(BUILD)/pnl_comparison

# binaries: \
	$(BUILD)/BlackScholes_gpu \
	$(BUILD)/BlackScholes_cpu \
	$(BUILD)/pnl_gpu \
	$(BUILD)/pnl_cpu \
	$(BUILD)/pnl_cuda \
	$(BUILD)/pnl_comparison

# $(BUILD)/BlackScholes_gpu: $(BUILD)/BlackScholes_reference.o $(BUILD)/BlackScholes_main_gpu.o
# 	$(CXX) $(CXXFLAGS) -o $@ $^ $(STDPARFLAGS_GPU) $(GPUFLAGS)

# $(BUILD)/BlackScholes_cpu: $(BUILD)/BlackScholes_reference.o $(BUILD)/BlackScholes_main_cpu.o
# 	$(CXX) $(CXXFLAGS) -o $@ $^ $(STDPARFLAGS_CPU)

# $(BUILD)/BlackScholes_main_gpu.o: BlackScholes_main.cpp BlackScholes_stdpar.hpp
# 	$(CXX) $(CXXFLAGS) -I. $(STDPARFLAGS_GPU) -c -o $@ $< $(GPUFLAGS)

# $(BUILD)/BlackScholes_main_cpu.o: BlackScholes_main.cpp BlackScholes_stdpar.hpp
# 	$(CXX) $(CXXFLAGS) -I. $(STDPARFLAGS_CPU) -c -o $@ $<

$(BUILD)/pnl_gpu.o: pnl.cpp BSM.hpp greek.hpp
	$(CXX) $(CXXFLAGS) -I. $(STDPARFLAGS_GPU) -DMAIN_FILE -c -o $@ $< $(GPUFLAGS)

$(BUILD)/pnl_gpu: $(BUILD)/pnl_gpu.o
	$(CXX) $(CXXFLAGS) -I. $(STDPARFLAGS_GPU) -o $@ $< $(GPUFLAGS)

$(BUILD)/pnl_cpu.o: pnl.cpp BSM.hpp greek.hpp
	$(CXX) $(CXXFLAGS) -I. $(STDPARFLAGS_CPU) -DMAIN_FILE -c -o $@ $<

$(BUILD)/pnl_cpu: $(BUILD)/pnl_cpu.o
	$(CXX) $(CXXFLAGS) -I. $(STDPARFLAGS_CPU) -o $@ $<

$(BUILD)/BlackScholes_reference.o: BlackScholes_reference.cpp BSM.hpp greek.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD)/pnl_cuda.o: pnl_cuda.cu BSM.hpp greek.hpp
	$(CXX) $(CXXFLAGS) -I. $(STDPARFLAGS_GPU) -DMAIN_FILE -c -o $@ $< $(GPUFLAGS)

$(BUILD)/pnl_cuda: $(BUILD)/pnl_cuda.o
	$(CXX) $(CXXFLAGS) -I. $(STDPARFLAGS_GPU) -o $@ $< $(GPUFLAGS)

$(BUILD)/pnl_omp_comparison.o: pnl_omp.cpp BSM.hpp greek.hpp
	$(CXX) $(CXXFLAGS) -I. $(STDPARFLAGS_OMP) -c -o $@ $< -Minfo=mp,accel

$(BUILD)/pnl_cuda_comparison.o: pnl_cuda.cu BSM.hpp greek.hpp
	$(CXX) $(CXXFLAGS) -I. $(STDPARFLAGS_GPU) -c -o $@ $< $(GPUFLAGS)

$(BUILD)/pnl_gpu_comparison.o: pnl.cpp BSM.hpp greek.hpp
	$(CXX) $(CXXFLAGS) -I. $(STDPARFLAGS_GPU) -c -o $@ $< $(GPUFLAGS)

$(BUILD)/pnl_comparison.o: pnl_comparison.cu BSM.hpp greek.hpp pnl_cuda.cuh pnl.hpp
	$(CXX) $(CXXFLAGS) -I. $(STDPARFLAGS_GPU) -DMAIN_FILE -DNVTX_ENABLE=1 -c -o $@ $< $(GPUFLAGS)

$(BUILD)/pnl_comparison: $(BUILD)/pnl_comparison.o $(BUILD)/pnl_cuda_comparison.o $(BUILD)/pnl_gpu_comparison.o $(BUILD)/pnl_omp_comparison.o
	$(CXX) $(CXXFLAGS) -I. $(STDPARFLAGS_GPU) -o $@ $^ $(GPUFLAGS)

clean:
	rm -rf $(BUILD_H100) $(BUILD_A100) $(BUILD_GH)

cleangh:
	rm -f $(BUILD_GH)/*.o $(BUILD_GH)/pnl_comparison

cleanh100:
	rm -f $(BUILD_H100)/*.o $(BUILD_H100)/pnl_comparison

cleana100:
	rm -f $(BUILD_A100)/*.o $(BUILD_A100)/pnl_comparison 