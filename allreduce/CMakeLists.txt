cmake_minimum_required(VERSION 3.18)
project(allreduce_examples CUDA CXX)

# Set CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75)  # Default to Turing (RTX 20xx) architecture

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA optimization and cooperative groups
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-relaxed-constexpr")

# Add CUDA source files
add_executable(gemini gemini.cu)
add_executable(grok grok.cu)
add_executable(gpt gpt.cu)

# Set output directory for executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin) 