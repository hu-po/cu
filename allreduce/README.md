# CUDA AllReduce Examples

This project contains example implementations of AllReduce operations in CUDA, demonstrating different approaches to parallel reduction.

## Prerequisites

- CUDA Toolkit (version 11.0 or later)
- CMake (version 3.18 or later)
- A CUDA-capable GPU

## Building the Project

1. Create a build directory:
```bash
mkdir build
cd build
```

2. Configure the project:
```bash
cmake ..
```

3. Build the project:
```bash
make
```

The executables will be created in the `build/bin` directory.

## Available Examples

- `gemini`: Implementation of AllReduce using shared memory
- `grok`: Alternative implementation with different optimization strategies
- `gpt`: Another variation of AllReduce implementation

## Running the Examples

After building, you can run any of the examples from the `build/bin` directory:

```bash
./bin/gemini
./bin/grok
./bin/gpt
```

## Notes

- The default CUDA architecture is set to 75 (Turing/RTX 20xx). If you have a different GPU, you may need to adjust the `CMAKE_CUDA_ARCHITECTURES` value in `CMakeLists.txt`.
- The examples are compiled with optimization level 3 (-O3) for best performance. 