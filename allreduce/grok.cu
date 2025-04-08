#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Warp-level AllReduce kernel using shuffle operations
__device__ float warpAllReduce(float val, thread_block_tile<32> tile) {
    for (int offset = tile.size() / 2; offset > 0; offset /= 2) {
        val += tile.shfl_down(val, offset);
    }
    // Broadcast result from lane 0 to all lanes
    return tile.shfl(val, 0);
}

// Block-level AllReduce kernel using warp-level primitives
__device__ float blockAllReduce(float val, thread_block block) {
    __shared__ float shared[32]; // One element per warp
    
    // Create a tile for the current warp
    auto tile = tiled_partition<32>(block);
    int warpId = block.thread_rank() / warpSize;
    int numWarps = block.size() / warpSize;
    
    // First do warp-level reduction
    float warpResult = warpAllReduce(val, tile);
    
    // Write warp results to shared memory
    if (tile.thread_rank() == 0) {
        shared[warpId] = warpResult;
    }
    block.sync();
    
    // First thread of first warp reduces warp results
    if (block.thread_rank() < numWarps) {
        float warpVal = shared[block.thread_rank()];
        auto firstWarpTile = tiled_partition<32>(block);
        warpVal = warpAllReduce(warpVal, firstWarpTile);
        if (block.thread_rank() == 0) {
            shared[0] = warpVal;
        }
    }
    block.sync();
    
    // Broadcast final result to all threads
    return shared[0];
}

// Global AllReduce kernel
__global__ void allReduceKernel(const float* input, float* output, int size) {
    auto block = this_thread_block();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    // Each thread loads and sums multiple elements
    for (int i = tid; i < size; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    
    // Perform block-level reduction
    sum = blockAllReduce(sum, block);
    
    // First thread in each block adds to global sum
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

int main() {
    const int size = 1024;        // Size of the array
    const int blockSize = 256;    // Threads per block
    const int numBlocks = (size + blockSize - 1) / blockSize;
    
    // Allocate and initialize host data
    float* hostInput = new float[size];
    float hostOutput = 0.0f;
    float cpuSum = 0.0f;
    
    printf("Initializing data...\n");
    for (int i = 0; i < size; i++) {
        hostInput[i] = static_cast<float>(i);
        cpuSum += hostInput[i];
    }
    
    // Allocate device memory
    float *deviceInput, *deviceOutput;
    CUDA_CHECK(cudaMalloc(&deviceInput, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&deviceOutput, sizeof(float)));
    
    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(deviceInput, hostInput, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(deviceOutput, 0, sizeof(float)));
    
    printf("Performing AllReduce...\n");
    allReduceKernel<<<numBlocks, blockSize>>>(deviceInput, deviceOutput, size);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(&hostOutput, deviceOutput, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify results
    printf("\nResults:\n");
    printf("GPU AllReduce sum: %.0f\n", hostOutput);
    printf("CPU Reference sum: %.0f\n", cpuSum);
    printf("Difference: %.6f\n", fabs(hostOutput - cpuSum));
    
    // Cleanup
    delete[] hostInput;
    CUDA_CHECK(cudaFree(deviceInput));
    CUDA_CHECK(cudaFree(deviceOutput));
    
    return 0;
}