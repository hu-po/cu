#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h> // For rand() and srand()
#include <time.h>   // For srand() seed

// --- CUDA Error Checking Macro ---
// Very important for debugging CUDA code!
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// --- CUDA Kernel for AllReduce (Sum) ---
__global__ void sumReduceKernel(const float* d_input, float* d_outputSum, int n) {
    // --- 1. Shared Memory Allocation ---
    // Allocate shared memory for this block. Size should match blockDim.x
    extern __shared__ float sdata[]; // Dynamically sized shared memory

    // --- 2. Index Calculation ---
    unsigned int tid = threadIdx.x; // Thread ID within the block
    unsigned int bid = blockIdx.x;  // Block ID within the grid
    unsigned int gid = blockDim.x * bid + tid; // Global thread ID

    // --- 3. Load Data into Shared Memory ---
    // Each thread loads one element from global memory into shared memory
    // Check bounds to avoid reading past the end of the input array
    if (gid < n) {
        sdata[tid] = d_input[gid];
    } else {
        sdata[tid] = 0.0f; // Neutral element for summation
    }

    // Synchronize within the block to ensure all data is loaded into sdata
    __syncthreads();

    // --- 4. Intra-Block Reduction ---
    // Perform reduction in shared memory. This uses a common parallel reduction pattern.
    // We divide the number of active threads by 2 in each step.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        // Only the first 's' threads in the block participate in the addition
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        // Synchronize within the block to ensure additions are complete before the next step
        __syncthreads();
    }

    // --- 5. Write Block Result to Global Memory (Atomic) ---
    // Only the first thread of each block (tid == 0) writes the block's partial sum
    // to the global output location using an atomic operation.
    // atomicAdd ensures that additions from different blocks don't conflict.
    if (tid == 0) {
        atomicAdd(d_outputSum, sdata[0]);
    }
}

// --- Host Code (main function) ---
int main() {
    // --- Configuration ---
    int dataSize = 1024 * 1024; // Example data size (1 million floats)
    int blockSize = 256;       // Threads per block (power of 2 is often good)

    // Calculate grid size based on data size and block size
    int gridSize = (dataSize + blockSize - 1) / blockSize;

    printf("Data Size: %d elements\n", dataSize);
    printf("Block Size: %d threads\n", blockSize);
    printf("Grid Size: %d blocks\n", gridSize);

    // --- Memory Allocation ---
    // Host memory
    float* h_input = (float*)malloc(dataSize * sizeof(float));
    if (h_input == NULL) {
        fprintf(stderr, "Failed to allocate host input memory\n");
        return EXIT_FAILURE;
    }
    float h_outputSum = 0.0f; // Host variable for final result

    // Device memory
    float* d_input = NULL;
    float* d_outputSum = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_input, dataSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_outputSum, sizeof(float))); // Allocate space for ONE float result

    // --- Initialize Data ---
    srand(time(NULL)); // Seed random number generator
    float hostSumCheck = 0.0f;
    printf("Initializing host data...\n");
    for (int i = 0; i < dataSize; ++i) {
        h_input[i] = (rand() % 100) / 10.0f; // Simple random floats between 0.0 and 9.9
        hostSumCheck += h_input[i];
    }
    printf("Host-calculated sum (for verification): %f\n", hostSumCheck);

    // Initialize device output sum to 0
    CUDA_CHECK(cudaMemset(d_outputSum, 0, sizeof(float)));

    // --- Copy Data Host -> Device ---
    printf("Copying data from Host to Device...\n");
    CUDA_CHECK(cudaMemcpy(d_input, h_input, dataSize * sizeof(float), cudaMemcpyHostToDevice));

    // --- Kernel Launch ---
    // Calculate required dynamic shared memory size: blockSize floats
    size_t sharedMemBytes = blockSize * sizeof(float);

    printf("Launching CUDA kernel...\n");
    // Launch the kernel: grid, block, shared memory size, stream (0=default)
    sumReduceKernel<<<gridSize, blockSize, sharedMemBytes>>>(d_input, d_outputSum, dataSize);

    // Check for kernel launch errors (important!)
    CUDA_CHECK(cudaGetLastError());

    // --- Synchronize Device ---
    // Wait for the kernel to complete before copying the result back
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Kernel execution finished.\n");

    // --- Copy Result Device -> Host ---
    printf("Copying result from Device to Host...\n");
    CUDA_CHECK(cudaMemcpy(&h_outputSum, d_outputSum, sizeof(float), cudaMemcpyDeviceToHost));

    // --- Verification ---
    printf("\n--- Results ---\n");
    printf("GPU Calculated Sum (AllReduce Result): %f\n", h_outputSum);
    printf("CPU Calculated Sum (Reference)      : %f\n", hostSumCheck);

    // Compare results (account for potential floating-point inaccuracies)
    float tolerance = 1e-4 * dataSize; // Tolerance depends on data size and values
    if (fabs(h_outputSum - hostSumCheck) < tolerance) {
        printf("Result VERIFIED!\n");
    } else {
        printf("Result MISMATCH! Difference: %f\n", fabs(h_outputSum - hostSumCheck));
    }

    // --- Cleanup ---
    printf("Cleaning up memory...\n");
    free(h_input);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_outputSum));

    return EXIT_SUCCESS;
}