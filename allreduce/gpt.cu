#include <cstdio>
#include <cuda_runtime.h>

// Kernel that performs a block-wide AllReduce using shared memory.
__global__ void allreduce_kernel(float* data, int count) {
    // Allocate shared memory dynamically.
    extern __shared__ float shared_data[];

    // Each thread loads one element from global memory into shared memory.
    int tid = threadIdx.x;
    if (tid < count) {
        shared_data[tid] = data[tid];
    } else {
        shared_data[tid] = 0.0f; // safety; this branch won’t be reached if count == blockDim.x.
    }
    __syncthreads();

    // Perform reduction in shared memory.
    // Here we assume count is a power of two.
    for (int stride = count / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    // After the loop, thread 0 holds the sum.
    float blockSum = shared_data[0];

    // Broadcast the block sum to every element in global memory.
    // (Each thread writes the same final sum)
    if (tid < count) {
        data[tid] = blockSum;
    }
}

int main() {
    // Number of elements equals the number of threads in one block.
    const int count = 256;
    const size_t bytes = count * sizeof(float);

    // Host allocation and initialization.
    float h_data[count];
    // For example, initialize each element with 1.0 so the sum will be count (i.e. 256).
    for (int i = 0; i < count; i++) {
        h_data[i] = 1.0f;
    }

    // Allocate device memory.
    float *d_data = nullptr;
    cudaMalloc(&d_data, bytes);
    
    // Copy data to device.
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    // Launch kernel with one block and 'count' threads.
    // We allocate shared memory size equal to count * sizeof(float).
    allreduce_kernel<<<1, count, bytes>>>(d_data, count);
    
    // Wait for the kernel to finish.
    cudaDeviceSynchronize();

    // Copy the result back to the host.
    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

    // Display the result: each element should now contain the sum (which is 256 in this example).
    printf("After AllReduce, each element is: %f\n", h_data[0]);
    
    // Optionally, print all elements:
    // for (int i = 0; i < count; i++) {
    //     printf("%f ", h_data[i]);
    // }
    // printf("\n");

    // Free device memory.
    cudaFree(d_data);

    return 0;
}
