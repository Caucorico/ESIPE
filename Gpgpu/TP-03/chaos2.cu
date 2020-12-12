#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "cuStopwatch.cu"

// Building lookup table for square of numbers in a stupid way
__global__ void build_lookup() {
    __shared__ uint32_t lookup[1024];
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	// Build the table
    if(tid < 1024){
        lookup[tid] = 0;
        for(uint32_t i = 0; i < tid * tid; i++) lookup[tid] += 1;

	/* If we didn't sync here, some threads will begin the 0 check before the end. And this is not that what we want. */
	__syncthreads();

        // Check the table, there can be no zero entries
        if(tid < 32){
            for(uint32_t i = 0; i < 1024; i+=32)
                if(lookup[tid+i]!=(tid+i)*(tid+i)) printf("Error on entry %u!\n", tid+i);
        }
    }
	return;
}

int main() {
	// Perform computation
    cuStopwatch sw1;
    sw1.start();
	build_lookup<<<1, 1024>>>();
    printf("Computation time: %.4fms\n", sw1.stop());
    //printf("%s", cudaGetErrorString(cudaPeekAtLastError()));
	return 0;
}
