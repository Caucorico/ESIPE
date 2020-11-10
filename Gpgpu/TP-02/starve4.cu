#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <time.h>
#include "cuStopwatch.cu"

#define SHIFT 13

__global__ void transpose_1(const uint32_t* mat, uint32_t* mat_trans, uint32_t w, uint32_t h) {
    uint32_t xid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t yid = threadIdx.y + blockIdx.y * blockDim.y;
	if((xid < h)&&(yid < w)){
        mat_trans[yid + w * xid] = mat[xid + w * yid];
    }
	return;
}

__global__ void transpose_2(const uint32_t* mat, uint32_t* mat_trans, uint32_t w, uint32_t h) {
    uint32_t xid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t yid = threadIdx.y + blockIdx.y * blockDim.y;
	if((xid < h)&&(yid < w)){
        mat_trans[xid + h * yid] = mat[yid + h * xid];
    }
	return;
}

void randgen(uint32_t* arr, size_t count){
    uint32_t state = time(NULL);
    for(uint32_t i = 0; i < count; i++){
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        arr[i] = state;
    }
    return;
}

int main() {
    // Allocate memory, filling in random data and transfer to device
    uint32_t *mat_host, *mat_dev, *mat_res_dev;
    const uint32_t mat_size = 1 << (SHIFT * 2);
    const uint32_t mat_side = 1 << SHIFT;
    cudaHostAlloc((void**)&mat_host, mat_size*sizeof(uint32_t), cudaHostAllocDefault);
    cudaMalloc((void**)&mat_dev, mat_size*sizeof(uint32_t));
    cudaMalloc((void**)&mat_res_dev, mat_size*sizeof(uint32_t));
    printf("Copying data to device\n");
    randgen(mat_host, mat_size);
    cudaMemcpy(mat_dev, mat_host, mat_size*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaFreeHost(mat_host);
    
	// Performing matrix transposition on a 2^13 * 2^13 matrix
    dim3 blocksize(32, 32);
    dim3 gridsize(mat_side / 32, mat_side / 32);
    printf("First method\n");
    cuStopwatch sw1;
    sw1.start();
	transpose_1<<<gridsize, blocksize>>>(mat_dev, mat_res_dev, mat_side, mat_side);
    printf("%.4fms\n", sw1.stop());
    printf("\nSecond method\n");
    cuStopwatch sw2;
    sw2.start();
	transpose_2<<<gridsize, blocksize>>>(mat_dev, mat_res_dev, mat_side, mat_side);
    printf("%.4fms\n", sw2.stop());
    
    // Free memory
    cudaFree(mat_dev);
    cudaFree(mat_res_dev);
	return 0;
}