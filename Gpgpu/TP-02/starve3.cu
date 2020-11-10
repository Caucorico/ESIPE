#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <time.h>
#include "cuStopwatch.cu"

#define SHIFT 13

// computing sliding square sum on columns of a matrix, window of size 128
__global__ void col_square_sum_1(const uint32_t* mat, uint32_t* mat_res, uint32_t w, uint32_t h) {
    uint32_t xid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t yid = threadIdx.y + blockIdx.y * blockDim.y;
	if((xid < h)&&(yid < w)){
        uint32_t size = w * h;
        uint32_t cur_idx = xid + yid * w;
        uint32_t sum = 0;
        for(uint32_t i = 0; i < 128; i++){
            sum += mat[cur_idx] * mat[cur_idx];
            cur_idx += w;
            if(cur_idx >= size) cur_idx -= size;
        }
        mat_res[xid + w * yid] = sum;
    }
	return;
}
// computing sliding square sum on columns of a matrix, window of size 128
// using shared memory
__global__ void col_square_sum_2(const uint32_t* mat, uint32_t* mat_res, uint32_t w, uint32_t h) {
    __shared__ uint32_t cache[128 * 2 * 32];
    uint32_t xid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t yid = threadIdx.y + blockIdx.y * blockDim.y;
	if((xid < h)&&(yid < w)){
        uint32_t size = w * h;
        uint32_t cur_idx = xid + yid * w;
        uint32_t cache_idx = threadIdx.y + threadIdx.x * 128 * 2;
        for(uint32_t i = 0; i < 8; i++){
            cache[cache_idx] += mat[cur_idx];
            cur_idx += w * 32;
            cache_idx += 32;
            if(cur_idx >= size) cur_idx -= size;
        }
        __syncthreads();
        uint32_t sum = 0;
        cache_idx = threadIdx.x + threadIdx.y * 128 * 2;
        for(uint32_t i = 0; i < 128; i++){
            sum += cache[cache_idx] * cache[cache_idx];
            cache_idx++;
        }
        mat_res[xid + w*yid] = sum;
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
    
	// Performing odd-even computing on 2^25 integers
    dim3 blocksize(32, 32);
    dim3 gridsize(mat_side / 32, mat_side / 32);
    printf("First method\n");
    cuStopwatch sw1;
    sw1.start();
	col_square_sum_1<<<gridsize, blocksize>>>(mat_dev, mat_res_dev, mat_side, mat_side);
    printf("%.4fms\n", sw1.stop());
    printf("\nSecond method\n");
    cuStopwatch sw2;
    sw2.start();
	col_square_sum_2<<<gridsize, blocksize, sizeof(uint32_t) * 128 * 2 * 32>>>(mat_dev, mat_res_dev, mat_side, mat_side);
    printf("%.4fms\n", sw2.stop());
    
    // Free memory
    cudaFree(mat_dev);
    cudaFree(mat_res_dev);
	return 0;
}