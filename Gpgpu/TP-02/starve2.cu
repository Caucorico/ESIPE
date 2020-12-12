#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <time.h>
#include "cuStopwatch.cu"

#define SHIFT 27

__global__ void compute_parity_1(const uint32_t* arr, uint32_t* arr_res, uint32_t size) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < size){
        if((tid % 2) == 0){
            uint32_t popcnt = 0;
            uint32_t n = arr[tid];
            while(n != 0){
                popcnt += n%2;
                n /= 2;
            }
            arr_res[tid] = popcnt;
        }else{
            uint32_t step = 0;
            uint64_t n = arr[tid];
            while(n != 1){
                step++;
                if(n%2) n = 3*n + 1; else n /= 2;
            }
            arr_res[tid] = step;
        }
    }
	return;
}

__global__ void compute_parity_2(const uint32_t* arr, uint32_t* arr_res, uint32_t size) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < size){
        if(tid * 2 < size){
            tid *= 2;
            uint32_t popcnt = 0;
            uint32_t n = arr[tid];
            while(n != 0){
                popcnt += n%2;
                n /= 2;
            }
            arr_res[tid] = popcnt;
        }else{
            tid = tid * 2 - size + 1;
            uint32_t step = 0;
            uint64_t n = arr[tid];
            while(n != 1){
                step++;
                if(n%2) n = 3*n + 1; else n /= 2;
            }
            arr_res[tid] = step;
        }
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
    uint32_t *arr_host, *arr_dev, *arr_res_dev;
    const uint32_t arr_size = 1 << SHIFT;
    cudaHostAlloc((void**)&arr_host, arr_size*sizeof(uint32_t), cudaHostAllocDefault);
    cudaMalloc((void**)&arr_dev, arr_size*sizeof(uint32_t));
    cudaMalloc((void**)&arr_res_dev, arr_size*sizeof(uint32_t));
    printf("Copying data to device\n");
    randgen(arr_host, arr_size);
    cudaMemcpy(arr_dev, arr_host, arr_size*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaFreeHost(arr_host);
    
	// Performing odd-even computing on 2^25 integers
    printf("First method\n");
    cuStopwatch sw1;
    sw1.start();
	compute_parity_1<<<(1<<SHIFT-10), 1024>>>(arr_dev, arr_res_dev, arr_size);
    printf("%.4fms\n", sw1.stop());
    printf("\nSecond method\n");
    cuStopwatch sw2;
    sw2.start();
	compute_parity_2<<<(1<<SHIFT-10), 1024>>>(arr_dev, arr_res_dev, arr_size);
    printf("%.4fms\n", sw2.stop());
    
    // Free memory
    cudaFree(arr_dev);
    cudaFree(arr_res_dev);
	return 0;
}