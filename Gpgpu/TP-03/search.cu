#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <time.h>
#include "cuStopwatch.cu"

#define SHIFT 27

__global__ void search_1(const uint32_t* arr, uint32_t size, uint32_t* res, uint32_t val) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t gridsize = blockDim.x * gridDim.x;
    uint32_t warpcnt = 0;
	while(tid < size){
        uint32_t herecount = (arr[tid] == val) ? 1 : 0;
        herecount += __shfl_down_sync(0xffffffff, herecount, 16);
        herecount += __shfl_down_sync(0xffffffff, herecount, 8);
        herecount += __shfl_down_sync(0xffffffff, herecount, 4);
        herecount += __shfl_down_sync(0xffffffff, herecount, 2);
        herecount += __shfl_down_sync(0xffffffff, herecount, 1);
        warpcnt += herecount;
        tid += gridsize;
    }
    if((threadIdx.x & 31) == 0) atomicAdd(res, warpcnt);
	return;
}

__global__ void search_2(const uint32_t* arr, uint32_t size, uint32_t* res, uint32_t val) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t gridsize = blockDim.x * gridDim.x;
	while(tid < size){
        uint32_t ishere = (arr[tid] == val) ? 1 : 0;
        if(__any_sync(0xffffffff, ishere)) 
            if ((threadIdx.x & 31) == 0) atomicAdd(res, 1);
        tid += gridsize;
    }
	return;
}

__global__ void search_3(const uint32_t* arr, uint32_t size, uint32_t* res, uint32_t val) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t gridsize = blockDim.x * gridDim.x;
	while(tid < size){
        if(arr[tid] == val) atomicAdd(res, 1);
        tid += gridsize;
    }
	return;
}

/* In the Algo number 4, each thread will check some index of the array.
   The thread number 10 will check the indexes (i*gridsize) + 10.
   i take the value in {0..(size(res)/gridsize)}
 */
__global__ void search_4(const uint32_t* arr, uint32_t size, uint32_t* res, uint32_t val) {
    if(*res != 0) return;
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t gridsize = blockDim.x * gridDim.x;
	while((tid < size) && (*res == 0)){
        if(arr[tid] == val) (*res)++;
        tid += gridsize;
    }
	return;
}

void randgen(uint32_t* arr, size_t count, uint32_t mask){
    uint32_t state = time(NULL);
    for(uint32_t i = 0; i < count; i++){
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        arr[i] = state & mask;
    }
    return;
}

int main() {
    // Allocate memory, filling in random data and transfer to device
    uint32_t *arr_host, *arr_dev, *res_dev;
    uint32_t res;
    const uint32_t arr_size = 1 << SHIFT;
    cudaHostAlloc((void**)&arr_host, arr_size*sizeof(uint32_t), cudaHostAllocDefault);
    cudaMalloc((void**)&arr_dev, arr_size*sizeof(uint32_t));
    cudaMalloc((void**)&res_dev, sizeof(uint32_t));
    printf("Finding 42 in %d elements\n", arr_size);

    // Search the element 42 using different kernels
    for(int target_shift = 12; target_shift <= 32; target_shift+=4){
        randgen(arr_host, arr_size, (1<<target_shift) - 1);
        uint32_t exactcnt = 0;
        float elapsed = 0;
        for(int i=0; i<arr_size; i++)
            if(arr_host[i] == 42) exactcnt++;
        printf("\nShift %d, with %d elements equal to 42 to be found\n", target_shift, exactcnt);
        cudaMemcpyAsync(arr_dev, arr_host, arr_size*sizeof(uint32_t), cudaMemcpyHostToDevice);
        // Performing odd-even computing on 2^25 integers
        cuStopwatch sw1;
        sw1.start();
        res = 0;
        cudaMemcpyAsync(res_dev, &res, sizeof(uint32_t), cudaMemcpyHostToDevice);
        search_1<<<256, 1024>>>(arr_dev, arr_size, res_dev, 42);
        cudaMemcpyAsync(&res, res_dev, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        elapsed = sw1.stop();
        if(res != 0)
            printf("Method 1: %7.4fms, found, returning %u.\n", elapsed, res);
        else
            printf("Method 1: %7.4fms, not found, returning %u.\n", elapsed, res);
        cuStopwatch sw2;
        sw2.start();
        res = 0;
        cudaMemcpyAsync(res_dev, &res, sizeof(uint32_t), cudaMemcpyHostToDevice);
        search_2<<<256, 1024>>>(arr_dev, arr_size, res_dev, 42);
        cudaMemcpyAsync(&res, res_dev, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        elapsed = sw2.stop();
        if(res != 0)
            printf("Method 2: %7.4fms, found, returning %u.\n", elapsed, res);
        else
            printf("Method 2: %7.4fms, not found, returning %u.\n", elapsed, res);        
        cuStopwatch sw3;
        sw3.start();
        res = 0;
        cudaMemcpyAsync(res_dev, &res, sizeof(uint32_t), cudaMemcpyHostToDevice);
        search_3<<<256, 1024>>>(arr_dev, arr_size, res_dev, 42);
        cudaMemcpyAsync(&res, res_dev, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        elapsed = sw3.stop();
        if(res != 0)
            printf("Method 3: %7.4fms, found, returning %u.\n", elapsed, res);
        else
            printf("Method 3: %7.4fms, not found, returning %u.\n", elapsed, res);
        cuStopwatch sw4;
        sw4.start();
        res = 0;
        cudaMemcpyAsync(res_dev, &res, sizeof(uint32_t), cudaMemcpyHostToDevice);
        search_4<<<256, 1024>>>(arr_dev, arr_size, res_dev, 42);
        cudaMemcpyAsync(&res, res_dev, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        elapsed = sw4.stop();
        if(res != 0)
            printf("Method 4: %7.4fms, found, returning %u.\n", elapsed, res);
        else
            printf("Method 4: %7.4fms, not found, returning %u.\n", elapsed, res);    
    }

    // Free memory
    cudaFreeHost(arr_host);
    cudaFree(arr_dev);
    cudaFree(res_dev);
	return 0;
}
