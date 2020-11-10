#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "cuStopwatch.cu"

// computing sum-product numbers
__global__ void kernel(uint32_t n) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t sum = 0;
    uint32_t prod = 1;
    uint32_t k = tid;
    if(tid <= n){
        while(k != 0){
            uint32_t digit = k % 10;
            k /= 10;
            sum += digit;
            prod *= digit;
        }
        if(tid == sum * prod) printf("%u\n", tid);
    }
	return;
}

int main() {
	// Checking numbers under 2^27 for sum-product numbers
    printf("First invocation\n");
    cuStopwatch sw1;
    sw1.start();
	kernel<<<(1<<27), 1>>>(1<<27);
    printf("%.4fms\n", sw1.stop());
    printf("\nSecond invocation\n");
    cuStopwatch sw2;
    sw2.start();
    kernel<<<(1<<17), (1<<10)>>>(1<<27);
    printf("%.4fms\n", sw2.stop());
	return 0;
}