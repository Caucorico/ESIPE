#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void kernel() {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t n = tid;
	uint32_t sum = 0;
    uint32_t prod = 1;
    while(n != 0){
        uint32_t digit = n % 10;
        n /= 10;
        sum += digit;
        prod *= digit;
    }
    if(sum*prod == tid) printf("%u\n", tid);
	return;
}

void checkrange(uint32_t range){
    double dim = sqrt(range);
	uint32_t thread_number = (uint32_t)ceil(range/(dim));

	if ( thread_number > 1024 ) {
		printf("Impossible to run more threads than 1024.\nNumber reduce to 1024. \n");
		thread_number = 1024;
	}

    printf("Checking %u for sum-product numbers\n", range);
    /* If the number of threads is greater than 1024, the code will not be executed. */
    kernel<<<(uint32_t)dim, thread_number, 0>>>();
    cudaDeviceSynchronize();
}

int main() {
	// main iteration
	checkrange(1024);
    checkrange(16777216);
	return 0;
}