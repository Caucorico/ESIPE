#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel() {
	printf("%d, %d\n", threadIdx.x, blockIdx.x);
	return;
}

int main() {
	// main iteration
	kernel <<<16, 4, 0>>>();
	return 0;
}