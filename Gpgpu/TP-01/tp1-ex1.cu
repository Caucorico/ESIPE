#include <stdio.h>
#include <cuda_runtime.h>

int main()
{
	cudaDeviceProp* cdp = (cudaDeviceProp*) malloc(sizeof(cudaDeviceProp));
	int deviceCount = 0, i;
	cudaGetDeviceCount(&deviceCount);
	printf("Number of devices : %d\n", deviceCount);

	for ( i = 0 ; i < deviceCount ; i++ )
	{
		cudaGetDeviceProperties(cdp, i);
		printf("Device n°%d (%s)\n", i, cdp->name);
		printf("frequency : %d KHz\n", cdp->clockRate);
		printf("Global memory size : %zd bytes\n", cdp->totalGlobalMem);
		printf("WarpSize : %d threads\n", cdp->warpSize);
	}

	free(cdp);
	return 0;
}
