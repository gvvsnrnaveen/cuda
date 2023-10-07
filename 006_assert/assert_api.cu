#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

__global__ void testKernel(int N){
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	assert(gid < N);
}

int main(int argc, char **argv){
	int deviceId; 
	cudaDeviceProp deviceProp;
	cudaError_t error;

	int blocks = 2;
	int threads = 32;

	deviceId = findCudaDevice(argc, (const char**)argv);
	printf("Device id found: %d\n", deviceId);

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, deviceId));
	printf("Device Name: %s\n", deviceProp.name);

	dim3 dimGrid(blocks);
	dim3 dimBlock(threads);

	testKernel<<<dimGrid, dimBlock>>>(60);

	printf("Begin: Assert\n");
	error = cudaDeviceSynchronize();
	printf("End: Assert\n");

	if(error == cudaErrorAssert){
		printf("Cuda Assert: %s\n", cudaGetErrorString(error));
	}

	return 0;
}
