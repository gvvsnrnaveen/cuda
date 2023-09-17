#include <stdio.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define ADDITION_CONST 50

__global__ void kernel_addition(int *a){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	a[idx] += ADDITION_CONST;
}

bool verify_result(int *a, int length){
	int i = 0;
	for(i = 0; i < length; i++){
		if(a[i] != ADDITION_CONST)
			return false;
	}
	return true;
}

int main(int argc, char **argv){

	int deviceId;
	cudaDeviceProp deviceProps;

	deviceId = findCudaDevice(argc, (const char**)argv);
	printf("Cuda device found: %d\n", deviceId);

	checkCudaErrors(cudaGetDeviceProperties(&deviceProps, deviceId));
	printf("Cuda Device Name: %s\n", deviceProps.name);

	int n = 16 * 1024 * 1024;
	int nbytes = n * sizeof(int);
	int blocksize = 256;
	int threads = 512;

	int *a = NULL;
	int *d_a = NULL;

	checkCudaErrors(cudaMallocHost((void**)&a, nbytes));
	memset(a, 0, nbytes);

	checkCudaErrors(cudaMalloc((void**)&d_a, nbytes));
	checkCudaErrors(cudaMemset(d_a, 0, nbytes));

	cudaEvent_t start, stop;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaEventRecord(start, 0));
	checkCudaErrors(cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0));
	kernel_addition<<< n/blocksize, threads, 0 , 0>>>(d_a);
	checkCudaErrors(cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0));
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));

	float gpu_time = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

	printf("Result validation: %d\n", verify_result(a, n));
	printf("GPU Time: %f ms\n", gpu_time);

	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
	checkCudaErrors(cudaFreeHost(a));
	checkCudaErrors(cudaFree(d_a));

	return 0;
}
