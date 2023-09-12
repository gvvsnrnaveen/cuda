#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <helper_cuda.h>
#include <helper_functions.h>

__global__ void addition_kernel(int *a, int *b, int *c){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	c[idx] = a[idx] + b[idx];
}

__global__ void subtraction_kernel(int *a, int *b, int *c){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	c[idx] = b[idx] - a[idx];
}

bool validate_addition_result(int *c, int n){
	int i = 0;
	for(i = 0; i < n; i++){
		if(c[i] != ((3 * i) + (5 * i)))
			return false;
	}
	return true;
}

bool validate_subtraction_result(int *d, int n){
	int i = 0;
	for(i = 0; i < n; i++){
		if(d[i] != ((5 * i) - (3 * i)))
			return false;
	}
	return true;
}

int main(int argc, char **argv){
	int devId;
	int numberOfDevices = 0;
	cudaDeviceProp deviceProps;

	cudaGetDeviceCount(&numberOfDevices);
	printf("Number of cuda devices: %d\n", numberOfDevices);

	devId = findCudaDevice(argc, (const char**)argv);
	printf("Device ID found: %d\n", devId);

	checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devId));
	printf("Device Name: %s\n", deviceProps.name);

	int n = 16 * 1024 * 1024;
	int nbytes = n * sizeof(int);
	int i = 0;
	int *a = NULL, *b = NULL, *c = NULL, *d = NULL;
	checkCudaErrors(cudaMallocHost((void**)&a, nbytes));
	memset(a, 0, nbytes);
	checkCudaErrors(cudaMallocHost((void**)&b, nbytes));
	memset(b, 0, nbytes);
	checkCudaErrors(cudaMallocHost((void**)&c, nbytes));
	memset(c, 0, nbytes);
	checkCudaErrors(cudaMallocHost((void**)&d, nbytes));
	memset(d, 0, nbytes);

	for(i=0; i < n; i++){
		a[i] = 3 * i;
		b[i] = 5 * i;
	}

	int *d_a = NULL, *d_b = NULL, *d_c = NULL, *d_d = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_a, nbytes));
	checkCudaErrors(cudaMemset(d_a, 0, nbytes));
	checkCudaErrors(cudaMalloc((void**)&d_b, nbytes));
	checkCudaErrors(cudaMemset(d_b, 0, nbytes));
	checkCudaErrors(cudaMalloc((void**)&d_c, nbytes));
	checkCudaErrors(cudaMemset(d_c, 0, nbytes));
	checkCudaErrors(cudaMalloc((void**)&d_d, nbytes));
	checkCudaErrors(cudaMemset(d_d, 0, nbytes));

	dim3 threads = dim3(512, 1);
	dim3 blocks = dim3( n / threads.x, 1);

	cudaEvent_t start, stop;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaProfilerStart());

	cudaMemcpy(d_a, a, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, nbytes, cudaMemcpyHostToDevice);
	cudaEventRecord(start, 0);
	
	addition_kernel<<<blocks, threads, 0, 0>>>(d_a, d_b, d_c);
	subtraction_kernel<<<blocks, threads, 0, 0>>>(d_a, d_b, d_d);
	cudaEventRecord(stop, 0);
	cudaMemcpy(c, d_c, nbytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(d, d_d, nbytes, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);

	float gpu_time = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

	printf("GPU Time: %.2fms\n", gpu_time);
	printf("Addition Result is: %d\n", validate_addition_result(c, n));
	printf("Subtraction Result is: %d\n", validate_subtraction_result(d, n));
	
	checkCudaErrors(cudaFreeHost(a));
	checkCudaErrors(cudaFreeHost(b));
	checkCudaErrors(cudaFreeHost(c));
	checkCudaErrors(cudaFree(d_a));
	checkCudaErrors(cudaFree(d_b));
	checkCudaErrors(cudaFree(d_c));
	return 0;
}
