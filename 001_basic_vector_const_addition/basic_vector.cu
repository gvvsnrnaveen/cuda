#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

__global__ void vector_increment_kernel(int *ga, int value){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	ga[idx] += value;
}

bool check_output(int *a, int n, int value){
	for(int i = 0; i < n; i++){
		if( a[i] != value ){
			return false;
		}
	}
	return true;
}

int main(int argc, char **argv){
	int numberOfDevices = 0;
	int deviceId;
	cudaDeviceProp deviceProps;

	cudaGetDeviceCount(&numberOfDevices);
	printf("Number of devices: %d\n", numberOfDevices);

	deviceId = findCudaDevice(argc, (const char**)argv);
	checkCudaErrors(cudaGetDeviceProperties(&deviceProps, deviceId));
	printf("Cuda Device Name: %s\n", deviceProps.name);

	int n = 16 * 1024 * 1024;
	int nbytes = n * sizeof(int);
	int value = 26;

	size_t gpu_free_mem, gpu_total_mem;
	checkCudaErrors(cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem));
	printf("GPU Mem stats: free - %ld, total - %ld\n", gpu_free_mem, gpu_total_mem);

	int *a, *d_a;
	checkCudaErrors(cudaMallocHost((void**)&a, nbytes));
	memset(a, 0, nbytes);

	checkCudaErrors(cudaMalloc((void**)&d_a, nbytes));
	checkCudaErrors(cudaMemset(a, 0, nbytes));

	checkCudaErrors(cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem));
	printf("GPU Mem stats: free - %ld, total - %ld\n", gpu_free_mem, gpu_total_mem);

	dim3 threads = dim3(512, 1);
	dim3 blocks = dim3( n /threads.x, 1);


	cudaMemcpy(d_a, a, nbytes, cudaMemcpyHostToDevice);

	vector_increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
	cudaMemcpy(a, d_a, nbytes, cudaMemcpyDeviceToHost);

	printf("Result: %d\n", check_output(a, n, value));

	checkCudaErrors(cudaFreeHost(a));
	checkCudaErrors(cudaFree(d_a));

	checkCudaErrors(cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem));
	printf("GPU Mem stats: free - %ld, total - %ld\n", gpu_free_mem, gpu_total_mem);

	return 0;
}
