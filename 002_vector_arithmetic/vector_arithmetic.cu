#include <stdio.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

__global__ void vector_addition_kernel(int *ga, int *gb, int *gc, int n){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < n){
		gc[idx] = ga[idx] + gb[idx];
	}
}

__global__ void vector_subtraction_kernel(int *ga, int *gb, int *gc, int n){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n){
		gc[idx] = ga[idx] - gb[idx];
	}
}

__global__ void vector_multiplication_kernel(int *ga, int *gb, int *gc, int n){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < n){
		gc[idx] = ga[idx] * gb[idx];
	}
}

__global__ void vector_division_kernel(int *ga, int *gb, int *gc, int n){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < n){
		gc[idx] = ga[idx] / gb[idx];
	}
}

bool check_output(int *ptr, int n, int value, int *result){
	for(int i = 0; i < n; i++){
		*result = ptr[i];
		if(ptr[i] != value)
			return false;
	}

	return true;
}

int main(int argc, char **argv){

	int numberOfDevices = 0;
	int deviceId; 
	cudaDeviceProp deviceProps;
	size_t gpu_free_mem, gpu_total_mem;
	


	cudaGetDeviceCount(&numberOfDevices);
	printf("Number of cuda devices: %d\n", numberOfDevices);
	
	deviceId = findCudaDevice(argc, (const char**)argv);
	printf("Device Number: %d\n", deviceId);

	checkCudaErrors(cudaGetDeviceProperties(&deviceProps, deviceId));
	printf("Device name: %s\n", deviceProps.name);

	cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem);
	printf("GPU Mem stats: free - %ld, total - %ld \n", gpu_free_mem, gpu_total_mem);

	int n = 16 * 1024 * 1024;
	int nbytes = n * sizeof(int);

	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	int i = 0;
	int result = 0;

	checkCudaErrors(cudaMallocHost((void**)&a, nbytes));
	memset(a, 0, nbytes);
	checkCudaErrors(cudaMallocHost((void**)&b, nbytes));
	memset(b, 0, nbytes);
	for(i=0; i<n; i++){
		a[i] = 15;
		b[i] = 5;
	}
	checkCudaErrors(cudaMallocHost((void**)&c, nbytes));
	memset(c, 0, nbytes);

	checkCudaErrors(cudaMalloc((void**)&d_a, nbytes));
	checkCudaErrors(cudaMemset(d_a, 0, nbytes));
	checkCudaErrors(cudaMalloc((void**)&d_b, nbytes));
	checkCudaErrors(cudaMemset(d_b, 0, nbytes));
	checkCudaErrors(cudaMalloc((void**)&d_c, nbytes));
	checkCudaErrors(cudaMemset(d_c, 0, nbytes));

	cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem);
	printf("GPU Mem stats: free - %ld, total - %ld \n", gpu_free_mem, gpu_total_mem);


	dim3 threads = dim3(512, 1);
	dim3 blocks = dim3( n / threads.x, 1);

	checkCudaErrors(cudaMemcpy(d_a, a, nbytes, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_b, b, nbytes, cudaMemcpyHostToDevice));
	vector_addition_kernel<<<blocks, threads, 0, 0>>>(d_a, d_b, d_c, n);
	checkCudaErrors(cudaMemcpy(c, d_c, nbytes, cudaMemcpyDeviceToHost));
	printf("Result of addition: %s\n", check_output(c, n, 20, &result) ? "PASS": "FAIL");
	printf("Result: %d\n", result);

	vector_subtraction_kernel<<<blocks, threads, 0, 0>>>(d_a, d_b, d_c, n);
	checkCudaErrors(cudaMemcpy(c, d_c, nbytes, cudaMemcpyDeviceToHost));
	printf("Result of subtraction: %s\n", check_output(c, n, 10, &result) ? "PASS": "FAIL");
	printf("Result: %d\n", result);

	vector_multiplication_kernel<<<blocks, threads, 0, 0>>>(d_a, d_b, d_c, n);
	checkCudaErrors(cudaMemcpy(c, d_c, nbytes, cudaMemcpyDeviceToHost));
	printf("Result of multiplication: %s\n", check_output(c, n, 75, &result) ? "PASS": "FAIL");
	printf("Result: %d\n", result);

	vector_division_kernel<<<blocks, threads, 0, 0>>>(d_a, d_b, d_c, n);
	checkCudaErrors(cudaMemcpy(c, d_c, nbytes, cudaMemcpyDeviceToHost));
	printf("Result of division: %s\n", check_output(c, n, 3, &result) ? "PASS": "FAIL");
	printf("Result: %d\n", result);


	checkCudaErrors(cudaFree(d_a));
	checkCudaErrors(cudaFree(d_b));
	checkCudaErrors(cudaFree(d_c));
	checkCudaErrors(cudaFreeHost(a));
	checkCudaErrors(cudaFreeHost(b));
	checkCudaErrors(cudaFreeHost(c));

	cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem);
	printf("GPU Mem stats: free - %ld, total - %ld \n", gpu_free_mem, gpu_total_mem);

	return 0;
}
