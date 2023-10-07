#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <cstring>
#include <iostream>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <builtin_types.h>

#define checkCudaDrvErrors(val) check((val), #val, __FILE__, __LINE__) 

void randomFloat(float *f, int n){
	for (int i = 0; i < n; i++){
		f[i] = rand() / (float)RAND_MAX;
	}
}

void print_data(float *a, float *b, float *c, int n){
	for(int i = 0; i < n; i++){
		printf("[%f] + [%f] = [%f]\n", a[i], b[i], c[i]);
	}
}

int main(int argc, char **argv){
	int deviceId = 0;
	cudaDeviceProp deviceProp;
	CUmodule cuModule;
	CUfunction cuKernel;
	CUresult result;

	float *a = NULL, *b = NULL, *c = NULL;
	float *d_a = NULL, *d_b = NULL, *d_c = NULL;

	int n = 16 * 1024 *1024;
	if(argc < 3){
		printf("usage: %s <fatbin_file_location> <kernel_function>\n", argv[0]);
		return -1;
	}

	deviceId = findCudaDevice(argc, (const char**)argv);
	printf("cuda device found: %d\n", deviceId);

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, deviceId));
	printf("cuda device name: %s\n", deviceProp.name);

	result = cuInit(0);
	if(result != 0){
		printf("cuda failed: %d\n", result);
		return -1;
	}

	// Load module data in binary mode
	std::ostringstream binstream;
	std::ifstream fatbinFile(argv[1], std::ios::binary);
	binstream << fatbinFile.rdbuf();
	fatbinFile.close();

	// print the fatbin file size
	printf("fatbin file size: %ld\n", binstream.str().size());

	// load the fatbin file as module
	result = cuModuleLoadData(&cuModule, binstream.str().c_str());
	if(result != 0){
		printf("cuda module loading failed\n");
		return -1;
	}
	printf("Module Loaded\n");

	// Retreive the function in the fatbin file
	result = cuModuleGetFunction(&cuKernel, cuModule, argv[2]);
	if(result != 0){
		printf("cuda module function loading failed\n");
		return -1;
	}
	printf("Module function retrieved\n");

	// allocate host memory
	checkCudaErrors(cudaMallocHost((void**)&a, sizeof(float) * n ));
	checkCudaErrors(cudaMallocHost((void**)&b, sizeof(float) * n ));
	checkCudaErrors(cudaMallocHost((void**)&c, sizeof(float) * n ));

	// allocate device memory
	checkCudaErrors(cudaMalloc((void**)&d_a, sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&d_b, sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&d_c, sizeof(float) * n));

	// generate random float data
	randomFloat(a, n);
	randomFloat(b, n);
	
	void *args[] = {&d_a, &d_b, &d_c, &n};
	int threadsPerBlock = 512;
	int blocksPerGrid = (n / threadsPerBlock - 1) / threadsPerBlock;

	cudaEvent_t start, stop;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaEventRecord(start, 0));
	checkCudaErrors(cudaMemcpy(d_a, a, sizeof(float) * n, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_b, b, sizeof(float) * n, cudaMemcpyHostToDevice));
		
	// run kernel
	result = cuLaunchKernel(cuKernel, threadsPerBlock, 1, 1, blocksPerGrid, 1, 1, 0, 0, args, NULL);
	if(result != 0){
		printf("launch kernel failed: %d\n", result);
	} else {
		printf("kernel success\n");
	}

	checkCudaErrors(cudaMemcpy(c, d_c, sizeof(float) * n, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));

	float gpu_time = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));
	printf("GPU Time: %f ms\n", gpu_time);

	// free up the host and device memory
	checkCudaErrors(cudaFreeHost(a));
	checkCudaErrors(cudaFreeHost(b));
	checkCudaErrors(cudaFreeHost(c));
	checkCudaErrors(cudaFree(d_a));
	checkCudaErrors(cudaFree(d_b));
	checkCudaErrors(cudaFree(d_c));
	return 0;
}
