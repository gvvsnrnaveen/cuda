#include <stdio.h>
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h>


__global__ void sin_cos_kernel(float *a, int offset){
	int idx = offset + threadIdx.x + blockIdx.x * blockDim.x;
	a[idx] = sinf((float)(idx)) + cosf((float)(idx));
}


bool verify_data(float *src, float *dst, int length){
	printf("Verifying: %d elements\n", length);
	for(int i = 0; i < length - 1; i++){
		if(src[i] != dst[i]){
			printf("Result failed at: %d\n", i);
			return false;
		}
	}
	return true;
}

int main(int argc, char **argv){
	int deviceId;
	cudaDeviceProp deviceProps;
	int nstreams = 8;
	int blocksize = 256;
	int n = 16 * 1024 * blocksize * nstreams;
	int streamsize = n / nstreams;
	int streambytes = streamsize * sizeof(float);
	int nbytes = n * sizeof(float);

	deviceId = findCudaDevice(argc, (const char**)argv);
	printf("Cuda device found: %d\n", deviceId);

	checkCudaErrors(cudaGetDeviceProperties(&deviceProps, deviceId));
	printf("Cuda device name: %s\n", deviceProps.name);

	if(!deviceProps.concurrentKernels){
		printf("Concurrent kernels are not supported\n");
		return -1;
	}
	printf("Concurrent kernels are supported: number of concurrent kernels: %d\n", deviceProps.multiProcessorCount);
	printf("============================================\n\n");

	
	float *a = NULL;
	checkCudaErrors(cudaMallocHost((void**)&a, nbytes));
	memset(a, 0, nbytes);

	float *verify = NULL;
	checkCudaErrors(cudaMallocHost((void**)&verify, nbytes));
	memset(verify, 0, nbytes);

	float *d_a = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_a, nbytes));
	checkCudaErrors(cudaMemset(a, 0, nbytes));

	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	// serial execution of the entire data
	cudaEventRecord(start, 0);
	checkCudaErrors(cudaMemcpy(d_a, a, nbytes, cudaMemcpyHostToDevice));
	sin_cos_kernel<<<n/blocksize, blocksize>>>(d_a, 0);
	checkCudaErrors(cudaMemcpy(a, d_a, nbytes, cudaMemcpyDeviceToHost));
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// copy the result to the verify variable, 
	// so that we can verify the concurrent kernel results
	memcpy(verify, a, nbytes);

	printf("Verifying result: %d\n", verify_data(a, verify, n));

	float gpu_time = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));
	printf("Time taken for serial execution: %f ms\n", gpu_time);
	printf("============================================\n\n");


	cudaStream_t streams[nstreams];
	for(int i = 0; i < nstreams; i++){
		checkCudaErrors(cudaStreamCreate(&streams[i]));
	}

	// running the streams in loop of HDCopy, kernel, DHCopy
	memset(a, 0, nbytes);
	checkCudaErrors(cudaEventRecord(start, 0));
	for(int i = 0; i < nstreams; i++){
		int offset = i * streamsize;
		checkCudaErrors(cudaMemcpyAsync(&d_a[offset], &a[offset], streambytes, cudaMemcpyHostToDevice, streams[i]));
		sin_cos_kernel<<<streamsize/blocksize, blocksize, 0, streams[i]>>>(d_a, offset);
		checkCudaErrors(cudaMemcpyAsync(&a[offset], &d_a[offset], streambytes, cudaMemcpyDeviceToHost, streams[i]));
	}
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));

	gpu_time = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));
	printf("%d stream execution time: %f ms\n", nstreams, gpu_time);
	printf("Stream result verification: %d\n", verify_data(a, verify, n));
	printf("=============================================\n\n");


	// running the streams in individual loops - loop HDCopy, loop kernel, loop DHCopy
	memset(a, 0, nbytes);
	checkCudaErrors(cudaEventRecord(start, 0));
	for(int i = 0; i < nstreams; i++){
		int offset = i * streamsize;
		checkCudaErrors(cudaMemcpyAsync(&d_a[offset], &a[offset], streambytes, cudaMemcpyHostToDevice, streams[i]));
	}
	for(int i = 0; i < nstreams; i++){
		int offset = i * streamsize;
		sin_cos_kernel<<<streamsize/blocksize, blocksize, 0, streams[i]>>>(d_a, offset);
	}
	for(int i = 0; i < nstreams; i++){
		int offset = i * streamsize;
		checkCudaErrors(cudaMemcpyAsync(&a[offset], &d_a[offset], streambytes, cudaMemcpyDeviceToHost, streams[i]));
	}
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	gpu_time = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));
	printf("%d stream execution time individual loop: %f ms\n", nstreams, gpu_time);
	printf("Stream result verification: %d\n", verify_data(a, verify, n));

	for(int i = 0; i < nstreams; i++){
		checkCudaErrors(cudaStreamDestroy(streams[i]));
	}

	checkCudaErrors(cudaFreeHost(a));
	checkCudaErrors(cudaFreeHost(verify));
	checkCudaErrors(cudaFree(d_a));

	return 0;
}
