extern "C" __global__ void kernel_multi_vector_addition(const float *a, const float *b, float *c, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		c[i] = a[i] + b[i];
	}
}

extern "C" __global__ void kernel_multi_vector_multiplication(const float *a, const float *b, float *c, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		c[i] = a[i] * b[i];
	}
}
