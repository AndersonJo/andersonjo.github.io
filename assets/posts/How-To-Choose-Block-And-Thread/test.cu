#include <stdio.h>
#define N (1024*33)

__global__ void add(int *a, int *b, int *result){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while(tid<N){
		result[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
	printf("gridDim.x: %d, blockIdx.x: %d, threadIdx.x: %d, blockDim.x: %d\n",gridDim.x, blockIdx.x, threadIdx.x, blockDim.x);
}

void deviceInfo(){
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
	printf("maxGridSize: %p\n", prop.maxGridSize[0]);
	printf("maxGridSize: %d\n", prop.maxGridSize[1]);
	printf("maxGridSize: %d\n", prop.maxGridSize[2]);
}

int main(int argc, char **argv) {
	int a[N];
	int b[N];
	int c[N];
	int *dev_a;
	int *dev_b;
	int *dev_c;

	deviceInfo();



	for(int i=0; i<N; i++){
		a[i] = i;
		b[i] = i*2;
	}

	cudaMalloc((void**)&dev_a, sizeof(int)*N);
	cudaMalloc((void**)&dev_b, sizeof(int)*N);
	cudaMalloc((void**)&dev_c, sizeof(int)*N);

	cudaMemcpy(dev_a, a, sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, sizeof(int)*N, cudaMemcpyHostToDevice);

	add<<<1, 10>>>(dev_a, dev_b, dev_c);

	cudaMemcpy(c, dev_c, sizeof(int)*N, cudaMemcpyDeviceToHost);

	double result =0;
	for(int i=0; i<N; i++){
//		printf("%d + %d = %d\n", a[i], b[i], c[i]);
		result += c[i];
	}
	printf("result: %f", result);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
