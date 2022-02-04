//
// Created by wyl on 2022/2/4.
//
#include <iostream>

__global__ void add(int* a, int* b, int* c, int num){
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= 0 && index < num){
        c[index] = a[index] + b[index];
    }
}

int main(){
    int num = 10000;
    int* a = new int[num];
    int* b = new int[num];
    for(int i=0; i<num; i++){
        a[i] = i;
        b[i] = i*i;
    }

    int *a_cuda, *b_cuda, *c_cuda;
    cudaMalloc((void**)&a_cuda, num * sizeof(int));
    cudaMalloc((void**)&b_cuda, num * sizeof(int));
    cudaMalloc((void**)&c_cuda, num * sizeof(int));

    cudaMemcpy(a_cuda, a, num*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_cuda, b, num*sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize(num/blockSize.x+1);
    add<<<gridSize, blockSize>>>(a_cuda, b_cuda, c_cuda, num);

    int* c = new int[num];
    cudaMemcpy(c, c_cuda, num*sizeof(int), cudaMemcpyDeviceToHost);

    // cpu
    int* c_cpu = new int[num];
    for(int i=0; i<num; i++){
        c_cpu[i] = a[i] + b[i];
    }

    for(int i=0; i<num; i++){
        printf("%d + %d = %d vs %d\n", (int)a[i], (int)b[i], (int)c[i], (int)c_cpu[i]);
    }

    cudaFree(a_cuda);
    cudaFree(b_cuda);
    cudaFree(c_cuda);

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;

}
