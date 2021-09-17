//
// Created by wyl on 2020/11/14.
//

#include <iostream>
#include <sys/time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

// 核函数
// 1D1D  两个向量加法kernel
__global__ void add(float* x, float * y, float* z, int n)
{
    // 获取全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    z[index] = x[index] + y[index];
}

int main()
{
    int N = 102400;
    int nBytes = N * sizeof(float);

    // 申请托管内存
    float *x, *y, *z;
    cudaMallocManaged((void**)&x, nBytes);
    cudaMallocManaged((void**)&y, nBytes);
    cudaMallocManaged((void**)&z, nBytes);

    // 初始化数据
    for (int i = 0; i < N; ++i)
    {
        x[i] = 10.0;
        y[i] = 20.0;
    }

    // 定义kernel的执行配置
    dim3 blockSize(512);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    // 执行kernel
    // 创建gridSize个线程块在GPU上运行
    for(int i=0; i<1000; i++){
        add <<< gridSize, blockSize >>>(x, y, z, N);
    }

    // 同步device 保证结果能正确访问
    cudaDeviceSynchronize();
    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(z[i] - 30.0));
    std::cout << "最大误差: " << maxError << std::endl;

    // 释放内存
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return 0;
}