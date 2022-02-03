//
// Created by wyl on 2021/9/17.
//
#include<iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define byte unsigned char
// 有问题
__global__ void SobelInCuda(const byte* src, byte* dst, const int width, const int height){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int index = idy * width + idx;
    float gx, gy;
    if(idx>0 && idx<width-1 && idy>0 && idy<height-1){
        gx = -src[index-width-1]+src[index-width+1]-2*src[index-1]+2*src[index+1]-src[index+width-1]+src[index+width+1];
        gy = -src[index-width-1]-2*src[index-width]-src[index-width+1]+src[index+width-1]+2*src[index+width]+src[index+width+1];
        dst[index] = (abs(gx) + abs(gy)) / 2;
//        dst[index] = sqrt(gx*gx + gy*gy);
    }
}

int main(){
    Mat img = imread("../1.jpg", 0);
    int width = img.cols;
    int height = img.rows;

    int len = width * height * sizeof(byte);

    byte* imgData = new byte[len];
    std::memcpy(imgData, img.data, len);

    // 创建GPU内存
    byte *dst;
    byte *src;
    cudaMalloc((void**)&dst, width*height*sizeof(byte));
    cudaMalloc((void**)&src, width*height*sizeof(byte));

    // img传递到gpu
    cudaMemcpy(src, imgData, width*height*sizeof(byte), cudaMemcpyHostToDevice);

    // 定义grid和block
    dim3 blockSize(32, 32);
    dim3 gridSize(width/blockSize.x+1, height/blockSize.y+1);

    // 运行核函数
    SobelInCuda<<<gridSize, blockSize>>>(src, dst, width, height);

    // gpu返回至cpu
    cudaMemcpy(imgData, dst, width*height*sizeof(byte), cudaMemcpyDeviceToHost);

    Mat res(height, width, CV_8UC1, imgData);

    cudaFree(src);
    cudaFree(dst);
    imshow("cuda", res);
    imwrite("cuda.jpg", res);
    waitKey();
}
