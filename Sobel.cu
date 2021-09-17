//
// Created by wyl on 2021/9/17.
//
#include<iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
__global__ void SobelInCuda(char* src, char* dst, int width, int height){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int index = idy * width + idx;
    int gx, gy;
    if(idx>0 && idx<width-1 && idy>0 && idy<height-1){
        gx = -src[index-width-1]+src[index-width+1]-2*src[index-1]+2*src[index+1]-src[index+width-1]+src[index+width+1];
        gy = -src[index-width-1]-2*src[index-width]-src[index-width+1]+src[index+width-1]+2*src[index+width]+src[index+width+1];
        dst[index] = (abs(gx) + abs(gy)) / 2;
    }
}

int main(){
    Mat img = imread("../1.jpg", 0);
    int width = img.cols;
    int height = img.rows;

    // 创建GPU内存
    char *dst;
    char *src;
    cudaMalloc((void**)&dst, width*height*sizeof(char));
    cudaMalloc((void**)&src, width*height*sizeof(char));

    // 定义grid和block
    dim3 blockSize(32, 32);
    dim3 gridSize(width/blockSize.x+1, height/blockSize.y+1);

    // img传递到gpu
    cudaMemcpy(src, img.data, width*height*sizeof(char), cudaMemcpyHostToDevice);

    // 运行核函数
    SobelInCuda<<<gridSize, blockSize>>>(src, dst, width, height);

    // gpu返回至cpu
    Mat res(height, width, CV_8UC1, Scalar(0));
    cudaMemcpy(res.data, dst, width*height*sizeof(char), cudaMemcpyDeviceToHost);
    cudaFree(src);
    cudaFree(dst);
    imshow("", res);
    waitKey();
};