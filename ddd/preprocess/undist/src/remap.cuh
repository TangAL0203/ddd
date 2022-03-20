#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void cudaRemap(uchar *pSrcImg, uchar *pDstImg, float *pMapx, float *pMapy,
                          int inWidth, int inHeight, int outWidth, int outHeight, int channels);