#include <thread>
#include <stdlib.h>
#include <iostream>
#include "remap.cuh"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


__global__ void cudaRemap(uchar *pSrcImg, uchar *pDstImg, float *pMapx, float *pMapy,
                          int inWidth, int inHeight, int outWidth, int outHeight, int channels)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t output_img_size = outWidth * outHeight;
    uint32_t total_thread_number = gridDim.x * blockDim.x;

    for (; idx < output_img_size; idx += total_thread_number) {

        uint32_t idx_y = idx / outWidth;
        uint32_t idx_x = idx - idx_y * outWidth;

        float u = pMapx[idx];
        float v = pMapy[idx];

        int u1 = floor(u);
        int v1 = floor(v);
        int u2 = u1 + 1;
        int v2 = v1 + 1;
        if (u1 >= 0 && v1 >= 0 && u2 < inWidth && v2 < inHeight)
        {
            float dx = u - u1;
            float dy = v - v1;
            float weight1 = (1 - dx) * (1 - dy);
            float weight2 = dx * (1 - dy);
            float weight3 = (1 - dx) * dy;
            float weight4 = dx * dy;

            int resultIdx = idx * 3;
            for (int chan = 0; chan < channels; chan++)
            {
                pDstImg[resultIdx + chan] = uchar(weight1 * pSrcImg[(v1 * inWidth + u1) * 3 + chan] +
                                                  weight2 * pSrcImg[(v1 * inWidth + u2) * 3 + chan] +
                                                  weight3 * pSrcImg[(v2 * inWidth + u1) * 3 + chan] +
                                                  weight4 * pSrcImg[(v2 * inWidth + u2) * 3 + chan] + 0.5);
            }
        }

    }
}