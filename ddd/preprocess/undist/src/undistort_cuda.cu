#include <thread>
#include "undistort_cuda.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "remap.cuh"

using namespace cv;
using namespace std;

void UndistortImagesCuda::getUndistortMap()
{
    cv::Size imageSize(imgWidth, imgHeight);
    auto t1 = std::chrono::system_clock::now();
    cv::fisheye::initUndistortRectifyMap(cameraMatrix, distCoeffs, Matx33d::eye(), cameraMatrix, imageSize, CV_32FC1, map1, map2);
    auto t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time = t2 - t1;
    cout << "getUndistortMap cost time is: " << elapsed_time.count() * 1000 << "(ms)" << endl;
}

void UndistortImagesCuda::undistortImage(string imagePath)
{
    cout << " Undistorting image... " << endl;

    //    string imagePath;
    //    getline(cin, imagePath);

    Mat inputImage = imread(imagePath, CV_LOAD_IMAGE_COLOR);
    if (!inputImage.data)
    {
        cout << " Could not open or find the image: " << imagePath << endl;
        cout << " Verify if the input images path are absolute," << endl;
        cout << " or change the program directory." << endl;
        exit(EXIT_FAILURE);
    }
    imwrite("src.jpg", inputImage);

    cv::Mat outputImage = cv::Mat(imgHeight, imgWidth, CV_8UC3);

    cudaError err;
    dim3 block(16, 16);
    dim3 grid((imgWidth + block.x - 1) / block.x, (imgHeight + block.y - 1) / block.y);

    uchar *pSrcImgData = NULL;
    uchar *pDstImgData = NULL;
    float *pMapxData = NULL;
    float *pMapyData = NULL;

    // cout << "src map1 行数: " << map1.rows << endl;
    // cout << "src map1 列数: " << map1.cols << endl;
    // cout << "src 通道: " << map1.channels() << endl;

    map1 = map1.reshape(imgHeight * imgWidth, 1);
    map2 = map2.reshape(imgHeight * imgWidth, 1);

    // cudaEvent_t start;
    // cudaEvent_t stop;
    // cudaEventRecord(start, 0);

    {
        err = cudaMalloc(&pMapxData, imgHeight * imgWidth * sizeof(float));
        err = cudaMalloc(&pMapyData, imgHeight * imgWidth * sizeof(float));
        err = cudaMalloc(&pDstImgData, imgHeight * imgWidth * sizeof(uchar) * channels);
        err = cudaMalloc(&pSrcImgData, imgHeight * imgWidth * sizeof(uchar) * channels);
    }
    {
        err = cudaMemcpy(pMapxData, map1.data, imgHeight * imgWidth * sizeof(float), cudaMemcpyHostToDevice);
        err = cudaMemcpy(pMapyData, map2.data, imgHeight * imgWidth * sizeof(float), cudaMemcpyHostToDevice);
        err = cudaMemcpy(pSrcImgData, inputImage.data, imgHeight * imgWidth * sizeof(uchar) * channels, cudaMemcpyHostToDevice);
    }

    auto t1 = std::chrono::system_clock::now();
    // cudaRemap<<<grid, block>>>(pSrcImgData, pDstImgData, pMapxData, pMapyData, imgWidth, imgHeight, imgWidth, imgHeight, channels);
    cudaRemap<<<8, 1024>>>(pSrcImgData, pDstImgData, pMapxData, pMapyData, imgWidth, imgHeight, imgWidth, imgHeight, channels);
    cudaThreadSynchronize();

    auto t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time = t2 - t1;
    cout << "undistortImage cost time is: " << elapsed_time.count() * 1000 << "(ms)" << endl;

    err = cudaGetLastError();
    err = cudaMemcpy(outputImage.data, pDstImgData, imgWidth * imgHeight * sizeof(uchar) * channels, cudaMemcpyDeviceToHost);

    // float elapsed_time;
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsed_time, start, stop);
    // cout << "undistortImage cost time is: " << elapsed_time*1000 << "(ms)" << endl;

    imwrite("undist.jpg", outputImage);

    cout << " Done !" << endl;
}
