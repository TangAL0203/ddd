#include <thread>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "undistort.hpp"

using namespace cv;
using namespace std;

void UndistortImages::getUndistortMap()
{
    std::vector<float> distCoeffs;
    distCoeffs.push_back(-0.01875662);
    distCoeffs.push_back(0.01232133);
    distCoeffs.push_back(-0.02096538);
    distCoeffs.push_back(0.0072541);
    cv::Size imageSize(imgWidth, imgHeight);
    auto t1 = std::chrono::system_clock::now();
    cv::fisheye::initUndistortRectifyMap(cameraMatrix, distCoeffs, Matx33d::eye(), cameraMatrix, imageSize, CV_32FC1, map1, map2);
    auto t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time = t2 - t1;
    cout << "getUndistortMap cost time is: " << elapsed_time.count() * 1000 << "(ms)" << endl;
}

void UndistortImages::undistortImage(string imagePath)
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
    Mat outputImage;
    auto t1 = std::chrono::system_clock::now();
    cv::remap(inputImage, outputImage, map1, map2, INTER_LINEAR);
    auto t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time = t2 - t1;
    cout << "undistortImage cost time is: " << elapsed_time.count() * 1000 << "(ms)" << endl;
    imwrite("undist.jpg", outputImage);

    cout << " Done !" << endl;
}
