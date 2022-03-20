#ifndef _UNDISTORT_HPP
#define _UNDISTORT_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

class UndistortImages
{
private:
    // Camera parameters
    cv::Mat cameraMatrix = (Mat_<float>(3, 3) << 931.9622,   0.    , 935.8878,  0.    , 938.0016, 516.46242, 0, 0, 1);
    // cv::Mat distCoeffs = (Mat_<float>(4, 1) << -0.01875662,  0.01232133, -0.02096538,  0.0072541);
    // std::vector<float> distCoeffs = {-0.01875662,  0.01232133, -0.02096538,  0.0072541};
    const int imgWidth = 1920;
    const int imgHeight = 1080;
    const int channels = 3;
    cv::Mat map1;
    cv::Mat map2;
    std::vector<std::string> inputImagesPaths;

public:
    void getUndistortMap();
    void undistortImage(string imagePath);
};

#endif
