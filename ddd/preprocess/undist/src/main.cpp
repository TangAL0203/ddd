#include "undistort.hpp"
#include "undistort_cuda.cuh"
#include <iostream>
#include <cstdlib>

using namespace cv;
using namespace std;

int main()
{
    bool use_cuda = true;
    string imagePath;
    getline(cin, imagePath);

    if (use_cuda)
    {
        UndistortImagesCuda undistortImages;
        for (int i = 0; i < 100; i++)
        {
            undistortImages.getUndistortMap();
            undistortImages.undistortImage(imagePath);
        }
        while(1);
    }
    else
    {
        UndistortImages undistortImages;
        for (int i = 0; i < 100; i++)
        {
            undistortImages.getUndistortMap();
            undistortImages.undistortImage(imagePath);
        }
    }

    return 1;
}
