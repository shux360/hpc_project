#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;

double computeRMSE(const Mat &a, const Mat &b)
{
    if (a.empty() || b.empty() || a.size() != b.size() || a.type() != b.type())
    {
        throw runtime_error("Images must have same size and type");
    }
    double mse = 0.0;
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < a.cols; j++)
        {
            double diff = a.at<uchar>(i, j) - b.at<uchar>(i, j);
            mse += diff * diff;
        }
    }
    mse /= (a.rows * a.cols);
    return sqrt(mse);
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        cerr << "Usage: ./rmse_compare <serial_image> <parallel_image>\n";
        return 1;
    }
    Mat a = imread(argv[1], IMREAD_GRAYSCALE);
    Mat b = imread(argv[2], IMREAD_GRAYSCALE);
    try
    {
        double rmse = computeRMSE(a, b);
        cout << "RMSE: " << rmse << endl;
    }
    catch (const exception &e)
    {
        cerr << e.what() << endl;
        return 1;
    }
    return 0;
}