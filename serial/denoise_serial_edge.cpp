#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cmath>

using namespace std;
using namespace cv;
using namespace std::chrono;

static const float GAUSS[3][3] = {
    {1 / 16.0f, 2 / 16.0f, 1 / 16.0f},
    {2 / 16.0f, 4 / 16.0f, 2 / 16.0f},
    {1 / 16.0f, 2 / 16.0f, 1 / 16.0f}};

static const int SOBEL_X[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}};

static const int SOBEL_Y[3][3] = {
    {-1, -2, -1},
    {0, 0, 0},
    {1, 2, 1}};

Mat gaussianBlurSerial(const Mat &input)
{
    Mat output = input.clone();
    for (int i = 1; i < input.rows - 1; i++)
    {
        for (int j = 1; j < input.cols - 1; j++)
        {
            float sum = 0.0f;
            for (int ki = -1; ki <= 1; ki++)
            {
                for (int kj = -1; kj <= 1; kj++)
                {
                    sum += input.at<uchar>(i + ki, j + kj) * GAUSS[ki + 1][kj + 1];
                }
            }
            output.at<uchar>(i, j) = static_cast<uchar>(sum);
        }
    }
    return output;
}

Mat sobelEdgeSerial(const Mat &input)
{
    Mat output = Mat::zeros(input.size(), CV_8UC1);
    for (int i = 1; i < input.rows - 1; i++)
    {
        for (int j = 1; j < input.cols - 1; j++)
        {
            int gx = 0, gy = 0;
            for (int ki = -1; ki <= 1; ki++)
            {
                for (int kj = -1; kj <= 1; kj++)
                {
                    uchar pixel = input.at<uchar>(i + ki, j + kj);
                    gx += pixel * SOBEL_X[ki + 1][kj + 1];
                    gy += pixel * SOBEL_Y[ki + 1][kj + 1];
                }
            }
            int mag = static_cast<int>(sqrt(gx * gx + gy * gy));
            output.at<uchar>(i, j) = saturate_cast<uchar>(mag);
        }
    }
    return output;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        cerr << "Usage: ./serial_denoise_edge <image_path>\n";
        return 1;
    }

    Mat img = imread(argv[1], IMREAD_GRAYSCALE);
    if (img.empty())
    {
        cerr << "Failed to load image\n";
        return 1;
    }

    auto start = high_resolution_clock::now();

    Mat denoised = gaussianBlurSerial(img);
    Mat edges = sobelEdgeSerial(denoised);

    auto end = high_resolution_clock::now();
    double elapsed = duration<double, milli>(end - start).count();

    imwrite("serial_denoised.png", denoised);
    imwrite("serial_edges.png", edges);

    cout << "Serial execution time: " << elapsed << " ms\n";
    return 0;
}