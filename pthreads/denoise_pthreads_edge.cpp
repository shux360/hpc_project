#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
#include <pthread.h>
#include <vector>

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

struct ThreadData
{
    const Mat *input;
    Mat *output;
    int startRow;
    int endRow;
    bool doSobel;
};

void *worker(void *arg)
{
    ThreadData *data = (ThreadData *)arg;
    const Mat &input = *(data->input);
    Mat &output = *(data->output);
    for (int i = max(1, data->startRow); i < min(input.rows - 1, data->endRow); i++)
    {
        for (int j = 1; j < input.cols - 1; j++)
        {
            if (!data->doSobel)
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
            else
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
    }
    pthread_exit(nullptr);
}

Mat runPthreadStage(const Mat &input, int numThreads, bool sobel)
{
    Mat output = sobel ? Mat::zeros(input.size(), CV_8UC1) : input.clone();
    vector<pthread_t> threads(numThreads);
    vector<ThreadData> threadData(numThreads);
    int rowsPerThread = input.rows / numThreads;
    for (int t = 0; t < numThreads; t++)
    {
        threadData[t].input = &input;
        threadData[t].output = &output;
        threadData[t].startRow = t * rowsPerThread;
        threadData[t].endRow = (t == numThreads - 1) ? input.rows : (t + 1) * rowsPerThread;
        threadData[t].doSobel = sobel;
        pthread_create(&threads[t], nullptr, worker, &threadData[t]);
    }
    for (int t = 0; t < numThreads; t++)
    {
        pthread_join(threads[t], nullptr);
    }
    return output;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        cerr << "Usage: ./pthread_denoise_edge <image_path> <threads>\n";
        return 1;
    }
    string imagePath = argv[1];
    int threads = atoi(argv[2]);
    Mat img = imread(imagePath, IMREAD_GRAYSCALE);
    if (img.empty())
    {
        cerr << "Failed to load image\n";
        return 1;
    }
    auto start = high_resolution_clock::now();
    Mat denoised = runPthreadStage(img, threads, false);
    Mat edges = runPthreadStage(denoised, threads, true);
    auto end = high_resolution_clock::now();
    double elapsed = duration<double, milli>(end - start).count();
    imwrite("pthread_denoised.png", denoised);
    imwrite("pthread_edges.png", edges);
    cout << "Pthreads execution time: " << elapsed << " ms\n";
    cout << "Threads: " << threads << "\n";
    return 0;
}