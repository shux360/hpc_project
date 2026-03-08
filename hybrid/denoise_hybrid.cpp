#include <opencv2/opencv.hpp>
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

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

void gaussianLocalOMP(const Mat &input, Mat &output, int threads)
{
#pragma omp parallel for num_threads(threads) schedule(static)
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
}

void sobelLocalOMP(const Mat &input, Mat &output, int threads)
{
#pragma omp parallel for num_threads(threads) schedule(static)
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
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc < 3)
    {
        if (rank == 0)
            cerr << "Usage: mpirun -np <p> ./hybrid_denoise_edge <image_path> <omp_threads>\n";
        MPI_Finalize();
        return 1;
    }
    int ompThreads = atoi(argv[2]);
    Mat fullImg;
    int rows, cols;
    if (rank == 0)
    {
        fullImg = imread(argv[1], IMREAD_GRAYSCALE);
        if (fullImg.empty())
        {
            cerr << "Failed to load image\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        rows = fullImg.rows;
        cols = fullImg.cols;
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int localRows = rows / size;
    int remainder = rows % size;
    int myRows = localRows + (rank < remainder ? 1 : 0);
    vector<int> sendcounts(size), displs(size);
    if (rank == 0)
    {
        int offset = 0;
        for (int i = 0; i < size; i++)
        {
            int r = localRows + (i < remainder ? 1 : 0);
            sendcounts[i] = r * cols;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }
    Mat localImg(myRows, cols, CV_8UC1);
    MPI_Scatterv(rank == 0 ? fullImg.data : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR,
                 localImg.data, myRows * cols, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    double start = MPI_Wtime();
    Mat localDenoised = localImg.clone();
    gaussianLocalOMP(localImg, localDenoised, ompThreads);
    Mat localEdges = Mat::zeros(localDenoised.size(), CV_8UC1);
    sobelLocalOMP(localDenoised, localEdges, ompThreads);
    double end = MPI_Wtime();
    double localTime = (end - start) * 1000.0;
    double maxTime;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    Mat finalDenoised, finalEdges;
    if (rank == 0)
    {
        finalDenoised = Mat(rows, cols, CV_8UC1);
        finalEdges = Mat(rows, cols, CV_8UC1);
    }
    MPI_Gatherv(localDenoised.data, myRows * cols, MPI_UNSIGNED_CHAR,
                rank == 0 ? finalDenoised.data : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(localEdges.data, myRows * cols, MPI_UNSIGNED_CHAR,
                rank == 0 ? finalEdges.data : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        imwrite("hybrid_denoised.png", finalDenoised);
        imwrite("hybrid_edges.png", finalEdges);
        cout << "Hybrid MPI+OpenMP execution time: " << maxTime << " ms\n";
        cout << "Processes: " << size << ", OpenMP threads/process: " << ompThreads << "\n";
    }
    MPI_Finalize();
    return 0;
}