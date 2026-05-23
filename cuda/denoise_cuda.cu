#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;
using namespace std::chrono;

__constant__ float d_gauss[9] = {
    1/16.0f, 2/16.0f, 1/16.0f,
    2/16.0f, 4/16.0f, 2/16.0f,
    1/16.0f, 2/16.0f, 1/16.0f
};

__constant__ int d_sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ int d_sobel_y[9] = {-1, -2, -1,  0, 0, 0,  1,  2,  1};

__global__ void gaussianBlurKernel(const unsigned char* input, unsigned char* output, int rows, int cols)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= rows || j >= cols) return;

    if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
        output[i * cols + j] = input[i * cols + j];
        return;
    }

    float sum = 0.0f;
    for (int ki = -1; ki <= 1; ki++) {
        for (int kj = -1; kj <= 1; kj++) {
            sum += input[(i + ki) * cols + (j + kj)] * d_gauss[(ki + 1) * 3 + (kj + 1)];
        }
    }
    output[i * cols + j] = (unsigned char)min(max((int)sum, 0), 255);
}

__global__ void sobelEdgeKernel(const unsigned char* input, unsigned char* output, int rows, int cols)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= rows || j >= cols) return;

    if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
        output[i * cols + j] = 0;
        return;
    }

    int gx = 0, gy = 0;
    for (int ki = -1; ki <= 1; ki++) {
        for (int kj = -1; kj <= 1; kj++) {
            unsigned char pixel = input[(i + ki) * cols + (j + kj)];
            gx += pixel * d_sobel_x[(ki + 1) * 3 + (kj + 1)];
            gy += pixel * d_sobel_y[(ki + 1) * 3 + (kj + 1)];
        }
    }
    int mag = (int)sqrtf((float)(gx * gx + gy * gy));
    output[i * cols + j] = (unsigned char)min(mag, 255);
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        cerr << "Usage: ./cuda_denoise_edge <image_path> [block_size]" << endl;
        cerr << "  block_size: thread block dimension (8, 16, or 32). Default: 16" << endl;
        return 1;
    }

    string imagePath = argv[1];
    int bs = (argc >= 3) ? atoi(argv[2]) : 16;

    // Clamp to valid CUDA block sizes for 2D kernels (max 1024 threads/block)
    if (bs != 8 && bs != 16 && bs != 32) {
        cerr << "Invalid block size " << bs << ". Using default 16." << endl;
        bs = 16;
    }

    Mat img = imread(imagePath, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Failed to load image" << endl;
        return 1;
    }

    int rows = img.rows;
    int cols = img.cols;
    size_t imgSize = (size_t)rows * cols * sizeof(unsigned char);

    unsigned char *d_input, *d_denoised, *d_edges;
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_denoised, imgSize);
    cudaMalloc(&d_edges, imgSize);

    cudaMemcpy(d_input, img.data, imgSize, cudaMemcpyHostToDevice);

    dim3 blockSize(bs, bs);
    dim3 gridSize((cols + bs - 1) / bs, (rows + bs - 1) / bs);

    auto start = high_resolution_clock::now();

    gaussianBlurKernel<<<gridSize, blockSize>>>(d_input, d_denoised, rows, cols);
    cudaDeviceSynchronize();

    sobelEdgeKernel<<<gridSize, blockSize>>>(d_denoised, d_edges, rows, cols);
    cudaDeviceSynchronize();

    auto end = high_resolution_clock::now();
    double elapsed = duration<double, milli>(end - start).count();

    Mat denoised(rows, cols, CV_8UC1);
    Mat edges(rows, cols, CV_8UC1);
    cudaMemcpy(denoised.data, d_denoised, imgSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(edges.data, d_edges, imgSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_denoised);
    cudaFree(d_edges);

    imwrite("cuda_denoised.png", denoised);
    imwrite("cuda_edges.png", edges);

    cout << "CUDA execution time: " << elapsed << " ms" << endl;
    cout << "Block size: " << bs << "x" << bs
         << " (" << bs * bs << " threads/block)" << endl;
    cout << "Grid size: " << gridSize.x << "x" << gridSize.y
         << " (" << gridSize.x * gridSize.y << " blocks)" << endl;
    cout << "Image size: " << cols << "x" << rows << endl;

    return 0;
}
