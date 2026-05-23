#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <fstream>

using namespace std;
using namespace cv;
using namespace std::chrono;

// Calculate RMSE between two images
double calculateRMSE(const Mat& img1, const Mat& img2) {
    if (img1.size() != img2.size()) return -1;
    
    double sum = 0.0;
    int total = img1.rows * img1.cols;
    
    for (int i = 0; i < img1.rows; i++) {
        for (int j = 0; j < img1.cols; j++) {
            double diff = (double)img1.at<uchar>(i, j) - (double)img2.at<uchar>(i, j);
            sum += diff * diff;
        }
    }
    
    return sqrt(sum / total);
}

// Run a command and measure execution time
pair<double, bool> runBenchmark(const string& cmd) {
    auto start = high_resolution_clock::now();
    int result = system(cmd.c_str());
    auto end = high_resolution_clock::now();
    
    double elapsed = duration<double, milli>(end - start).count();
    return {elapsed, result == 0};
}

int main() {
    string outputFile = "benchmark_results.csv";
    ofstream results(outputFile);
    
    // CSV Header
    results << "Implementation,Threads/Procs,Trial,Execution_Time_ms,RMSE,Success\n";
    
    string imagePath = "images/noisy/1024X1024_noise.jpg";
    int trials = 3;
    vector<int> threadCounts = {1, 2, 4, 8};
    
    cout << "Loading reference image: " << imagePath << "\n";
    Mat refImg = imread(imagePath, IMREAD_GRAYSCALE);
    if (refImg.empty()) {
        cerr << "Failed to load reference image\n";
        return 1;
    }
    
    // First run serial to get baseline
    cout << "\n=== SERIAL BASELINE ===\n";
    for (int trial = 0; trial < trials; trial++) {
        string cmd = "./serial/denoise_serial_edge \"" + imagePath + "\" > nul 2>&1";
        auto [elapsed, success] = runBenchmark(cmd);
        
        Mat denoised = imread("serial_denoised.png", IMREAD_GRAYSCALE);
        double rmse = denoised.empty() ? -1 : calculateRMSE(refImg, denoised);
        
        cout << "Trial " << (trial+1) << ": " << fixed << setprecision(2) << elapsed << " ms";
        if (!denoised.empty()) cout << ", RMSE: " << rmse;
        cout << "\n";
        
        results << "Serial,1," << trial << "," << elapsed << "," << rmse << "," << (success ? "YES" : "NO") << "\n";
    }
    
    // OpenMP benchmarks
    cout << "\n=== OPENMP ===\n";
    for (int threads : threadCounts) {
        for (int trial = 0; trial < trials; trial++) {
            string cmd = "./openmp/denoise_openmp_edge \"" + imagePath + "\" " + to_string(threads) + " > nul 2>&1";
            auto [elapsed, success] = runBenchmark(cmd);
            
            Mat denoised = imread("openmp_denoised.png", IMREAD_GRAYSCALE);
            double rmse = denoised.empty() ? -1 : calculateRMSE(refImg, denoised);
            
            cout << "Threads=" << threads << ", Trial " << (trial+1) << ": " << fixed << setprecision(2) << elapsed << " ms";
            if (!denoised.empty()) cout << ", RMSE: " << rmse;
            cout << "\n";
            
            results << "OpenMP," << threads << "," << trial << "," << elapsed << "," << rmse << "," << (success ? "YES" : "NO") << "\n";
        }
    }
    
    // Pthreads benchmarks
    cout << "\n=== PTHREADS ===\n";
    for (int threads : threadCounts) {
        for (int trial = 0; trial < trials; trial++) {
            string cmd = "./pthreads/denoise_pthreads_edge \"" + imagePath + "\" " + to_string(threads) + " > nul 2>&1";
            auto [elapsed, success] = runBenchmark(cmd);
            
            Mat denoised = imread("pthread_denoised.png", IMREAD_GRAYSCALE);
            double rmse = denoised.empty() ? -1 : calculateRMSE(refImg, denoised);
            
            cout << "Threads=" << threads << ", Trial " << (trial+1) << ": " << fixed << setprecision(2) << elapsed << " ms";
            if (!denoised.empty()) cout << ", RMSE: " << rmse;
            cout << "\n";
            
            results << "Pthreads," << threads << "," << trial << "," << elapsed << "," << rmse << "," << (success ? "YES" : "NO") << "\n";
        }
    }
    
    results.close();
    cout << "\n✓ Results saved to " << outputFile << "\n";
    
    return 0;
}
