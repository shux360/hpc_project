#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

using namespace std;
using namespace cv;

struct ComparisonResult {
    double rmse;
    double mae;
    double maxDiff;
    double minDiff;
    int differentPixels;
    int totalPixels;
    double similarityPercent;
    bool valid;
};

ComparisonResult compareImages(const Mat &img1, const Mat &img2, const string &name) {
    ComparisonResult result = {0, 0, 0, 0, 0, 0, 0, false};
    
    if (img1.empty() || img2.empty()) {
        cerr << "[ERROR] " << name << ": One or both images are empty\n";
        return result;
    }
    
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        cerr << "[ERROR] " << name << ": Images have different dimensions or types\n";
        cout << "  Serial: " << img1.size() << " Type: " << img1.type() << "\n";
        cout << "  Parallel: " << img2.size() << " Type: " << img2.type() << "\n";
        return result;
    }
    
    result.totalPixels = img1.rows * img1.cols;
    double sumSquaredDiff = 0;
    double sumAbsDiff = 0;
    result.minDiff = 255;
    
    for (int i = 0; i < img1.rows; i++) {
        for (int j = 0; j < img1.cols; j++) {
            uchar p1 = img1.at<uchar>(i, j);
            uchar p2 = img2.at<uchar>(i, j);
            
            double diff = abs((int)p1 - (int)p2);
            sumSquaredDiff += diff * diff;
            sumAbsDiff += diff;
            
            result.maxDiff = max(result.maxDiff, diff);
            result.minDiff = min(result.minDiff, diff);
            
            if (diff != 0) {
                result.differentPixels++;
            }
        }
    }
    
    result.rmse = sqrt(sumSquaredDiff / result.totalPixels);
    result.mae = sumAbsDiff / result.totalPixels;
    result.similarityPercent = ((result.totalPixels - result.differentPixels) * 100.0) / result.totalPixels;
    result.valid = true;
    
    return result;
}

void printComparison(const ComparisonResult &result, const string &methodName) {
    cout << "\n" << string(70, '=') << "\n";
    cout << "METHOD: " << methodName << "\n";
    cout << string(70, '=') << "\n";
    
    cout << fixed << setprecision(6);
    cout << "  RMSE (Root Mean Square Error): " << result.rmse << "\n";
    cout << "  MAE  (Mean Absolute Error):    " << result.mae << "\n";
    cout << "  Max Difference:                " << result.maxDiff << "\n";
    cout << "  Min Difference:                " << result.minDiff << "\n";
    cout << "  Different Pixels:              " << result.differentPixels << " / " << result.totalPixels << "\n";
    cout << fixed << setprecision(4);
    cout << "  Similarity to Serial:          " << result.similarityPercent << "%\n";
    
    // Status interpretation
    cout << "\n  Status: ";
    if (!result.valid) {
        cout << "FAILED (comparison could not be completed)\n";
    } else if (result.rmse < 0.001 && result.differentPixels == 0) {
        cout << "PERFECT MATCH (Identical to serial)\n";
    } else if (result.rmse < 1.0 && result.similarityPercent > 99.99) {
        cout << "EXCELLENT (Negligible difference)\n";
    } else if (result.rmse < 1.0 && result.similarityPercent > 99.0) {
        cout << "ACCEPTABLE (Minor rounding differences)\n";
    } else {
        cout << "FAILED (Significant difference)\n";
    }
}

int main(int argc, char **argv) {
    string basePath = (argc >= 2) ? argv[1] : "./";
    if (!basePath.empty() && basePath.back() != '/' && basePath.back() != '\\') {
        basePath += "/";
    }
    
    cout << "\n" << string(70, '=') << "\n";
    cout << "HPC IMAGE PROCESSING - ACCURACY COMPARISON TOOL\n";
    cout << "Comparing all parallel implementations against serial baseline\n";
    cout << string(70, '=') << "\n";
    
    // Load serial baseline (reference)
    Mat serialDenoised = imread(basePath + "serial_denoised.png", IMREAD_GRAYSCALE);
    Mat serialEdges = imread(basePath + "serial_edges.png", IMREAD_GRAYSCALE);
    
    if (serialDenoised.empty() || serialEdges.empty()) {
        basePath = "../";
        serialDenoised = imread(basePath + "serial_denoised.png", IMREAD_GRAYSCALE);
        serialEdges = imread(basePath + "serial_edges.png", IMREAD_GRAYSCALE);
        if (serialDenoised.empty() || serialEdges.empty()) {
            cerr << "[ERROR] Serial baseline files not found\n";
            cerr << "  Expected: serial_denoised.png, serial_edges.png in ./ or ../\n";
            cerr << "  Usage: ./compare_accuracy [project_output_path]\n";
            return 1;
        }
    }
    
    cout << "\n[INFO] Serial baseline loaded\n";
    cout << "  Denoised: " << serialDenoised.size() << " pixels\n";
    cout << "  Edges: " << serialEdges.size() << " pixels\n";
    
    vector<pair<string, pair<Mat, Mat>>> implementations = {
        {"OpenMP", {imread(basePath + "openmp_denoised.png", IMREAD_GRAYSCALE), 
                    imread(basePath + "openmp_edges.png", IMREAD_GRAYSCALE)}},
        {"Pthreads", {imread(basePath + "pthread_denoised.png", IMREAD_GRAYSCALE), 
                      imread(basePath + "pthread_edges.png", IMREAD_GRAYSCALE)}},
        {"MPI", {imread(basePath + "mpi_denoised.png", IMREAD_GRAYSCALE), 
                 imread(basePath + "mpi_edges.png", IMREAD_GRAYSCALE)}},
        {"Hybrid", {imread(basePath + "hybrid_denoised.png", IMREAD_GRAYSCALE), 
                    imread(basePath + "hybrid_edges.png", IMREAD_GRAYSCALE)}}
    };
    
    cout << "\n" << string(70, '=') << "\n";
    cout << "DENOISING STAGE COMPARISON (Gaussian Blur)\n";
    cout << string(70, '=') << "\n";
    
    vector<pair<string, ComparisonResult>> denoisedResults;
    for (const auto &impl : implementations) {
        if (impl.second.first.empty()) {
            cerr << "[WARNING] " << impl.first << " denoised image not found\n";
            continue;
        }
        ComparisonResult result = compareImages(serialDenoised, impl.second.first, impl.first);
        printComparison(result, impl.first + " (Denoised)");
        if (result.valid) {
            denoisedResults.push_back({impl.first, result});
        }
    }
    
    cout << "\n" << string(70, '=') << "\n";
    cout << "EDGE DETECTION STAGE COMPARISON (Sobel Operator)\n";
    cout << string(70, '=') << "\n";
    
    vector<pair<string, ComparisonResult>> edgeResults;
    for (const auto &impl : implementations) {
        if (impl.second.second.empty()) {
            cerr << "[WARNING] " << impl.first << " edges image not found\n";
            continue;
        }
        ComparisonResult result = compareImages(serialEdges, impl.second.second, impl.first);
        printComparison(result, impl.first + " (Edges)");
        if (result.valid) {
            edgeResults.push_back({impl.first, result});
        }
    }
    
    // Summary table
    cout << "\n" << string(70, '=') << "\n";
    cout << "SUMMARY TABLE - DENOISING\n";
    cout << string(70, '=') << "\n";
    cout << left << setw(15) << "Method" << setw(12) << "RMSE" << setw(12) << "MAE" 
         << setw(15) << "Match %" << setw(15) << "Status\n";
    cout << string(70, '-') << "\n";
    
    for (const auto &entry : denoisedResults) {
        const ComparisonResult &result = entry.second;
        cout << left << setw(15) << entry.first;
        cout << fixed << setprecision(6) << setw(12) << result.rmse;
        cout << fixed << setprecision(6) << setw(12) << result.mae;
        cout << fixed << setprecision(2) << setw(15) << result.similarityPercent << "%";
        
        if (result.rmse < 0.001) {
            cout << "Perfect\n";
        } else if (result.rmse < 1.0) {
            cout << "Excellent\n";
        } else {
            cout << "Check\n";
        }
    }
    
    cout << "\n" << string(70, '=') << "\n";
    cout << "SUMMARY TABLE - EDGE DETECTION\n";
    cout << string(70, '=') << "\n";
    cout << left << setw(15) << "Method" << setw(12) << "RMSE" << setw(12) << "MAE" 
         << setw(15) << "Match %" << setw(15) << "Status\n";
    cout << string(70, '-') << "\n";
    
    for (const auto &entry : edgeResults) {
        const ComparisonResult &result = entry.second;
        cout << left << setw(15) << entry.first;
        cout << fixed << setprecision(6) << setw(12) << result.rmse;
        cout << fixed << setprecision(6) << setw(12) << result.mae;
        cout << fixed << setprecision(2) << setw(15) << result.similarityPercent << "%";
        
        if (result.rmse < 0.001) {
            cout << "Perfect\n";
        } else if (result.rmse < 1.0) {
            cout << "Excellent\n";
        } else {
            cout << "Check\n";
        }
    }
    
    cout << "\n" << string(70, '=') << "\n";
    cout << "CONCLUSION\n";
    cout << string(70, '=') << "\n";
    
    bool allPerfect = true;
    for (const auto &entry : edgeResults) {
        const ComparisonResult &result = entry.second;
        if (result.rmse >= 0.001 || result.differentPixels > 0) {
            allPerfect = false;
            break;
        }
    }
    
    if (allPerfect) {
        cout << "ALL IMPLEMENTATIONS MATCH SERIAL PERFECTLY\n";
        cout << "   All parallel versions produce identical edge detection results.\n";
    } else {
        cout << "SOME IMPLEMENTATIONS SHOW DIFFERENCES\n";
        cout << "   Review results above for details.\n";
    }
    
    cout << "\n";
    return 0;
}
