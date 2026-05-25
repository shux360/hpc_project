#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;
using namespace cv;

struct Metrics
{
    double rmse;
    double mae;
    double maxDiff;
    int differentPixels;
    int totalPixels;
};

Metrics compareImages(const Mat &a, const Mat &b)
{
    if (a.empty() || b.empty() || a.size() != b.size() || a.type() != b.type())
    {
        throw runtime_error("Images must have same size and type");
    }

    Metrics metrics = {0.0, 0.0, 0.0, 0, a.rows * a.cols};
    double sumSquaredDiff = 0.0;
    double sumAbsDiff = 0.0;

    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < a.cols; j++)
        {
            double diff = abs((int)a.at<uchar>(i, j) - (int)b.at<uchar>(i, j));
            sumSquaredDiff += diff * diff;
            sumAbsDiff += diff;
            metrics.maxDiff = max(metrics.maxDiff, diff);
            if (diff != 0)
            {
                metrics.differentPixels++;
            }
        }
    }

    metrics.rmse = sqrt(sumSquaredDiff / metrics.totalPixels);
    metrics.mae = sumAbsDiff / metrics.totalPixels;
    return metrics;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        cerr << "Usage: ./rmse_compare <serial_image> <parallel_image> [rmse_threshold]\n";
        return 1;
    }

    double threshold = (argc >= 4) ? atof(argv[3]) : 0.5;
    Mat a = imread(argv[1], IMREAD_GRAYSCALE);
    Mat b = imread(argv[2], IMREAD_GRAYSCALE);

    try
    {
        Metrics metrics = compareImages(a, b);
        double similarity = ((metrics.totalPixels - metrics.differentPixels) * 100.0) / metrics.totalPixels;

        cout << fixed << setprecision(6);
        cout << "RMSE: " << metrics.rmse << "\n";
        cout << "MAE: " << metrics.mae << "\n";
        cout << "Max difference: " << metrics.maxDiff << "\n";
        cout << "Different pixels: " << metrics.differentPixels << " / " << metrics.totalPixels << "\n";
        cout << setprecision(4) << "Similarity: " << similarity << "%\n";
        cout << "Status: " << (metrics.rmse <= threshold ? "PASS" : "FAIL") << "\n";

        if (metrics.rmse > threshold)
        {
            return 2;
        }
    }
    catch (const exception &e)
    {
        cerr << e.what() << endl;
        return 1;
    }
    return 0;
}
