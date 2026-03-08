// metrics.c
#include "metrics.h"
#include <math.h>

double rmse_u8(const Image* a, const Image* b) {
  const int n = a->width * a->height;
  double sse = 0.0;
  for (int i=0; i<n; i++) {
    double d = (double)a->data[i] - (double)b->data[i];
    sse += d*d;
  }
  return sqrt(sse / (double)n);
}
