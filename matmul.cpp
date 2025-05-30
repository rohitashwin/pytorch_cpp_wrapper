#include "matmul.hpp"
#include <iostream>

void matmul_impl(float *a, float *b, float *c, int m, int n, int k) {
    static bool initialized = false;
    // if (!initialized) {
        std::cout << "\033[1;32mmatmul called from cpp\033[0m" << std::endl;
    //     initialized = true;
    // }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            c[i * n + j] = 0.0f;
            for (int l = 0; l < k; ++l) {
                c[i * n + j] += a[i * k + l] * b[l * n + j];
            }
        }
    }
}