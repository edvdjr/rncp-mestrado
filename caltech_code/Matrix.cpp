#include "Matrix.h"

double dot_product(const vvi& window, const vvd& kernel, int w_row, int w_col) {

    double res = 0;
    for (int i = 0; i < static_cast<int>(kernel.size()); ++i)
        for (int j = 0; j < static_cast<int>(kernel[0].size()); ++j)
            res += kernel[i][j] * window[i + w_row][j + w_col];
    return res;
}

vvd convolution(const vvi& input, const vvd& kernel, int stride) {

    // Assume no padding
    int in_size_h = input.size(), in_size_w = input[0].size();
    int k_size_h = kernel.size(), k_size_w = kernel[0].size();
    int conv_size_w = 1 + ceil(1.0 * (in_size_w - k_size_w) / stride);
    int conv_size_h = 1 + ceil(1.0 * (in_size_h - k_size_h) / stride);
    
    vvd conv(conv_size_h, vd(conv_size_w));
    for (int i = 0; i < conv_size_h; ++i)
        for (int j = 0; j < conv_size_w; ++j)
            conv[i][j] = dot_product(input, kernel, i, j);
    return conv;
}

vvvd get_random_3D_vector(int d_size, int h_size, int w_size, double m, double s) {

    std::default_random_engine generator;
    long long seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed);
    std::normal_distribution<double> distribution (m, s);
    vvvd t(d_size, vvd(h_size, vd(w_size)));
    for (int i = 0; i < d_size; ++i)
        for (int j = 0; j < h_size; ++j)
            for (int k = 0; k < w_size; ++k) {
                double rdm = fmax(0.01, distribution(generator));
                t[i][j][k] = fmin(0.99, rdm);
            }
    return t;
}

double get_mean(const vvd& v) {

    double sum {0};
    int tot {0};
    for (auto l : v) for (auto c : l) { ++tot, sum += c; }
    return sum / tot;
}

double get_max(const vvd& v) {
    
    double M {-DBL_MAX};
    for (auto l : v) for (auto c : l) { M = fmax(M, c); }
    return M;
}

void multiply_by_scalar(vvd& v, double s) { for (auto& l : v) for (auto& c : l) { c *= s; } }
void add_scalar(vvd& v, double s) { for (auto& l : v) for (auto& c : l) { c += s; } }

// .csv format
std::string vd_to_str(const vd& v) {

    std::string line;
    for (int i = 1; i <= static_cast<int>(v.size()); ++i) {
        if (i > 1) line += ",";
        line += std::to_string(v[i - 1]);
    }
    return line;
}

void print(const vi& v) {
    
    for (int i = 0; i < static_cast<int>(v.size()); ++i) {
        if (i) printf(" ");
        printf("%3d", v[i]);
    }
}

void print(const vvi& v) {
    
    printf("\n");
    for (int i = 0; i < static_cast<int>(v.size()); ++i) {
        print(v[i]);
        printf("\n");
    }
}

void print(const vd& v) {
    
    for (int i = 0; i < static_cast<int>(v.size()); ++i) {
        if (i) printf(" ");
        printf("%5.3f", v[i]);
    }
}

void print(const vvd& v) {
    
    printf("\n");
    for (int i = 0; i < static_cast<int>(v.size()); ++i) {
        print(v[i]);
        printf("\n");
    }
}

void print(const vvvd& v) {
    
    for (int i = 0; i < static_cast<int>(v.size()); ++i) {
        printf("\n%d:\n", i);
        print(v[i]);
    }
}