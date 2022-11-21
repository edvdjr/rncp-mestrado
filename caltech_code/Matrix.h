#ifndef MATRIX_H
#define MATRIX_H
 
#include <vector>
#include <chrono>
#include <random>
#include <cfloat>

typedef std::vector<int> vi;
typedef std::vector<vi> vvi;
typedef std::vector<vvi> vvvi;
typedef std::vector<double> vd;
typedef std::vector<vd> vvd;
typedef std::vector<vvd> vvvd;
typedef std::vector<vvvd> vvvvd;
typedef std::vector<bool> vb;
typedef std::vector<vb> vvb;
typedef std::vector<vvb> vvvb;

double dot_product(const vvi& window, const vvd& kernel, int w_row, int w_col);
vvd convolution(const vvi& window, const vvd& kernel, int stride = 1);
vvvd get_random_3D_vector(int d_size, int w_size, int h_size, double m = 0.8, double s = 0.05);
double get_mean(const vvd& v);
double get_max(const vvd& v);
void multiply_by_scalar(vvd& v, double s);
void add_scalar(vvd& v, double s);
std::string vd_to_str(const vd& v);
void print(const vi& v);
void print(const vvi& v);
void print(const vd& v);
void print(const vvd& v);
void print(const vvvd& v);

#endif