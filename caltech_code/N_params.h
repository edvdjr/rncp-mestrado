#ifndef N_PARAMS_H
#define N_PARAMS_H

#include <vector>

struct N_params {
    
    int k, r2, c2; // kernel, row, col
    double v;      // potential

    N_params(int _k = -1, int _r = -1, int _c = -1, double _v = -1) :
            k(_k), r2(_r), c2(_c), v(_v) {}

    bool operator<(const N_params& o) const { return v < o.v; }

    bool operator>(const N_params& o) const { return v > o.v; }

    void init(int _k, int _r, int _c, double _v) {
        k = _k, r2 = _r, c2 = _c, v = _v;
    }

    void init(const N_params &o) { k=o.k, r2=o.r2, c2=o.c2, v=o.v; }

    void print() { printf("\nk = %2d; r2 = %2d; c2 = %2d; V = %lf\n", k, r2, c2, v); }
};

typedef std::vector <N_params> vNP;

#endif