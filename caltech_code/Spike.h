#ifndef SPIKE_H
#define SPIKE_H

#include <queue>
#include <vector>
#include <cstdio>

struct Spike {
    
    double a;     // activation
    int ch, r, c; // channel, row, column

    Spike () : a(0), ch(0), r(0), c(0) {}

    Spike (double _a, int _ch, int _r, int _c) {
        a = _a, ch = _ch, r = _r, c = _c;
    }

    bool operator<(const Spike& other) const {
        return a < other.a;
    }

    bool operator>(const Spike& other) const {
        return a > other.a;
    }

    void print() {
        printf("a = %2.2lf; ch = %3d; r = %3d; c = %3d\n", a, ch, r, c);
    }
};

typedef std::vector<Spike> Spike_train;
typedef std::vector<Spike_train> Spike_train_list;

#endif