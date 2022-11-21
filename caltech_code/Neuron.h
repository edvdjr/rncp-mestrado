#ifndef NEURON_H
#define NEURON_H

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <climits>
#include <cmath>
#include <stdexcept>

#include "Matrix.h"
#include "Spike.h"

class Neuron {

public:
    Neuron() {}
    Neuron(int _n_channels, int _k_size_h, int _k_size_w, int _r1, int _r2,
           int _c1, int _c2, int _n_steps, int _i_kernel, double _th,
           double _mu, double _Vb, double _Vr, double _ap, double _am, bool _stoch);

    bool operator < (const Neuron& other) const { return get_spike() < other.get_spike(); }
    bool operator > (const Neuron& other) const { return get_spike() > other.get_spike(); }
    
    void stimulate(Spike& spike, double w, int step);
    void dont_stimulate(int step);
    void reset();
    void fire(int step);
    void stdp(vvvd& w);
    void update_ap(double max_ap) { ap = fmin(2 * ap, max_ap); am = -0.75 * ap; }
    int get_fire_time() const { return f_time; }
    int get_r1() const { return r1; }
    int get_r2() const { return r2; }
    int get_c1() const { return c1; }
    int get_c2() const { return c2; }
    int get_i_kernel() const { return i_kernel; }
    bool check_threshold_at_step(int step);
    bool has_fired() const { return fired; }
    double get_th () const { return th;    }
    double get_mu () const { return mu;    }
    double get_Vb () const { return Vb;    }
    double get_Vr () const { return Vr;    }
    double get_ap () const { return ap;    }
    double get_am () const { return am;    }
    vd get_potentials() const { return V;  }
    Spike get_spike() const { return Spike(V[f_time], i_kernel, r2, c2); }

    bool has_th; // does this neuron have threshold? (useful to get global pooling input)

private:
    
    vd V;                   // potentials vector
    int n_channels;         // number of input channels (from neuronal map)
    int k_size_h, k_size_w; // kernels height & width (from neuronal map)
    int r1, r2, c1, c2;// 1: first row/column of presynaptic neurons (last_row=r1+k_size_h-1)
                            // 2: row/column of this neuron inside the neuronal map
    // r1 and c1 may be negative, since the window covered by this neuron in the presynaptic layer
    // may have its center at the first neuron of the map, thus: r1 = 0 (row)-k_size_h = -k_size_h
    // c1 = 0 (column) - k_size_w = -k_size_w
    int n_steps;            // number of time steps of the simulation
    int i_kernel;           // index of the kernel sheet this neuron belongs to    
    double th, mu, Vb, Vr;  // threshold, decay, base, rest
    double ap, am;          // learning rates (a+, a-)
    bool fired;
    bool stoch;             // is this a stochastic neuron?
    int f_time;             // spike time
    vvvb potentiated;       // presynaptic neurons that cooperated to the spike of this neuron
};

typedef std::vector<Neuron> vN;
typedef std::vector<vN> Neuronal_map;
typedef std::priority_queue<Neuron, vN, std::greater<Neuron>> pqN;

#endif