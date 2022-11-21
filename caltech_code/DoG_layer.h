#ifndef DOG_LAYER_H
#define DOG_LAYER_H

#include <iostream>
#include <cmath>
#include <cfloat>
#include <map>
#include <algorithm>

#include "Matrix.h"
#include "Spike.h"
#include "Util.h"

class DoG_layer {

public:
    DoG_layer() {}
    DoG_layer(int _in_size_h, int _in_size_w, int _k_size, int _stride, unsigned _n_steps,
              double _s1, double _s2, double _p_th, double _n_th);
    DoG_layer(std::string desc);
    Spike_train_list output(const vvi& input);
    std::string description() const;
    std::string friendly_description() const;
    int get_nm_l() const { return 1; }
    int get_nm_h() const { return (in_size_h - k_size) / stride + 1; }
    int get_nm_w() const { return (in_size_w - k_size) / stride + 1; }
    unsigned get_n_steps() const { return n_steps; }

private:
    double get_gauss_dist(int x, int y, double sigma);
    void create_DoG_filter();
    void set_attributes(std::map<std::string, double> atts_map);
    std::string friendly_kernel_to_string() const;
    Spike_train_list get_layer_output_from_train(Spike_train& train, unsigned n_steps);

    vvd kernel;                // DoG filter
    int in_size_h, in_size_w;  // input sizes (height & width)
    int k_size;                // kernel size (height & width)
    int stride;
    unsigned n_steps;          // number of time steps of the simulation
    double s1, s2, p_th, n_th; // sigma 1 & sigma 2 & positive & negative thresholds
};

#endif