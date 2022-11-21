#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include <stdexcept>
#include <vector>
#include <map>

#include "Neuron.h"
#include "Util.h"

class Pooling_layer {

public:
    Pooling_layer();
    Pooling_layer(int _n_channels, int _in_size_h, int _in_size_w, 
                  int _k_size, int _stride, int _n_steps, bool _stoch);
    Pooling_layer(const std::string& desc);

    int get_nm_l() const { return n_channels; }
    int get_nm_h() const { return nm_size_h;  }
    int get_nm_w() const { return nm_size_w;  }
    
    vvvd get_potentials() const;
    std::string description() const;
    std::string friendly_description(int num) const;
    Spike_train output(const Spike_train& in_train, int step);
    void reset();

    static vd global_pooling(std::vector<Neuronal_map>& neuronal_maps);

private:
    void config_neurons();
    void split_attributes_from(std::vector<std::string>& vs,
                               std::map<std::string, double>& atts_map,
                               std::string& kernel_line);
    void set_attributes(std::map<std::string, double>& atts_map);
    void set_neurons(std::map<std::string, double>& atts_map);
    std::string friendly_attributes_to_string() const;
    std::string friendly_kernels_to_string() const;
    std::string friendly_neurons_to_string() const;
    
    // Constructor parameters
    int n_channels;           // number of input channels
    int in_size_h, in_size_w; // input height & width
    int k_size;               // kernels height & width
    int stride;
    int n_steps;              // number of time steps of the simulation
    bool stoch;               // is this a stochastic layer?

    // Other attributes
    std::vector<Neuronal_map> neuronal_maps; // sheets of neurons
    int nm_size_h, nm_size_w; // height&  width of neuronal maps
};

#endif