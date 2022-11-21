#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include <stdexcept>
#include <map>
#include <algorithm>
#include <omp.h>

#include "Neuron.h"
#include "Util.h"
#include "File_handle.h"
#include "N_params.h"

class Convolutional_layer {

public:
    Convolutional_layer();
    Convolutional_layer(int _n_channels, int _in_size_h, int _in_size_w, int _k_size,
                        int _stride, int _n_steps, int _n_kernels, int _inhibit_w, bool _stoch);
    Convolutional_layer(const std::string& desc);

    void config_neurons(double th = 0.0, double mu = 1.0, double Vb = 0.0, double Vr = 0.0,
                        double ap = 0.0, double am = 0.0);
    int get_nm_l() { return n_kernels; }
    int get_nm_h() { return nm_size_h; }
    int get_nm_w() { return nm_size_w; }

    int get_in_size_h () { return in_size_h;    }
    int get_in_size_w () { return in_size_w;    }
    int get_k_size    () { return k_size;       }
    vvvvd get_kernels () const { return kernels; }
    int get_iterations_until_convergence() { return iterations_until_convergence; }
    void increment_iterations_until_convergence() { ++iterations_until_convergence; }
    int get_spikes_until_convergence() { return spikes_until_convergence; }
    vvvd get_potentials () const;
    std::string description() const;
    std::string friendly_description(int num) const;
    std::vector<Neuronal_map> get_maps() const { return neuronal_maps; }
    double get_Cl();
    void update_ap(double max_ap);
    void set_has_th(bool has_th);
    void reset();
    
    Spike_train output(const Spike_train& in_train, int step);
    bool learning; // must neurons learn?

    
protected:
    void set_kernels(vvvvd _kernels) { kernels = _kernels; };
    void get_candidates_to_fire(pqN& candidates_to_fire, const Spike& spike, int nm_index,
                                vvvb& stimulated);

private:
    void split_attributes_from(std::vector<std::string>& vs,
                               std::map<std::string, double>& atts_map,
                               std::string& kernels_line);
    void set_attributes(std::map<std::string, double>& atts_map);
    void set_kernels(const std::string& line);
    void set_neurons(std::map<std::string, double>& atts_map);
    void receive_spike(const Spike& spike, vvvb& stimulated, vNP& winners, int step);
    void competition(const vNP& winners, Spike_train& out, int step);
    void update_idle_neurons(const vvvb& stimulated, int step);
    void stimulate(Neuron& neuron, pqN& candidates_to_fire, const Spike& spike);
    Spike_train propagate_spike(const Spike& spike, vvvb& stimulated);
    std::string attributes_to_string() const;
    std::string kernels_to_string() const;
    std::string neurons_to_string() const;
    std::string friendly_attributes_to_string() const;
    std::string friendly_kernels_to_string() const;
    std::string friendly_neurons_to_string() const;
    
    // Constructor parameters
    int n_channels;           // number of input channels
    int in_size_h, in_size_w; // input size (height & width)
    int k_size;               // kernel size (height & width)
    int stride;
    int n_steps;              // number of time steps of the simulation
    int n_kernels;            // number of different kernels (= number of feature maps)
    int inhibit_w;            // inhibit window
    bool stoch;               // is this a stochastic layer?

    // Other attributes
    std::vector<Neuronal_map> neuronal_maps; // sheets of neurons
    int nm_size_h, nm_size_w; // height & width of neuronal maps
    vvvvd kernels;            // vector of kernels (synaptic weights)
    vvvb allowed_to_fire;     // neurons allowed to fire
    vb STDPed_maps;           // which maps had at least one neuron that has fired
    int iterations_until_convergence;
    int spikes_until_convergence;
};

#endif
