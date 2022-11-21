#ifndef NETWORK_H
#define NETWORK_H

#include "DoG_layer.h"
#include "Convolutional_layer.h"
#include "Pooling_layer.h"
#include "File_handle.h"

class Network {

public:
    Network();
    Network(const std::string& file_name);

    void unsupervised_learning(const vvvi& train_dataset, const vi& train_labels,
                               std::string& out_str, int train_n_imgs);
    void create_train_file(const std::string& file_name);
    void unsupervised_testing(const vvvi& train_dataset, const vi& train_labels,
                              const vvvi& test_dataset, const vi& test_labels,
                              std::string& train_out_str, std::string& test_out_str,
                              int train_n_imgs, int test_n_imgs);
    void create_test_file(const std::string& file_name);
    void save(const std::string& file_name);
    std::string description() const;
    std::string friendly_description() const;
    bool has_layer(int l) { return l < static_cast<int>(conv_layers.size()); }
    bool has_kernel(int l, int o) { return has_layer(l) && o < conv_layers[l].get_nm_l(); }
    vvvd get_kernel(int l, int o) const { return (conv_layers[l].get_kernels())[o]; }
    vvvd get_potentials(int l) const { return conv_layers[l].get_potentials(); }
    vvvvd get_all_potentials() const;
    int get_total_time() { return total_time; }
    
    static int img_input_width;
    static int img_input_height;

private:
    template<class T>
    void get_output_dimensions(T layer, int& p_output_w, int& p_output_h, int& p_output_d);
    void forward_phase(const vvvi& train_dataset, int train_n_imgs);
    void forward(Spike_train& spike_input, int step);
    bool run_steps(const vvi& img, bool is_learning);
    void global_pooling_phase(const vvvi& dataset, const vi& labels,
                              std::string& out_str, int n_imgs);
    std::vector<std::string> get_vector_str_layers(const std::string& file_name);

    int n_steps, learning_layer;
    double total_time, max_ap;
    DoG_layer dog_layer;
    std::vector<Convolutional_layer> conv_layers;
    std::vector<Pooling_layer> pool_layers;
    int max_dog_spikes, min_dog_spikes, update_ap_step;
    double mean_dog_spikes;
};

#endif