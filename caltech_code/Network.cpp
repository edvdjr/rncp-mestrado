#include "Network.h"

const std::string NET_NAME = "Network";
const double CONVERGENCE {0.01};

Network::Network() {

    n_steps = 30, learning_layer = 0, total_time = 0, max_ap = 0.15, update_ap_step = 1000;
    int p_output_d, p_output_h, p_output_w; // Output dimensions

    int dog_window_size = 7, dog_stride = 1;
    double dog_p_th = 15.0, dog_n_th = -40, ap_value = 0.0005;
    vd ap(3, ap_value), am(3, 0.0), conv_th = {1.0, 60.0, 2.0};
    vi conv_n_maps = {4, 20, 10}, conv_window_size = {5, 16, 5}, conv_stride = {1, 1, 1};
    vi pool_window_size = {7, 2}, pool_stride = {6, 2};
    for (unsigned i = 0; i < ap.size(); ++i) { am[i] = -0.75 * ap[i]; }

    conv_layers.clear();
    pool_layers.clear();

    dog_layer = DoG_layer(img_input_height, img_input_width, dog_window_size, dog_stride,
                          n_steps, 1.0, 2.0, dog_p_th, dog_n_th);
    get_output_dimensions(dog_layer, p_output_d, p_output_h, p_output_w);
    
    conv_layers.push_back(Convolutional_layer(p_output_d, p_output_h, p_output_w,
                                              conv_window_size[0], conv_stride[0],
                                              n_steps, conv_n_maps[0], 1, false));
    conv_layers.back().config_neurons(conv_th[0], 1.0, 0.0, 0.0, ap[0], am[0]);
    get_output_dimensions(conv_layers.back(), p_output_d, p_output_h, p_output_w);
    
    pool_layers.push_back(Pooling_layer(p_output_d, p_output_h, p_output_w,
                                        pool_window_size[0], pool_stride[0], n_steps, false));
    get_output_dimensions(pool_layers.back(), p_output_d, p_output_h, p_output_w);
    
    conv_layers.push_back(Convolutional_layer(p_output_d, p_output_h, p_output_w,
                                              conv_window_size[1], conv_stride[1],
                                              n_steps, conv_n_maps[1], 1, false));
    conv_layers.back().config_neurons(conv_th[1], 1.0, 0.0, 0.0, ap[1], am[1]);
    get_output_dimensions(conv_layers.back(), p_output_d, p_output_h, p_output_w);

    pool_layers.push_back(Pooling_layer(p_output_d, p_output_h, p_output_w,
                                        pool_window_size[1], pool_stride[1], n_steps, false));
    get_output_dimensions(pool_layers.back(), p_output_d, p_output_h, p_output_w);
    
    conv_layers.push_back(Convolutional_layer(p_output_d, p_output_h, p_output_w,
                                              conv_window_size[2], conv_stride[2],
                                              n_steps, conv_n_maps[2], 1, false));
    conv_layers.back().config_neurons(conv_th[2], 1.0, 0.0, 0.0, ap[2], am[2]);

    printf("Trying best \nNet conf:\n%.2lf, %.1lf", dog_p_th, dog_n_th);
    for (unsigned i = 0; i < conv_layers.size(); ++i)
        printf(", %.1lf, %.4lf", conv_th[i], ap[i]);
    printf(", %d\n", update_ap_step);
}

Network::Network(const std::string& file_name) {

    std::vector<std::string> vs = get_vector_str_layers(file_name);
    dog_layer = DoG_layer(vs[1]);
    conv_layers.clear();
    pool_layers.clear();
    n_steps = dog_layer.get_n_steps();
    for (unsigned l = 2; l < vs.size(); ++l) {
        if (l & 1) pool_layers.push_back(Pooling_layer(vs[l]));
        else conv_layers.push_back(Convolutional_layer(vs[l]));
    }
}

std::vector<std::string> Network::get_vector_str_layers(const std::string& file_name) {

    std::string desc = read_from_file(file_name);

    std::vector<std::string> vs = split_str(desc, "\n");
    if (vs[0] != NET_NAME) {
        printf("\nDescription = %s. It should be %s\n", vs[0].c_str(), NET_NAME.c_str());
        throw std::runtime_error("ERROR IN NET: Wrong description.");
    }
    return vs;
}

template<class T>
void Network::get_output_dimensions(T layer, int& p_output_d, int& p_output_h, int& p_output_w) {

    p_output_d = layer.get_nm_l();
    p_output_h = layer.get_nm_h();
    p_output_w = layer.get_nm_w();
}

vvvvd Network::get_all_potentials() const {

    vvvvd potentials;
    for (unsigned i = 0; i < conv_layers.size(); ++i) {
        potentials.push_back(conv_layers[i].get_potentials());
        if (i < pool_layers.size())
            potentials.push_back(pool_layers[i].get_potentials());
    }
    return potentials;
}





// For training only





void Network::unsupervised_learning(const vvvi& train_dataset, const vi& train_labels,
                                    std::string& out_str, int train_n_imgs) {

    time_t start;
    time(&start);

    printf("\nUnsupervised Learning\n");
    learning_layer = 0;
    for(auto& cl : conv_layers) { cl.set_has_th(true); }
    forward_phase(train_dataset, train_n_imgs);

    print_time_and_text(start, "Training time:");
}

void Network::forward_phase(const vvvi& train_dataset, int train_n_imgs) {

    int iter = 0, n_imgs = static_cast<int>(train_dataset.size());
    max_dog_spikes = -1, min_dog_spikes = INT_MAX, mean_dog_spikes = 0;
    std::vector<std::string> Cl_str(conv_layers.size(), "");
    conv_layers[learning_layer].learning = true;
    printf("\nForward Phase:\n\n[Conv_0 is learning]\n\n");

    bool fit = false;
    for (; iter < train_n_imgs && !fit; ++iter) {
        for(auto& cl : conv_layers) { cl.reset(); }
        for(auto& pl : pool_layers) { pl.reset(); }
        if( !((iter + 1) % std::max(train_n_imgs/100, 1)) ) { //std::max to avoid 0 remainder
            printf("\tForward: %5d/%5d\r", iter + 1, train_n_imgs); fflush(stdout);
        }
        double layer_CL = conv_layers[learning_layer].get_Cl();
        Cl_str[learning_layer] += to_str(layer_CL) + ",";
        if (iter && !(iter % update_ap_step)) { conv_layers[learning_layer].update_ap(max_ap); }
        fit = run_steps(train_dataset[iter % n_imgs], true);
    }
    if (fit) { printf("\nNet converged after %d iterations\n", iter); }
    printf("\nStatus:\n");
    printf("\nMax_dog_spikes = %d;\nMin_dog_spikes = %d;\nMean_dog_spikes = %.2lf\n\n",
        max_dog_spikes, min_dog_spikes, mean_dog_spikes /= iter);
    for (unsigned i = 0; i < Cl_str.size(); ++i) {
        write_on_file(to_str(i) + "_Cl.csv", Cl_str[i]);
        printf("C%u = %lf; spikes = %d\n", i, conv_layers[i].get_Cl(),
                conv_layers[i].get_spikes_until_convergence());
    }
}





// For training and testing





// Return true if the net converged (all conv layers converged)
bool Network::run_steps(const vvi& img, bool is_learning) {

    Spike_train_list spike_train_input {dog_layer.output(img)};
    int tot = 0;
    for (auto& st : spike_train_input) tot += st.size();
    mean_dog_spikes += tot;
    max_dog_spikes = std::max(max_dog_spikes, tot);
    min_dog_spikes = std::min(min_dog_spikes, tot);
    for(int step = 0; step < static_cast<int>(spike_train_input.size()); ++step) {
        forward(spike_train_input[step], step);
        if (is_learning) {
            double layer_CL = conv_layers[learning_layer].get_Cl();    
            if(layer_CL < CONVERGENCE) {
                int spikes_conv = conv_layers[learning_layer].get_spikes_until_convergence();
                int ite_conv = conv_layers[learning_layer].get_iterations_until_convergence();
                printf("\n\tConv_%d converged after %d iterations at step %d and produced %d "
                       "spikes; mean = %lf;\n\tC%d = %lf; ", learning_layer,
                        ite_conv, step, spikes_conv, 1.0 * spikes_conv / ite_conv,
                        learning_layer, layer_CL);
                conv_layers[learning_layer].learning = false;
                ++learning_layer;
                if (learning_layer < static_cast<int>(conv_layers.size())) {
                    conv_layers[learning_layer].learning = true;
                    printf("\n[Conv_%d is learning]\n\n", learning_layer);
                } else { return true; }
            }
        }
    }
    return false;
}

// Propagate the spike train through the net
void Network::forward(Spike_train& spike_train_input, int step) {

    for(int idx = 0; idx <= learning_layer; ++idx) {
        if (!step && idx == learning_layer && conv_layers[learning_layer].learning)
            conv_layers[learning_layer].increment_iterations_until_convergence();
        spike_train_input = conv_layers[idx].output(spike_train_input, step);
        if (learning_layer > 0 && idx < static_cast<int>(pool_layers.size()))
            spike_train_input = pool_layers[idx].output(spike_train_input, step);
    }
}





// For testing only





void Network::global_pooling_phase(const vvvi& dataset, const vi& labels,
                                   std::string& out_str, int n_imgs) {
    int iter = 0;
    learning_layer = conv_layers.size() - 1;
    printf("\n\nApplying Global Pooling\n");
    for (auto& layer : conv_layers) layer.learning = false;
    conv_layers.back().set_has_th(false);
    std::string str = "";
    for (; iter < n_imgs; ++iter) {
        for(auto& l : conv_layers) { l.reset(); }
        for(auto& l : pool_layers) { l.reset(); }
        if(!((iter + 1) % (n_imgs/10))) {
            printf("\tGlobal pooling: %5d/%5d\r", iter + 1, n_imgs); fflush(stdout);
        }
        run_steps(dataset[iter], false);

        std::vector<Neuronal_map> n_map {conv_layers.back().get_maps()};
        vd pot {Pooling_layer::global_pooling(n_map)};
        str += vd_to_str(pot) + "," + to_str(labels[iter]) + "\n";
    }
    write_on_file(out_str, str);
}

void Network::unsupervised_testing(const vvvi& train_dataset, const vi& train_labels,
                                   const vvvi& test_dataset, const vi& test_labels,
                                   std::string& train_out_str, std::string& test_out_str,
                                   int train_n_imgs, int test_n_imgs) {
    time_t start;
    time(&start);
    global_pooling_phase(train_dataset, train_labels, train_out_str,
                         std::min(train_n_imgs, static_cast<int> (train_dataset.size())));
    print_time_and_text(start, "Global_pooling time over training samples:");

    printf("\n\nUnsupervised Testing\n");

    time(&start);
    global_pooling_phase(test_dataset, test_labels, test_out_str,
                         std::min(test_n_imgs, static_cast<int> (test_dataset.size())));
    print_time_and_text(start, "Global_pooling time over testing samples:");
}





// Description (to save file)





void Network::save(const std::string& file_name) {
    
    write_on_file(file_name, description());
    write_on_file("friendly_" + file_name, friendly_description());
}

std::string Network::description() const {
    
    std::string desc {NET_NAME + "\n"};
    desc += dog_layer.description() + "\n";
    for (unsigned i = 0; i < conv_layers.size(); ++i) {
        if (i) desc += "\n";
        desc += conv_layers[i].description();
        if (i < pool_layers.size())
            desc += "\n" + pool_layers[i].description();
    }
    return desc;
}

std::string Network::friendly_description() const {
    
    std::string desc {NET_NAME + " layers\n\n"};
    desc += dog_layer.friendly_description();

    for (unsigned i = 0; i < conv_layers.size(); ++i) {
        desc += "\n\n" + conv_layers[i].friendly_description(i);
        if (i < pool_layers.size())
            desc += "\n\n" + pool_layers[i].friendly_description(i);
    }
    return desc;
}
