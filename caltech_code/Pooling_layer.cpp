#include "Pooling_layer.h"

const std::string POOL_NAME = "Pooling_Layer";

Pooling_layer::Pooling_layer() {}

Pooling_layer::Pooling_layer(int _n_channels, int _in_size_h, int _in_size_w, int _k_size,
                             int _stride, int _n_steps, bool _stoch) :
    n_channels {_n_channels}, in_size_h {_in_size_h}, in_size_w {_in_size_w},
    k_size {_k_size}, stride {_stride}, n_steps {_n_steps}, stoch {_stoch} {
        config_neurons();
}





// Load from file





Pooling_layer::Pooling_layer(const std::string& desc) {

    std::map<std::string, double> atts_map;
    std::string kernel_line;

    std::vector<std::string> vs = split_str(desc, " ");
    if (vs[0] != POOL_NAME) {
        printf("\nDescription = %s. It should be %s\n", vs[0].c_str(), POOL_NAME.c_str());
        throw std::runtime_error("ERROR IN POOL: Wrong description.");
    }

    split_attributes_from(vs, atts_map, kernel_line);

    set_attributes(atts_map);
    config_neurons();
}

void Pooling_layer::split_attributes_from(std::vector<std::string>& vs,
                                          std::map<std::string, double>& atts_map,
                                          std::string& kernel_line) {

    for (unsigned i = 1; i < vs.size(); ++i) { //vs[0] is POOL_NAME
        std::vector<std::string> atts = split_str(vs[i], ":");
        try { atts_map[atts[0]] = std::stod(atts[1]); }
        catch (std::invalid_argument const&) {
            printf("ERROR IN POOL: stod. Got %s.\n", atts[1].c_str());
            throw 20;
        }
    }
}

void Pooling_layer::set_attributes(std::map<std::string, double>& atts_map) {

    if (atts_map["k_size"] > atts_map["in_size_h"] ||
        atts_map["k_size"] > atts_map["in_size_w"])
            throw std::runtime_error("ERROR IN POOL LAYER: k_size > in_size");
    n_channels = atts_map["n_channels"], stoch     = (atts_map["stoch"] > 0 ? true : false);
    in_size_h  = atts_map["in_size_h"],  in_size_w = atts_map["in_size_w"];
    k_size     = atts_map["k_size"],     stride    = atts_map["stride"];
    n_steps    = atts_map["n_steps"];
}

void Pooling_layer::config_neurons() {

    neuronal_maps.clear();
    for (int i = 0; i < n_channels; ++i) {
        Neuronal_map neuro_map;
        for (int r1 = 0, r2 = 0; r1 < in_size_h; r1 += stride, ++r2) {
            vN vn;
            for (int c1 = 0, c2 = 0; c1 < in_size_w; c1 += stride, ++c2) {
                vn.push_back(Neuron(1, k_size, k_size, r1, r2, c1, c2,
                                    n_steps, i, 1.0, 1.0, 0, 0, 0, 0, stoch));
                if (c1 + k_size >= in_size_w) break;
            }
            neuro_map.push_back(vn);
            if (r1 + k_size >= in_size_h) break;
        }
        neuronal_maps.push_back(neuro_map);
    }
    nm_size_h = static_cast<int>(neuronal_maps[0].size());
    nm_size_w = static_cast<int>(neuronal_maps[0][0].size());
}

void Pooling_layer::reset() {
    for(auto& nm : neuronal_maps) for(vN& vn : nm) for (auto& n : vn) n.reset();
}

vvvd Pooling_layer::get_potentials() const {

    vvvd potentials;
    for (auto& neuro_map : neuronal_maps) {
        potentials.push_back({});
        for (auto& vn : neuro_map)
            for (auto& n : vn) potentials.back().push_back(n.get_potentials());
    }
    return potentials;
}





// Output





Spike_train Pooling_layer::output(const Spike_train& in_train, int step) {

    Spike_train out;
    for (auto& spike : in_train) {
        int sch {spike.ch}, sr {spike.r}, sc {spike.c};
        int r_min {static_cast<int>(ceil((sr + 1.0 - k_size) / stride))};
        int r_max {static_cast<int>(floor((sr * 1.0) / stride))};
        int c_min {static_cast<int>(ceil((sc + 1.0 - k_size) / stride))};
        int c_max {static_cast<int>(floor((sc * 1.0) / stride))};
        for (int r = std::max(r_min, 0); r <= std::min(r_max, nm_size_h - 1); ++r)
            for (int c = std::max(c_min, 0); c <= std::min(c_max, nm_size_w - 1); ++c) {
                Neuron& n {neuronal_maps[sch][r][c]};
                int w_r {sr - n.get_r1()};
                int w_c {sc - n.get_c1()};
                double w {1.0};
                Spike spk {0, 0, w_r, w_c};
                n.stimulate(spk, w, step);
                n.fire(step);
                out.push_back(n.get_spike());
            }
    }
    return out;
}

vd Pooling_layer::global_pooling(std::vector<Neuronal_map>& nm) {

    int n_ch = static_cast<int>(nm.size()), n_r = static_cast<int>(nm[0].size());
    int n_c = static_cast<int>(nm[0][0].size());
    vd potentials(n_ch);
    for (int i = 0; i < n_ch; ++i) {
        double max_pot = 0.0;
        for (int j = 0; j < n_r; ++j)
            for (int k = 0; k < n_c; ++k) {
                double last_pot = nm[i][j][k].get_potentials().back();
                max_pot = std::fmax(max_pot, last_pot);
            }
        potentials[i] = max_pot;
    }
    return potentials;
}





// Description (To save file)





std::string Pooling_layer::description() const {

    std::string atts {POOL_NAME};
    atts += " n_channels:" + to_str(n_channels) + " stoch:" + (stoch ? to_str('1'):to_str('0'));
    atts += " in_size_h:"  + to_str(in_size_h)  + " in_size_w:" + to_str(in_size_w);
    atts += " k_size:"     + to_str(k_size)     + " stride:"    + to_str(stride);
    atts += " n_steps:"    + to_str(n_steps);
    
    return atts;
}

// Friendly Description (To present to user)
std::string Pooling_layer::friendly_description(int num) const {

    std::string desc {POOL_NAME + "_" + to_str(num) + "\n"};
    desc += friendly_attributes_to_string() + "\n\n";
    desc += friendly_neurons_to_string() + "\n\n";
    desc += friendly_kernels_to_string();
    
    return desc;
}

std::string Pooling_layer::friendly_attributes_to_string() const {

    std::string atts {"   Attributes:"};
    atts += "\n\tin_size_h:" + to_str(in_size_h) + "\n\tin_size_w:" + to_str(in_size_w);
    atts += "\n\tk_size:"    + to_str(k_size)    + "\n\tstride:"    + to_str(stride);
    atts += "\n\tn_steps:"   + to_str(n_steps)   + "\n\tstoch:" + (stoch ? "true" : "false");
    
    return atts;
}

std::string Pooling_layer::friendly_neurons_to_string() const {

    Neuron f_neuron {neuronal_maps[0][0][0]};

    std::string neurons_str {"   Neuron configuration:"};
    neurons_str += "\n\tth:" + to_str(f_neuron.get_th()) + "\n\tmu:" + to_str(f_neuron.get_mu());
    neurons_str += "\n\tVb:" + to_str(f_neuron.get_Vb()) + "\n\tVr:" + to_str(f_neuron.get_Vr());
    neurons_str += "\n\tap:" + to_str(f_neuron.get_ap()) + "\n\tam:" + to_str(f_neuron.get_am());
    
    return neurons_str;
}

std::string Pooling_layer::friendly_kernels_to_string() const {

    std::string kernels_str {"   All kernels equal to:"};
    for (int k = 0; k < k_size; ++k) {
        kernels_str += "\n\t";
        for (int l = 0; l < k_size; ++l) {
            if (l) kernels_str += " ";
            kernels_str += "1";
        }
    }
    return kernels_str;
}
