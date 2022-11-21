#include "Convolutional_layer.h"

const std::string CONV_NAME = "Convolutional_Layer";

Convolutional_layer::Convolutional_layer() {}

Convolutional_layer::Convolutional_layer(int _n_channels, int _in_size_h, int _in_size_w,
                                         int _k_size, int _stride, int _n_steps, int _n_kernels,
                                         int _inhibit_w, bool _stoch) :
    learning {false}, n_channels {_n_channels}, in_size_h {_in_size_h}, in_size_w {_in_size_w},
    k_size {_k_size}, stride {_stride}, n_steps {_n_steps}, n_kernels {_n_kernels},
    inhibit_w {_inhibit_w}, stoch {_stoch}, iterations_until_convergence {0},
    spikes_until_convergence {0} {

    vvvvd _kernels;
    for (int i = 0; i < _n_kernels; ++i)
        _kernels.push_back(get_random_3D_vector(_n_channels, _k_size, _k_size));
    set_kernels(_kernels);
    STDPed_maps.clear();
    STDPed_maps.resize(_n_kernels, false);
}





// Load from file





Convolutional_layer::Convolutional_layer(const std::string& desc) {

    learning = false;

    std::vector<std::string> vs = split_str(desc, " ");
    if (vs[0] != CONV_NAME) {
        printf("\nDescription = %s. It should be %s\n", vs[0].c_str(), CONV_NAME.c_str());
        throw std::runtime_error("ERROR IN CONV: Wrong description.");
    }

    std::map<std::string, double> atts_map;
    std::string kernels_line;

    split_attributes_from(vs, atts_map, kernels_line);
    set_attributes(atts_map);
    set_neurons(atts_map);
    set_kernels(kernels_line);
    STDPed_maps.clear();
    STDPed_maps.resize(n_kernels, false);
}

void Convolutional_layer::split_attributes_from(std::vector<std::string>& vs,
                                                std::map<std::string, double>& atts_map,
                                                std::string& kernels_line) {

    for (unsigned i = 1; i < vs.size(); ++i) { //vs[0] is CONV_NAME
        std::vector<std::string> atts = split_str(vs[i], ":");
        if (atts[0] == "kernels") kernels_line = atts[1];
        else {
            try { atts_map[atts[0]] = std::stod(atts[1]); }
            catch (std::invalid_argument const&) {
                printf("ERROR IN CONV: stod. Got %s.\n", atts[1].c_str());
                throw 20;
            }
        }
    }
}

void Convolutional_layer::set_attributes(std::map<std::string, double>& atts_map) {

    if (atts_map["k_size"] > atts_map["in_size_h"] ||
        atts_map["k_size"] > atts_map["in_size_w"])
            throw std::runtime_error("ERROR IN CONV LAYER: k_size > in_size");
    n_channels = atts_map["n_channels"], stoch     = (atts_map["stoch"] > 0 ? true : false);
    in_size_h  = atts_map["in_size_h"],  in_size_w = atts_map["in_size_w"];
    k_size     = atts_map["k_size"],     k_size    = atts_map["k_size"];
    stride     = atts_map["stride"],     n_steps   = atts_map["n_steps"];
    n_kernels  = atts_map["n_kernels"],  inhibit_w = atts_map["inhibit_w"];
}

void Convolutional_layer::set_neurons(std::map<std::string, double>& atts_map) {

    double th {atts_map["th"]}, mu {atts_map["mu"]}, Vb {atts_map["Vb"]};
    double Vr {atts_map["Vr"]}, ap {atts_map["ap"]}, am {atts_map["am"]};

    config_neurons(th, mu, Vb, Vr, ap, am);
}

void Convolutional_layer::set_kernels(const std::string& line) {

    int n {0};
    std::vector<double> vs {split_str_dbl(line, ";")};
    kernels.clear();
    kernels.resize(n_kernels, vvvd(n_channels, vvd(k_size, vd(k_size))));
    for (int i = 0; i < n_kernels; ++i)
        for (int j = 0; j < n_channels; ++j)
            for (int k = 0; k < k_size; ++k)
                for (int l = 0; l < k_size; ++l)
                    kernels[i][j][k][l] = vs[n++];
}

void Convolutional_layer::config_neurons(double th, double mu, double Vb,
                                         double Vr, double ap, double am) {
    neuronal_maps.clear();
    for (int i = 0; i < n_kernels; ++i) {
        Neuronal_map neuro_map;
        for (int r1 = 0, r2 = 0; r1 < in_size_h; r1 += stride, ++r2) {
            vN vn;
            for (int c1 = 0, c2 = 0; c1 < in_size_w; c1 += stride, ++c2) {
                vn.push_back(Neuron(n_channels, k_size, k_size, r1, r2, c1, c2,
                                    n_steps, i, th, mu, Vb, Vr, ap, am, stoch));
                if (c1 + k_size >= in_size_w) break;
            }
            neuro_map.push_back(vn);
            if (r1 + k_size >= in_size_h) break;
        }
        neuronal_maps.push_back(neuro_map);
    }
    nm_size_h = static_cast<int>(neuronal_maps[0].size());
    nm_size_w = static_cast<int>(neuronal_maps[0][0].size());
    allowed_to_fire.clear();
    allowed_to_fire.resize(n_kernels, vvb(nm_size_h, vb(nm_size_w, true)));
}

void Convolutional_layer::reset() {

    for(auto& nm : neuronal_maps) for(vN& vn : nm) for (auto& n : vn) n.reset();
    for(int i = 0; i < n_kernels; ++i) STDPed_maps[i] = false;
    allowed_to_fire.clear();
    allowed_to_fire.resize(n_kernels, vvb(nm_size_h, vb(nm_size_w, true)));
}

vvvd Convolutional_layer::get_potentials() const {

    vvvd potentials;
    for (auto& neuro_map : neuronal_maps) {
        potentials.push_back({});
        for (auto& vn : neuro_map)
            for (auto& n : vn) potentials.back().push_back(n.get_potentials());
    }
    return potentials;
}





// Output





Spike_train Convolutional_layer::output(const Spike_train& in_train, int step) {

    Spike_train out;
    vNP winners;
    vvvb stimulated (n_kernels, vvb(nm_size_h, vb(nm_size_w, false)));
    for (auto& spike : in_train) { receive_spike(spike, stimulated, winners, step); }
    sort(winners.begin(), winners.end(), std::greater<N_params>());
    competition(winners, out, step);
    update_idle_neurons(stimulated, step);
    return out;
}

// r_min, r_max, c_min and c_max are the bounds of the region on
// neuronal map that is affected by the spike, since one spike can
// stimulate multiple neurons
void Convolutional_layer::receive_spike(const Spike& spike, vvvb& stimulated,
                                        vNP& winners, int step) {

    int sch {spike.ch}, sr {spike.r}, sc {spike.c};
    int r_min {static_cast<int>(ceil((sr + 1.0 - k_size) / stride))};
    int r_max {static_cast<int>(floor((sr * 1.0) / stride))};
    int c_min {static_cast<int>(ceil((sc + 1.0 - k_size) / stride))};
    int c_max {static_cast<int>(floor((sc * 1.0) / stride))};
    for (int k = 0; k < n_kernels; ++k) {
        for (int r = std::max(r_min, 0); r <= std::min(r_max, nm_size_h - 1); ++r)
            for (int c = std::max(c_min, 0); c <= std::min(c_max, nm_size_w - 1); ++c)
                if (allowed_to_fire[k][r][c]) {
                    Neuron& n {neuronal_maps[k][r][c]};
                    int w_r {sr - n.get_r1()}, w_c {sc - n.get_c1()};
                    double w {kernels[k][sch][w_r][w_c]};
                    Spike spk {0, sch, w_r, w_c};
                    n.stimulate(spk, w, step);
                    stimulated[k][r][c] = true;
                    if (n.check_threshold_at_step(step))
                        winners.push_back({k, r, c, (n.get_potentials()) [step]});
                }
    }
}

void Convolutional_layer::competition(const vNP& winners, Spike_train& out, int step) {

    for (auto& w : winners) {
        int k = w.k, r2 = w.r2, c2 = w.c2;
        Neuron& n {neuronal_maps[k][r2][c2]};
        if (!allowed_to_fire[k][r2][c2]) continue;
        n.fire(step);
        out.push_back(n.get_spike());
        if (learning) {
            if (!STDPed_maps[k]) {
                n.stdp(kernels[k]);
                STDPed_maps[k] = true;
                for(vN& vn : neuronal_maps[k]) for (auto& n : vn) n.reset();
            }
            ++spikes_until_convergence;
        }
        for (int i = 0; i < n_kernels; ++i) { allowed_to_fire[i][r2][c2] = false; }
    }
}

// Neurons that weren't stimulated may have their membrane
// potential decreased due to leaky current
void Convolutional_layer::update_idle_neurons(const vvvb& stimulated, int step) {
    
    for(int i = 0; i < n_kernels; ++i) {
        if (STDPed_maps[i]) continue;
        for (int j = 0; j < nm_size_h; ++j)
            for (int k = 0; k < nm_size_w; ++k)
                if (!stimulated[i][j][k] && !(neuronal_maps[i][j][k].has_fired()))
                    neuronal_maps[i][j][k].dont_stimulate(step);
    }
}

void Convolutional_layer::set_has_th(bool has_th) {
    
    for(auto& nm : neuronal_maps) for(auto& vn : nm) for (auto& n : vn) { n.has_th = has_th; }
}

// Convergence level (Cl) is used to know if a layer has achieved a good level of learning
double Convolutional_layer::get_Cl() {

    double Cl {0};
    for (auto f:kernels) for (auto ch:f) for (auto r:ch) for (auto c:r) { Cl += c*(1-c); }
    return Cl / (n_kernels * n_channels * k_size * k_size);
}

void Convolutional_layer::update_ap(double max_ap) {

    for(auto& nm : neuronal_maps) for(vN& vn : nm) for (auto& n : vn) { n.update_ap(max_ap); }
}





// Description (to save file)





std::string Convolutional_layer::description() const {

    std::string desc {CONV_NAME};
    desc += attributes_to_string();
    desc += kernels_to_string();
    desc += neurons_to_string();
    return desc;
}

std::string Convolutional_layer::attributes_to_string() const {

    std::string atts {" n_channels:" + to_str(n_channels)};
    atts += " in_size_h:" + to_str(in_size_h) + " in_size_w:"  + to_str(in_size_w);
    atts += " k_size:"    + to_str(k_size);
    atts += " stride:"    + to_str(stride)    + " n_steps:"    + to_str(n_steps);
    atts += " n_kernels:" + to_str(n_kernels) + " inhibit_w:"  + to_str(inhibit_w);
    atts += " stoch:" + (stoch ? to_str('1') : to_str('0'));
    
    return atts;
}

std::string Convolutional_layer::kernels_to_string() const {

    std::string kernels_str {" kernels:"};
    for (int i = 0; i < n_kernels; ++i)
        for (int j = 0; j < n_channels; ++j)
            for (int k = 0; k < k_size; ++k)
                for (int l = 0; l < k_size; ++l) {
                    if (i || j || k || l) kernels_str += ";";
                    kernels_str += to_str(kernels[i][j][k][l]);
                }
    return kernels_str;
}

std::string Convolutional_layer::neurons_to_string() const {

    Neuron f_neuron {neuronal_maps[0][0][0]};

    std::string neurons_str;
    neurons_str += " th:"  + to_str(f_neuron.get_th()) + " mu:" + to_str(f_neuron.get_mu());
    neurons_str += " Vb:"  + to_str(f_neuron.get_Vb()) + " Vr:" + to_str(f_neuron.get_Vr());
    neurons_str += " ap:"  + to_str(f_neuron.get_ap()) + " am:" + to_str(f_neuron.get_am());
    
    return neurons_str;
}





// Friendly Description (To present to user)





std::string Convolutional_layer::friendly_description(int num) const {

    std::string desc {CONV_NAME + "_" + to_str(num) + "\n"};
    desc += friendly_attributes_to_string() + "\n\n";
    desc += friendly_neurons_to_string() + "\n\n";
    desc += friendly_kernels_to_string();
    
    return desc;
}

std::string Convolutional_layer::friendly_attributes_to_string() const {

    std::string atts {"   Attributes:"};
    atts += "\n\tn_channels:" + to_str(n_channels);
    atts += "\n\tin_size_h:"  + to_str(in_size_h) + "\n\tin_size_w:" + to_str(in_size_w);
    atts += "\n\tk_size:"     + to_str(k_size)    + "\n\tstride:"    + to_str(stride);
    atts += "\n\tn_steps:"    + to_str(n_steps)   + "\n\tn_kernels:" + to_str(n_kernels);
    atts += "\n\tinhibit_w:"  + to_str(inhibit_w) + "\n\tstoch:" + (stoch ? "true" : "false");
    
    return atts;
}

std::string Convolutional_layer::friendly_neurons_to_string() const {

    Neuron f_neuron {neuronal_maps[0][0][0]};

    std::string neurons_str {"   Neuron configuration:"};
    neurons_str += "\n\tth:" + to_str(f_neuron.get_th()) + "\n\tmu:" + to_str(f_neuron.get_mu());
    neurons_str += "\n\tVb:" + to_str(f_neuron.get_Vb()) + "\n\tVr:" + to_str(f_neuron.get_Vr());
    neurons_str += "\n\tap:" + to_str(f_neuron.get_ap()) + "\n\tam:" + to_str(f_neuron.get_am());
    
    return neurons_str;
}

std::string Convolutional_layer::friendly_kernels_to_string() const {

    std::string kernels_str {"   Kernels:"};
    for (int i = 0; i < n_kernels; ++i) {
        kernels_str += "\n\n\tKernel_" + to_str(i) + "\n\t";
        for (int j = 0; j < n_channels; ++j) {
            if (j) kernels_str += "\n\n\t";
            for (int k = 0; k < k_size; ++k) {
                if (k) kernels_str += "\n\t";
                for (int l = 0; l < k_size; ++l) {
                    if (l) kernels_str += " ";
                    kernels_str += to_str(kernels[i][j][k][l]);
                }
            }
        }
    }
    return kernels_str;
}
