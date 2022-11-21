#include "DoG_layer.h"

const std::string DOG_NAME {"DoG"};

DoG_layer::DoG_layer(int _in_size_h, int _in_size_w, int _k_size,
                     int _stride, unsigned _n_steps, double _s1, double _s2,
                     double _p_th, double _n_th) :
    in_size_h {_in_size_h}, in_size_w {_in_size_w}, k_size {_k_size},
    stride {_stride}, n_steps {_n_steps}, s1 {_s1}, s2 {_s2},
    p_th {_p_th}, n_th {_n_th} {
    
    if (_k_size > _in_size_w || _k_size > _in_size_h)
        throw std::runtime_error("ERROR IN DOG: k_size > in_size");
    
    create_DoG_filter();
}

DoG_layer::DoG_layer(std::string desc) {

    std::vector<std::string> vs {split_str(desc)};
    if (vs[0] != DOG_NAME) {
        printf("\nDescription = %s. It should be %s\n", vs[0].c_str(), DOG_NAME.c_str());
        throw std::runtime_error("ERROR IN DOG: Wrong description.");
    }
    std::map<std::string, double> atts_map;
    for (int i = 1; i < static_cast<int>(vs.size()); ++i) {
        std::vector<std::string> atts {split_str(vs[i], ":")};
        try { atts_map[atts[0]] = std::stod(atts[1]); }
        catch (int e) { printf("ERROR IN DOG: stod.\n");}
    }
    set_attributes(atts_map);
}

void DoG_layer::set_attributes(std::map<std::string, double> atts_map) {

    if (atts_map["k_size"] > atts_map["in_size_h"] ||
        atts_map["k_size"] > atts_map["in_size_w"])
            throw std::runtime_error("ERROR IN DOG: k_size > in_size");
    in_size_h = atts_map["in_size_h"], in_size_w = atts_map["in_size_w"];
    k_size = atts_map["k_size"];
    stride = atts_map["stride"], n_steps = atts_map["n_steps"];
    s1 = atts_map["s1"], s2 = atts_map["s2"];
    p_th = atts_map["p_th"], n_th = atts_map["n_th"];;
    create_DoG_filter();
}

void DoG_layer::create_DoG_filter() {
    
    kernel.resize(k_size, vd(k_size));
    int lim = k_size / 2;
    for (int x = -lim; x <= lim; ++x)
        for (int y = -lim; y <= lim; ++y) {
            int xi = x + lim, yi = y + lim;
            kernel[xi][yi] = get_gauss_dist(x, y, s1) - get_gauss_dist(x, y, s2);
        }
}

double DoG_layer::get_gauss_dist(int x, int y, double sigma) {
    return exp(-(x*x + y*y) / (2*sigma*sigma)) / (2*M_PI*sigma*sigma); // 2D DoG
}





// Output





Spike_train_list DoG_layer::output(const vvi& input) {
    
    // printf("input:\n");
    // print(input);

    vvd activ {convolution(input, kernel, stride)};

    // printf("After convolution:\n");
    // print(activ);
    
    int a_h {static_cast<int>(activ.size())}, a_w {static_cast<int>(activ[0].size())};
    Spike_train train;
    for (int r = 0; r < a_h; ++r) for (int c = 0; c < a_w; ++c)
            if ((activ[r][c] >= p_th) || (activ[r][c] <= n_th))    
                train.push_back({ fabs(activ[r][c]), 0, r, c });
    return get_layer_output_from_train(train, n_steps);
}

// Divides the spikes in steps
Spike_train_list DoG_layer::get_layer_output_from_train(Spike_train& train, unsigned n_steps) {

    std::sort(train.begin(), train.end());
    unsigned t_size = train.size(), count = 0;
    unsigned packet_size = t_size / n_steps;
    if (!packet_size) { printf("\n>>>>> Less spikes than steps\n"); }
    Spike_train_list out(n_steps);
    for (unsigned step = 0; step < n_steps; ++step)
        for(unsigned i = 0; i < packet_size && count < t_size; ++i)
            out[step].push_back(train[count++]);
    return out;
}





// Description





std::string DoG_layer::description() const {

    std::string desc {DOG_NAME};
    desc += " in_size_h:" + to_str(in_size_h) + " in_size_w:" + to_str(in_size_w);
    desc += " k_size:"    + to_str(k_size);
    desc += " stride:"    + to_str(stride)    + " n_steps:"   + to_str(n_steps);
    desc += " s1:"        + to_str(s1)        + " s2:"        + to_str(s2);
    desc += " p_th:"      + to_str(p_th)      + " n_th:"      + to_str(n_th);
    return desc;
}

std::string DoG_layer::friendly_description() const {

    std::string desc {DOG_NAME + "\n   Attributes:"};
    desc += "\n\tin_size_h:" + to_str(in_size_h) + "\n\tin_size_w:" + to_str(in_size_w);
    desc += "\n\tk_size:"    + to_str(k_size);
    desc += "\n\tstride:"    + to_str(stride)    + "\n\tn_steps:"   + to_str(n_steps);
    desc += "\n\ts1:"        + to_str(s1)        + "\n\ts2:"        + to_str(s2);
    desc += "\n\tp_th:"      + to_str(p_th)      + "\n\tn_th:"      + to_str(n_th);
    desc += "\n\n" + friendly_kernel_to_string();
    return desc;
}

std::string DoG_layer::friendly_kernel_to_string() const {

    std::string kernel_str {"   Kernel:"};
    for (int i = 0; i < k_size; ++i) {
        kernel_str += "\n\t";
        for (int j = 0; j < k_size; ++j) {
            if (j) kernel_str += " ";
            kernel_str += to_str(kernel[i][j]);
        }
    }
    return kernel_str;
}