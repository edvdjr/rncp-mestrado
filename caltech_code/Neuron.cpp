#include "Neuron.h"

Neuron::Neuron(int _n_channels, int _k_size_h, int _k_size_w, int _r1, int _r2,
               int _c1, int _c2, int _n_steps, int _i_kernel, double _th,
               double _mu, double _Vb, double _Vr, double _ap, double _am, bool _stoch) :
    has_th {true}, n_channels {_n_channels}, k_size_h {_k_size_h}, k_size_w {_k_size_w},
    r1 {_r1}, r2 {_r2}, c1 {_c1}, c2 {_c2}, n_steps {_n_steps}, i_kernel {_i_kernel},
    th {_th}, mu {_mu}, Vb {_Vb}, Vr {_Vr}, ap {_ap}, am {_am}, stoch {_stoch} {
        reset(); }

void Neuron::reset() {
    
    fired = false;
    f_time = -1;
    V.clear();
    V.resize(n_steps, Vb);
    potentiated.clear();
    potentiated.resize(n_channels, vvb(k_size_h, vb(k_size_w, false)));
}

void Neuron::stimulate(Spike& spike, double w, int step) {

    V[step] = mu * (V[step] - Vb) + Vb + w;
    for (unsigned i = step + 1; i < V.size(); ++i) V[i] = V[i - 1];
    potentiated[spike.ch][spike.r][spike.c] = true;
}

bool Neuron::check_threshold_at_step(int step) {

    if (!has_th) return false;
    double activation {V[step]}, threshold {th};
    if (stoch) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0, 1.0);
        threshold = dis(gen);
        activation = 1.0 / (1.0 + exp(-V[step] + th));
    }
    return activation >= threshold;
}

// Apply the current leakage
void Neuron::dont_stimulate(int step) { 

    if (step) {
        V[step] = mu * (V[step - 1] - Vb) + Vb;
        for (unsigned i = step + 1; i < V.size(); ++i) V[i] = V[i - 1];
    }
}

void Neuron::fire(int step) { f_time = step; fired = true; }

// Only for conv neurons
void Neuron::stdp(vvvd& w) {

    for (int i = 0; i < n_channels; ++i)
        for (int j = 0; j < k_size_h; ++j)
            for (int k = 0; k < k_size_w; ++k)
                w[i][j][k] += w[i][j][k] * (1.0 - w[i][j][k]) * (potentiated[i][j][k] ? ap : am);
}