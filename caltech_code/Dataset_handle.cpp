#include "Dataset_handle.h"

vvvi read_images(const std::string &full_path, int height, int width) {

    std::string desc = read_from_file(full_path);
    vvvi v_mtx;
    long unsigned int i = 0;
    std::vector<std::string> vl = split_str(desc, "\n");
    for (; i < vl.size(); ++i) {
        vi vs {split_str_int(vl[i], ";")};
        vvi mtx(height);
        for (int l = 0, n = 0; l < height; ++l)
            for (int c = 0; c < width; ++c)
                mtx[l].push_back(vs[n++]);
        v_mtx.push_back(mtx);
        // if(!(i % (vl.size()/10))) {
        //     printf("\tConverting to matrix: %5lu/%5lu\r", i + 1, vl.size()); fflush(stdout);
        // }
    }
    // printf("\tConverting to matrix: %5lu/%5lu\n", i, vl.size());
    return v_mtx;    
}

vi read_labels(const std::string &full_path) {

    std::string all_labels = read_from_file(full_path);
    std::vector<std::string> vl = split_str(all_labels, "\n");
    vi int_labels(vl.size(), 0);
    for (long unsigned int i = 0; i < vl.size(); ++i) {
        if (vl[i] == "Motor") { int_labels[i] = 1; }
    }
    return int_labels;
}