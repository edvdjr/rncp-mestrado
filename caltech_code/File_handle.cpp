#include "File_handle.h"

const std::string out_path = "out/";

std::string read_from_file(const std::string& file_name) {

    std::string line, text;
    std::ifstream myfile(file_name);
    if (myfile.is_open()) {
        bool first = true;
        while (getline (myfile, line)) {
            if (!first) text += "\n";
            text += line;
            first = false;
        }
        myfile.close();
    } else throw std::runtime_error("ERROR: Unable to open file");
    
    return text;
}

void write_on_file(const std::string& file_name, const std::string& text, std::ios_base::openmode mode) {

    std::ofstream myfile (out_path + file_name, mode);
    if (myfile.is_open()) {
        myfile << text;
        myfile.close();
    } else throw std::runtime_error("ERROR: Unable to open file");
}

void write_on_file(const std::string& file_name, const vi& labelset, std::ios_base::openmode mode) {

    std::ofstream myfile (file_name, mode);
    if (myfile.is_open()) {
        for (unsigned i = 0; i < labelset.size(); ++i) {
            if (i) { myfile << "\n"; }
            myfile << to_str(labelset[i]);
        }
        myfile.close();
    } else throw std::runtime_error("ERROR: Unable to open file");
}

void write_on_file(const std::string& file_name, const vvvi& dataset, std::ios_base::openmode mode) {

    std::ofstream myfile (file_name, mode);
    if (myfile.is_open()) {
        for (unsigned i = 0; i < dataset.size(); ++i) {
            if (i) { myfile << "\n"; }
            for (unsigned j = 0; j < dataset[0].size(); ++j) {
                if (j) { myfile << ";"; }
                for (unsigned k = 0; k < dataset[0][0].size(); ++k) {
                    if (k) { myfile << ";"; }
                    myfile << to_str(dataset[i][j][k]);
                }
            }
        }
        myfile.close();
    } else throw std::runtime_error("ERROR: Unable to open file");
}