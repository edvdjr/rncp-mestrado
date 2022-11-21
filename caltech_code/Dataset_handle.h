#ifndef DATASET_HANDLE_H
#define DATASET_HANDLE_H

#include <stdexcept>
#include <fstream>
#include <vector>

#include "Matrix.h"
#include "Util.h"
#include "File_handle.h"

vvvi read_images(const std::string &full_path, int height, int width);

vi read_labels(const std::string &full_path);

#endif