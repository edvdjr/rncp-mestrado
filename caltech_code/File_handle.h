#ifndef FILE_HANDLE_H
#define FILE_HANDLE_H

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include "Util.h"
#include "Matrix.h"

std::string read_from_file(const std::string& file_name);
void write_on_file(const std::string& file_name, const std::string& text,
     std::ios_base::openmode mode = std::ios::out | std::ios::trunc | std::ios::binary);
void write_on_file(const std::string& file_name, const vi& labelset,
     std::ios_base::openmode mode = std::ios::out | std::ios::trunc | std::ios::binary);
void write_on_file(const std::string& file_name, const vvvi& dataset,
     std::ios_base::openmode mode = std::ios::out | std::ios::trunc | std::ios::binary);
#endif