#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <time.h>

#define to_str(v) std::to_string(v)

std::vector<std::string> split_str(const std::string& s, const std::string& delimiter = " ");
std::vector<double> split_str_dbl(const std::string& s, const std::string& delimiter = " ");
std::vector<int> split_str_int(const std::string& s, const std::string& delimiter = " ");
std::vector<int> seconds_to_days(int n);
void print_time_from_seconds(int n);
void print_time_and_text(time_t start, const std::string& text);
bool between(int a, int b, int c);

#endif