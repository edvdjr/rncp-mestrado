#include "Util.h"

std::vector<std::string> split_str(const std::string& s, const std::string& delimiter) {

    size_t next = 0, last = 0;
    std::string token;
    std::vector<std::string> vs;
    while ((next = s.find(delimiter, last)) != std::string::npos) {
        token = s.substr(last, next - last);
        vs.push_back(token);
        last = next + delimiter.size();
    }
    vs.push_back(s.substr(last));
    return vs;
}

std::vector<double> split_str_dbl(const std::string& s, const std::string& delimiter) {

    size_t next = 0, last = 0;
    std::string token;
    std::vector<double> vs;
    while ((next = s.find(delimiter, last)) != std::string::npos) {
        token = s.substr(last, next - last);
        vs.push_back(std::stod(token));
        last = next + delimiter.size();
    }
    vs.push_back(std::stod(s.substr(last)));
    return vs;
}

std::vector<int> split_str_int(const std::string& s, const std::string& delimiter) {

    size_t next = 0, last = 0;
    std::string token;
    std::vector<int> vs;
    while ((next = s.find(delimiter, last)) != std::string::npos) {
        token = s.substr(last, next - last);
        vs.push_back(std::stoi(token));
        last = next + delimiter.size();
    }
    vs.push_back(std::stoi(s.substr(last)));
    return vs;
}

std::vector<int> seconds_to_days(int n) {

    std::vector<int> t(4, 0);    
    t[0] = n / (24 * 3600);
 
    n = n % (24 * 3600);
    t[1] = n / 3600;
 
    n %= 3600;
    t[2] = n / 60 ;
 
    n %= 60;
    t[3] = n;

    return t;
}

void print_time_from_seconds(int n) {

    std::vector<int> t = seconds_to_days(n);
    if (t[0]) printf(" %d day%s",  t[0], t[0] > 1 ? "s" : "");
    if (t[1]) printf(" %d hour%s", t[1], t[1] > 1 ? "s" : "");
    if (t[2]) printf(" %d min%s",  t[2], t[2] > 1 ? "s" : "");
    if (t[3]) printf(" %d sec%s",  t[3], t[3] > 1 ? "s" : "");   
}

void print_time_and_text(time_t start, const std::string& text) {

    time_t end;
    time(&end);
    double t = difftime(end, start);
    printf("\n\t**%s", text.c_str());
    print_time_from_seconds(static_cast<int>(t));
    printf(" **\n\n");
}

bool between(int a, int b, int c) { return b >= a && b <= c; }